"""Chat session handles for the LLM component."""

from __future__ import annotations

import abc
import asyncio
from collections.abc import AsyncIterator

from trillim.components.llm._engine import (
    EngineCrashedError,
    EngineError,
    EngineProgressTimeoutError,
)
from trillim.components.llm._events import (
    ChatDoneEvent,
    ChatEvent,
    ChatFinalTextEvent,
    ChatTokenEvent,
    ChatUsage,
)
from trillim.components.llm._limits import SESSION_TOKEN_LIMIT
from trillim.components.llm._validation import (
    SamplingOptions,
    validate_messages,
    validate_sampling_options,
)
from trillim.harnesses.search.provider import SearchAuthenticationError
from trillim.errors import (
    ContextOverflowError,
    ProgressTimeoutError,
    SessionBusyError,
    SessionClosedError,
    SessionExhaustedError,
    SessionStaleError,
)


_CHAT_SESSION_OWNER_TOKEN = object()
_CHAT_SESSION_CONSTRUCTION_ERROR = (
    "ChatSession cannot be constructed directly; use LLM.open_session()"
)
_ALLOW_CHAT_SESSION_SUBCLASS = False


def _create_chat_session(llm, messages) -> _ChatSession:
    return _ChatSession(llm, messages, _owner_token=_CHAT_SESSION_OWNER_TOKEN)


class ChatSession(abc.ABC):
    """Public chat-session handle returned by the LLM component."""

    _runtime_proxy = True

    def __init_subclass__(cls, **kwargs) -> None:
        del kwargs
        super().__init_subclass__()
        if not _ALLOW_CHAT_SESSION_SUBCLASS:
            raise TypeError("ChatSession cannot be subclassed publicly")

    def __new__(cls, *args, **kwargs):
        del args, kwargs
        if cls is ChatSession:
            raise TypeError(_CHAT_SESSION_CONSTRUCTION_ERROR)
        return super().__new__(cls)

    @property
    @abc.abstractmethod
    def state(self) -> str:
        """Return the current session state."""
        ...  # pragma: no cover

    @property
    @abc.abstractmethod
    def messages(self) -> tuple[dict[str, str], ...]:
        """Return a copy of the canonical session messages."""
        ...  # pragma: no cover

    @property
    @abc.abstractmethod
    def cached_token_count(self) -> int:
        """Return the last committed cached token count."""
        ...  # pragma: no cover

    @abc.abstractmethod
    async def __aenter__(self) -> ChatSession:
        """Enter the session as an async context manager."""
        ...  # pragma: no cover

    @abc.abstractmethod
    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Close the session when leaving an async context manager."""
        ...  # pragma: no cover

    @abc.abstractmethod
    async def close(self) -> None:
        """Close the session and cancel any active turn."""
        ...  # pragma: no cover

    @abc.abstractmethod
    def add_user(self, content: str) -> None:
        """Append a user message to the session."""
        ...  # pragma: no cover

    @abc.abstractmethod
    def add_system(self, content: str) -> None:
        """Append a system message to the session."""
        ...  # pragma: no cover

    @abc.abstractmethod
    async def chat(
        self,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        rep_penalty_lookback: int | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Collect the next assistant turn as a single string."""
        ...  # pragma: no cover

    @abc.abstractmethod
    async def stream_chat(
        self,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        rep_penalty_lookback: int | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ChatEvent]:
        """Stream structured events for the next assistant turn."""
        ...  # pragma: no cover


_ALLOW_CHAT_SESSION_SUBCLASS = True


class _ChatSession(ChatSession):
    """Private concrete chat-session implementation owned by one LLM runtime."""

    def __new__(cls, llm=None, messages=None, *, _owner_token=None):
        del llm, messages
        if _owner_token is not _CHAT_SESSION_OWNER_TOKEN:
            raise TypeError(_CHAT_SESSION_CONSTRUCTION_ERROR)
        return super().__new__(cls)

    def __init__(self, llm=None, messages=None, *, _owner_token=None) -> None:
        if _owner_token is not _CHAT_SESSION_OWNER_TOKEN:
            raise TypeError(_CHAT_SESSION_CONSTRUCTION_ERROR)
        if llm is None or messages is None:
            raise TypeError(_CHAT_SESSION_CONSTRUCTION_ERROR)
        self._llm = llm
        self._messages = [
            {"role": message.role, "content": message.content}
            for message in messages
        ]
        self._cached_token_count = 0
        self._state = "open"
        self._stale = False
        self._consumer_active = False
        self._active_task: asyncio.Task | None = None
        self._active_event_stream = None
        self._terminated = asyncio.Event()
        self._terminated.set()
    @property
    def state(self) -> str:
        """Return the current session state."""
        if self._stale:
            return "stale"
        return self._state

    @property
    def messages(self) -> tuple[dict[str, str], ...]:
        """Return a copy of the canonical session messages."""
        return tuple(message.copy() for message in self._messages)

    @property
    def cached_token_count(self) -> int:
        """Return the last committed cached token count."""
        return self._cached_token_count

    async def __aenter__(self) -> ChatSession:
        """Enter the session as an async context manager."""
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Close the session when leaving an async context manager."""
        await self.close()

    async def close(self) -> None:
        """Close the session and cancel any active turn."""
        if self._state != "owner_stopped":
            self._state = "closed"
        task = self._active_task
        if task is not None and not task.done():
            if task is asyncio.current_task():
                await self._reset_stream_consumer()
                return
            task.cancel()
        await self._wait_for_termination()

    def add_user(self, content: str) -> None:
        """Append a user message to the session."""
        self._append_message("user", content)

    def add_system(self, content: str) -> None:
        """Append a system message to the session."""
        self._append_message("system", content)

    async def chat(
        self,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        rep_penalty_lookback: int | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Collect the next assistant turn as a single string."""
        text = ""
        saw_done = False
        async for event in self.stream_chat(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            rep_penalty_lookback=rep_penalty_lookback,
            max_tokens=max_tokens,
        ):
            if isinstance(event, ChatTokenEvent):
                text += event.text
            elif isinstance(event, ChatFinalTextEvent):
                text = event.text
            elif isinstance(event, ChatDoneEvent):
                text = event.text
                saw_done = True
        if not saw_done:
            raise RuntimeError("Chat stream ended without a done event")
        return text

    async def stream_chat(
        self,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        rep_penalty_lookback: int | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ChatEvent]:
        """Stream structured events for the next assistant turn."""
        sampling = self._prepare_stream_chat(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            rep_penalty_lookback=rep_penalty_lookback,
            max_tokens=max_tokens,
        )
        try:
            async with await self._llm._admission.acquire():
                async for event in self._stream_chat_prepared(sampling):
                    yield event
        except EngineProgressTimeoutError as exc:
            await self._llm._recover_from_engine_failure()
            raise ProgressTimeoutError(str(exc)) from exc
        except (EngineCrashedError, EngineError) as exc:
            await self._llm._recover_from_engine_failure()
            raise RuntimeError(str(exc)) from exc

    def _prepare_stream_chat(
        self,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        rep_penalty_lookback: int | None = None,
        max_tokens: int | None = None,
    ) -> SamplingOptions:
        sampling = validate_sampling_options(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            rep_penalty_lookback=rep_penalty_lookback,
            max_tokens=max_tokens,
        )
        self._ensure_turn_startable()
        return sampling

    async def _stream_chat_prepared(
        self,
        sampling: SamplingOptions,
    ) -> AsyncIterator[ChatEvent]:
        first_event, full_text = await self._start_prepared_stream(sampling)
        async for event in self._consume_started_stream(first_event, full_text):
            yield event

    async def _start_prepared_stream(
        self,
        sampling: SamplingOptions,
    ) -> tuple[ChatEvent | None, str]:
        self._ensure_turn_startable()
        await self._begin_consumer()
        event_stream = self._llm._harness.stream_events(
            self,
            **sampling.to_kwargs(),
        )
        self._active_event_stream = event_stream
        try:
            first_event = await anext(event_stream)
            return first_event, self._accumulate_stream_text("", first_event)
        except StopAsyncIteration:
            return None, ""
        except asyncio.CancelledError:
            self._mark_stream_cancelled()
            await self._reset_stream_consumer()
            raise
        except SearchAuthenticationError as exc:
            self._mark_stream_failed(SearchAuthenticationError)
            await self._reset_stream_consumer()
            raise RuntimeError(str(exc)) from exc
        except ContextOverflowError:
            self._mark_stream_failed(ContextOverflowError)
            await self._reset_stream_consumer()
            raise
        except SessionExhaustedError:
            self._mark_stream_failed(SessionExhaustedError)
            await self._reset_stream_consumer()
            raise
        except EngineProgressTimeoutError:
            self._mark_stream_failed(EngineProgressTimeoutError)
            await self._reset_stream_consumer()
            raise
        except (EngineCrashedError, EngineError):
            self._mark_stream_failed(EngineError)
            await self._reset_stream_consumer()
            raise
        except Exception:
            self._mark_stream_failed(Exception)
            await self._reset_stream_consumer()
            raise

    async def _consume_started_stream(
        self,
        first_event: ChatEvent | None,
        full_text: str,
    ) -> AsyncIterator[ChatEvent]:
        try:
            if first_event is not None:
                yield first_event
            event_stream = self._active_event_stream
            if event_stream is not None:
                async for event in event_stream:
                    full_text = self._accumulate_stream_text(full_text, event)
                    yield event
            if self._state in {"closed", "owner_stopped"}:
                return
            yield self._finish_stream_success(full_text)
        except asyncio.CancelledError:
            self._mark_stream_cancelled()
            raise
        except SearchAuthenticationError as exc:
            self._mark_stream_failed(SearchAuthenticationError)
            raise RuntimeError(str(exc)) from exc
        except ContextOverflowError:
            self._mark_stream_failed(ContextOverflowError)
            raise
        except SessionExhaustedError:
            self._mark_stream_failed(SessionExhaustedError)
            raise
        except EngineProgressTimeoutError:
            self._mark_stream_failed(EngineProgressTimeoutError)
            raise
        except (EngineCrashedError, EngineError):
            self._mark_stream_failed(EngineError)
            raise
        except Exception:
            self._mark_stream_failed(Exception)
            raise
        finally:
            await self._reset_stream_consumer()

    def _append_message(self, role: str, content: str) -> None:
        self._ensure_mutable()
        validated = validate_messages(
            [*self._messages, {"role": role, "content": content}],
            require_user_turn=False,
            allow_empty=True,
        )
        self._messages = [
            {"role": message.role, "content": message.content}
            for message in validated
        ]

    def _ensure_mutable(self) -> None:
        if self._state == "closed":
            raise SessionClosedError("ChatSession is closed")
        if self._state == "owner_stopped":
            raise SessionClosedError("ChatSession owner has stopped")
        if self._stale:
            raise SessionStaleError(
                "ChatSession is stale after the active model changed; create a new session"
            )
        if self._state == "exhausted":
            raise SessionExhaustedError(
                f"ChatSession exceeded the {SESSION_TOKEN_LIMIT} token lifetime limit"
            )
        if self._consumer_active:
            raise SessionBusyError("ChatSession already has an active consumer")

    def _ensure_turn_startable(self) -> None:
        self._ensure_mutable()
        if not self._messages:
            raise ValueError("ChatSession has no messages")
        if self._messages[-1]["role"] == "assistant":
            raise ValueError(
                "ChatSession already has an assistant reply; append a new user or system message before chatting again"
            )

    async def _begin_consumer(self) -> None:
        if self._consumer_active:
            raise SessionBusyError("ChatSession already has an active consumer")
        self._consumer_active = True
        self._state = "streaming"
        self._active_task = asyncio.current_task()
        self._terminated.clear()

    def _prepare_generation(
        self,
        messages: list[dict[str, str]] | tuple[dict[str, str], ...] | None = None,
    ) -> list[int]:
        prompt = self._render_prompt(messages=messages, add_generation_prompt=True)
        token_ids = self._llm._tokenizer.encode(
            prompt,
            add_special_tokens=not bool(
                getattr(self._llm._tokenizer, "chat_template", None)
            ),
        )
        prepared = list(token_ids)
        prompt_tokens = len(prepared)
        if prompt_tokens >= self._llm.max_context_tokens:
            raise ContextOverflowError(prompt_tokens, self._llm.max_context_tokens)
        if prompt_tokens > SESSION_TOKEN_LIMIT:
            if messages is None or messages is self._messages:
                self._state = "exhausted"
            raise SessionExhaustedError(
                f"Chat session exceeds the {SESSION_TOKEN_LIMIT} token lifetime limit"
            )
        return prepared

    def _commit_assistant_turn(self, text: str) -> None:
        self._messages.append({"role": "assistant", "content": text})
        self._cached_token_count = self._llm._engine.cached_token_count

    def _commit_messages(self, messages: list[dict[str, str]]) -> None:
        self._messages = [message.copy() for message in messages]
        self._cached_token_count = self._llm._engine.cached_token_count

    def _render_prompt(
        self,
        *,
        messages: list[dict[str, str]] | tuple[dict[str, str], ...] | None = None,
        add_generation_prompt: bool,
    ) -> str:
        prompt_messages = self._renderable_messages(
            self._messages if messages is None else messages
        )
        tokenizer = self._llm._tokenizer
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template:
            return tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        prompt = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in prompt_messages
        )
        if not add_generation_prompt:
            return prompt
        if prompt:
            return f"{prompt}\nassistant: "
        return "assistant: "

    def _renderable_messages(
        self,
        messages: list[dict[str, str]] | tuple[dict[str, str], ...],
    ) -> list[dict[str, str]]:
        rendered: list[dict[str, str]] = []
        for message in messages:
            if message["role"] == "search":
                rendered.append(
                    {
                        "role": "system",
                        "content": f"Search results:\n{message['content']}",
                    }
                )
                continue
            rendered.append(message.copy())
        return rendered

    def _mark_stale(self) -> None:
        self._stale = True

    def _mark_owner_stopped(self) -> None:
        self._state = "owner_stopped"
        task = self._active_task
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

    async def _wait_for_termination(self) -> None:
        task = self._active_task
        if task is asyncio.current_task():
            return
        await self._terminated.wait()

    def _accumulate_stream_text(self, full_text: str, event: ChatEvent) -> str:
        if isinstance(event, ChatTokenEvent):
            return full_text + event.text
        if isinstance(event, ChatFinalTextEvent):
            return event.text
        return full_text

    def _finish_stream_success(self, full_text: str) -> ChatDoneEvent:
        self._cached_token_count = self._llm._engine.cached_token_count
        if self._cached_token_count >= SESSION_TOKEN_LIMIT:
            self._state = "exhausted"
        elif self._state not in {"closed", "owner_stopped"} and not self._stale:
            self._state = "open"
        return ChatDoneEvent(
            text=full_text,
            usage=ChatUsage(
                prompt_tokens=self._llm._harness.prompt_tokens,
                completion_tokens=self._llm._harness.completion_tokens,
                total_tokens=(
                    self._llm._harness.prompt_tokens
                    + self._llm._harness.completion_tokens
                ),
                cached_tokens=self._llm._harness.cached_tokens,
            ),
        )

    def _mark_stream_cancelled(self) -> None:
        if self._state not in {"closed", "owner_stopped"}:
            self._state = "closed"

    def _mark_stream_failed(self, error_type: type[BaseException]) -> None:
        if self._state in {"closed", "owner_stopped"} or self._stale:
            return
        if error_type in (SearchAuthenticationError, ContextOverflowError):
            self._state = "open"
            return
        if error_type is SessionExhaustedError:
            self._state = "exhausted"
            return
        self._state = "failed"

    async def _reset_stream_consumer(self) -> None:
        event_stream = self._active_event_stream
        try:
            if event_stream is not None:
                await event_stream.aclose()
        finally:
            self._active_event_stream = None
            self._consumer_active = False
            self._active_task = None
            self._terminated.set()

    def __del__(self):
        task = self._active_task
        try:
            if task is not None and not task.done():
                task.cancel()
        except Exception:
            pass


_ALLOW_CHAT_SESSION_SUBCLASS = False
