"""Chat session handles for the LLM component."""

from __future__ import annotations

import abc
import asyncio
from collections.abc import AsyncIterator
from enum import Enum

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
    validate_messages,
    validate_sampling_options,
    validate_user_message,
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


class _ChatSessionFSM(Enum):
    IDLE = "idle"
    STREAMING = "streaming"
    DONE = "done"


def _create_chat_session(llm, runtime, runtime_epoch: int) -> _ChatSession:
    return _ChatSession(
        llm,
        runtime,
        runtime_epoch,
        _owner_token=_CHAT_SESSION_OWNER_TOKEN,
    )


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
        """Return the committed session token count."""
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
        """Close the session."""
        ...  # pragma: no cover

    @abc.abstractmethod
    def append_message(self, role: str, content: str) -> None:
        """Append one validated role-bearing message."""
        ...  # pragma: no cover

    @abc.abstractmethod
    async def generate(
        self,
        user_message: str,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        rep_penalty_lookback: int | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ChatEvent]:
        """Stream structured events for a new user turn."""
        yield  # pragma: no cover

    @abc.abstractmethod
    async def collect(
        self,
        user_message: str,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        rep_penalty_lookback: int | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Collect a new user turn as a single assistant string."""
        ...  # pragma: no cover

    @abc.abstractmethod
    def new_chat(self) -> None:
        """Clear committed conversation messages."""
        ...  # pragma: no cover


_ALLOW_CHAT_SESSION_SUBCLASS = True


class _ChatSession(ChatSession):
    """Private concrete chat-session implementation owned by one LLM runtime."""

    def __new__(cls, llm=None, runtime=None, runtime_epoch=None, *, _owner_token=None):
        del llm, runtime, runtime_epoch
        if _owner_token is not _CHAT_SESSION_OWNER_TOKEN:
            raise TypeError(_CHAT_SESSION_CONSTRUCTION_ERROR)
        return super().__new__(cls)

    def __init__(self, llm=None, runtime=None, runtime_epoch=None, *, _owner_token=None) -> None:
        if _owner_token is not _CHAT_SESSION_OWNER_TOKEN:
            raise TypeError(_CHAT_SESSION_CONSTRUCTION_ERROR)
        if llm is None or runtime is None or runtime_epoch is None:
            raise TypeError(_CHAT_SESSION_CONSTRUCTION_ERROR)
        self._llm = llm
        self._runtime = runtime
        self._runtime_epoch = runtime_epoch
        self._harness = llm._build_harness(runtime)
        self._messages: list[dict[str, str]] = []
        self._cached_token_ids: tuple[int, ...] = ()
        self._messages_in_kv = 0
        self._state = _ChatSessionFSM.IDLE
        self._active_task: asyncio.Task | None = None
        self._active_event_stream = None
        self._pending_token_ids: tuple[int, ...] | None = None
        self._last_usage: ChatUsage | None = None
        self._close_requested = False

    @property
    def state(self) -> str:
        """Return the current session state."""
        if self._is_stale():
            return "stale"
        return self._state.value

    @property
    def messages(self) -> tuple[dict[str, str], ...]:
        """Return a copy of the canonical session messages."""
        return tuple(message.copy() for message in self._messages)

    @property
    def cached_token_count(self) -> int:
        """Return the committed session token count."""
        return len(self._cached_token_ids)

    async def __aenter__(self) -> ChatSession:
        self._llm._require_owner_loop()
        self._ensure_idle()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        """Cancel any active turn without closing the reusable session."""
        self._llm._require_owner_loop()
        task = self._active_task
        if task is None and self._active_event_stream is None:
            return
        self._close_requested = True
        if task is asyncio.current_task():
            # generate() will observe _close_requested after yielding control.
            return
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return
        stream = self._active_event_stream
        if stream is not None:
            await stream.aclose()
            self._rollback_messages()
            self._state = _ChatSessionFSM.IDLE

    def append_message(self, role: str, content: str) -> None:
        """Append one validated role-bearing message."""
        self._llm._require_owner_loop()
        self._ensure_idle()
        validated = validate_messages(
            [*self._messages, {"role": role, "content": content}],
            require_user_turn=False,
            allow_empty=True,
        )
        message = validated[-1]
        self._messages.append({"role": message.role, "content": message.content})

    async def generate(
        self,
        user_message: str,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        rep_penalty_lookback: int | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ChatEvent]:
        """Stream structured events for a new user turn."""
        sampling = validate_sampling_options(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            rep_penalty_lookback=rep_penalty_lookback,
            max_tokens=max_tokens,
        )
        content = validate_user_message(user_message)
        self._llm._require_owner_loop()
        self._ensure_idle()
        self._messages.append({"role": "user", "content": content})
        self._state = _ChatSessionFSM.STREAMING
        self._active_task = asyncio.current_task()
        self._pending_token_ids = None
        self._close_requested = False
        full_text = ""
        old_cached_tokens = len(self._cached_token_ids)
        event_stream = self._harness.stream_events(
            self,
            **sampling.to_kwargs(),
        )
        self._active_event_stream = event_stream
        try:
            async for event in event_stream:
                if isinstance(event, ChatTokenEvent):
                    full_text += event.text
                elif isinstance(event, ChatFinalTextEvent):
                    full_text = event.text
                yield event
                if self._close_requested:
                    await event_stream.aclose()
                    self._rollback_messages()
                    self._state = _ChatSessionFSM.IDLE
                    return
            if self._close_requested:
                self._rollback_messages()
                self._state = _ChatSessionFSM.IDLE
                return
            if self._messages[-1]["role"] != "assistant":
                self._messages.append({"role": "assistant", "content": full_text})
            if self._pending_token_ids is None:
                raise RuntimeError("LLM harness completed without committing token IDs")
            self._cached_token_ids = self._pending_token_ids
            self._messages_in_kv = len(self._messages)
            cached_token_count = len(self._cached_token_ids)
            if cached_token_count >= SESSION_TOKEN_LIMIT:
                self._state = _ChatSessionFSM.DONE
            else:
                self._state = _ChatSessionFSM.IDLE
            self._last_usage = ChatUsage(
                prompt_tokens=self._harness.prompt_tokens,
                completion_tokens=self._harness.completion_tokens,
                cached_tokens=old_cached_tokens,
                total_tokens=cached_token_count,
            )
            yield ChatDoneEvent(
                text=full_text,
                usage=self._last_usage,
            )
        except (asyncio.CancelledError, GeneratorExit):
            if self._active_event_stream is not None:
                await self._active_event_stream.aclose()
            self._rollback_messages()
            self._state = (
                _ChatSessionFSM.IDLE
                if self._close_requested
                else _ChatSessionFSM.DONE
            )
            raise
        except SearchAuthenticationError as exc:
            self._rollback_messages()
            self._state = _ChatSessionFSM.IDLE
            raise RuntimeError(str(exc)) from exc
        except ContextOverflowError:
            self._rollback_messages()
            self._state = _ChatSessionFSM.IDLE
            raise
        except SessionExhaustedError:
            self._rollback_messages()
            self._state = _ChatSessionFSM.DONE
            raise
        except EngineProgressTimeoutError as exc:
            self._rollback_messages()
            self._state = _ChatSessionFSM.DONE
            raise ProgressTimeoutError(str(exc)) from exc
        except (EngineCrashedError, EngineError) as exc:
            self._rollback_messages()
            self._state = _ChatSessionFSM.DONE
            raise RuntimeError(str(exc)) from exc
        except Exception:
            self._rollback_messages()
            self._state = _ChatSessionFSM.DONE
            raise
        finally:
            self._active_event_stream = None
            self._active_task = None
            self._pending_token_ids = None

    async def collect(
        self,
        user_message: str,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        rep_penalty_lookback: int | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Collect a new user turn as a single assistant string."""
        text = ""
        saw_done = False
        async for event in self.generate(
            user_message,
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

    def new_chat(self) -> None:
        """Clear committed conversation messages."""
        self._llm._require_owner_loop()
        self._ensure_idle()
        self._messages.clear()
        self._cached_token_ids = ()
        self._messages_in_kv = 0
        self._state = _ChatSessionFSM.IDLE

    def _prepare_generation(
        self,
        *,
        messages: list[dict[str, str]] | tuple[dict[str, str], ...],
    ) -> list[int]:
        prompt = self._render_prompt(
            messages=messages,
            add_generation_prompt=True,
        )
        tokenizer = self._runtime.tokenizer
        token_ids = list(self._cached_token_ids)
        token_ids.extend(
            tokenizer.encode(
                prompt,
                add_special_tokens=not bool(getattr(tokenizer, "chat_template", None)),
            )
        )
        prepared = list(token_ids)
        prompt_tokens = len(prepared)
        if prompt_tokens >= self._runtime.model.max_position_embeddings:
            raise ContextOverflowError(
                prompt_tokens,
                self._runtime.model.max_position_embeddings,
            )
        if prompt_tokens > SESSION_TOKEN_LIMIT:
            raise SessionExhaustedError(
                f"Chat session exceeds the {SESSION_TOKEN_LIMIT} token lifetime limit"
            )
        return prepared

    def _render_prompt(
        self,
        *,
        messages: list[dict[str, str]] | tuple[dict[str, str], ...],
        add_generation_prompt: bool,
    ) -> str:
        prompt_messages = self._renderable_messages(messages)
        if len(prompt_messages) < self._messages_in_kv:
            raise ValueError(
                "There are fewer messages in the thread than in the KV cache"
            )
        prompt_messages = prompt_messages[self._messages_in_kv :]
        tokenizer = self._runtime.tokenizer
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
            rendered.append(message)
        return rendered

    def _rollback_messages(self) -> None:
        del self._messages[self._messages_in_kv :]

    def _ensure_idle(self) -> None:
        if self._is_stale():
            raise SessionStaleError(
                "ChatSession is stale after the active model changed; create a new session"
            )
        if self._state is _ChatSessionFSM.DONE:
            raise SessionClosedError("ChatSession is closed")
        if self._state is not _ChatSessionFSM.IDLE:
            raise SessionBusyError("ChatSession already has an active generation")

    def _is_stale(self) -> bool:
        return self._runtime_epoch != self._llm._current_runtime_epoch()


_ALLOW_CHAT_SESSION_SUBCLASS = False
