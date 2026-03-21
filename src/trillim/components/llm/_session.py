"""Chat session handles for the LLM component."""

from __future__ import annotations

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
from trillim.components.llm._validation import validate_messages, validate_sampling_options
from trillim.errors import (
    ContextOverflowError,
    ProgressTimeoutError,
    SessionBusyError,
    SessionClosedError,
    SessionExhaustedError,
    SessionStaleError,
)


class ChatSession:
    """Owner-managed multi-turn chat state for one active LLM model."""

    _runtime_proxy = True

    def __init__(self, llm, messages) -> None:
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
        max_tokens: int | None = None,
    ) -> AsyncIterator[ChatEvent]:
        """Stream structured events for the next assistant turn."""
        sampling = validate_sampling_options(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
        )
        self._ensure_turn_startable()
        prompt_tokens = len(self._prepare_generation())
        if prompt_tokens >= self._llm.max_context_tokens:
            raise ContextOverflowError(prompt_tokens, self._llm.max_context_tokens)
        if prompt_tokens > SESSION_TOKEN_LIMIT:
            self._state = "exhausted"
            raise SessionExhaustedError(
                f"Chat session exceeds the {SESSION_TOKEN_LIMIT} token lifetime limit"
            )
        await self._begin_consumer()
        try:
            async with await self._llm._admission.acquire():
                full_text = ""
                async for event in self._llm._harness.stream_events(
                    self,
                    **sampling.to_kwargs(),
                ):
                    if isinstance(event, ChatTokenEvent):
                        full_text += event.text
                    elif isinstance(event, ChatFinalTextEvent):
                        full_text = event.text
                    yield event
                self._cached_token_count = self._llm._engine.cached_token_count
                if self._cached_token_count >= SESSION_TOKEN_LIMIT:
                    self._state = "exhausted"
                elif self._state not in {"closed", "owner_stopped"} and not self._stale:
                    self._state = "open"
                yield ChatDoneEvent(
                    text=full_text,
                    usage=ChatUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=self._llm._harness.completion_tokens,
                        total_tokens=prompt_tokens + self._llm._harness.completion_tokens,
                        cached_tokens=self._llm._engine.last_cache_hit,
                    ),
                )
        except asyncio.CancelledError:
            if self._state not in {"closed", "owner_stopped"}:
                self._state = "closed"
            raise
        except EngineProgressTimeoutError as exc:
            if self._state not in {"closed", "owner_stopped"} and not self._stale:
                self._state = "failed"
            await self._llm._recover_from_engine_failure()
            raise ProgressTimeoutError(str(exc)) from exc
        except (EngineCrashedError, EngineError) as exc:
            if self._state not in {"closed", "owner_stopped"} and not self._stale:
                self._state = "failed"
            await self._llm._recover_from_engine_failure()
            raise RuntimeError(str(exc)) from exc
        except Exception:
            if self._state not in {"closed", "owner_stopped"} and not self._stale:
                self._state = "failed"
            raise
        finally:
            self._consumer_active = False
            self._active_task = None
            self._terminated.set()

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

    def _prepare_generation(self) -> list[int]:
        prompt = self._render_prompt(add_generation_prompt=True)
        token_ids = self._llm._tokenizer.encode(
            prompt,
            add_special_tokens=not bool(
                getattr(self._llm._tokenizer, "chat_template", None)
            ),
        )
        return list(token_ids)

    def _commit_assistant_turn(self, text: str) -> None:
        self._messages.append({"role": "assistant", "content": text})
        self._cached_token_count = self._llm._engine.cached_token_count

    def _render_prompt(self, *, add_generation_prompt: bool) -> str:
        tokenizer = self._llm._tokenizer
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template:
            return tokenizer.apply_chat_template(
                self._messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        prompt = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in self._messages
        )
        if not add_generation_prompt:
            return prompt
        if prompt:
            return f"{prompt}\nassistant: "
        return "assistant: "

    def _mark_stale(self) -> None:
        self._stale = True

    def _mark_owner_stopped(self) -> None:
        self._state = "owner_stopped"
        task = self._active_task
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

    async def _wait_for_termination(self) -> None:
        task = self._active_task
        if task is None or task is asyncio.current_task():
            return
        await self._terminated.wait()

    def __del__(self):
        task = self._active_task
        try:
            if task is not None and not task.done():
                task.cancel()
        except Exception:
            pass
