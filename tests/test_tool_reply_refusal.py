"""A deferred tool reply the provider refuses because it is busy is asked for again.

OpenAI refuses a `response.create` while a response is active. A server-VAD
response started as the caller stops speaking is known to the client only from
its `response.created`, so the reply that speaks a finished tool result can be
sent into it and refused. Before the retry, that refusal ended the reply for good
and was reported `skipped`.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livekit.agents import llm
from livekit.agents.llm.chat_context import ChatContext, FunctionCallOutput
from livekit.agents.voice.events import ToolExecutionUpdatedEvent, ToolReplyUpdated
from livekit.agents.voice.tool_executor import (
    _MAX_REPLY_REFUSALS,
    _REFUSED_FOR_ACTIVE_RESPONSE,
    _PendingUpdate,
    _ToolExecutor,
)

pytestmark = pytest.mark.unit

_REFUSED = llm.RealtimeError("active response", code=_REFUSED_FOR_ACTIVE_RESPONSE)


class _Speech:
    def __init__(self, n: int, error: BaseException | None) -> None:
        self.id = f"speech_{n}"
        self.interrupted = False
        self.chat_items: list[Any] = [] if error else ["spoken"]
        self._error = error
        self._callbacks: list[Any] = []
        asyncio.get_running_loop().call_soon(self._finish)

    def exception(self) -> BaseException | None:
        return self._error

    def add_done_callback(self, cb: Any) -> None:
        self._callbacks.append(cb)

    def _finish(self) -> None:
        for cb in self._callbacks:
            cb(self)


class _Agent:
    def __init__(self) -> None:
        self.chat_ctx = ChatContext()


class _Activity:
    def __init__(self, agent: _Agent) -> None:
        self.agent = agent


class _Session:
    """Answers each reply request with the next outcome: an error or speech."""

    def __init__(self, outcomes: list[BaseException | None]) -> None:
        self._outcomes = outcomes
        self.agent = _Agent()
        self.requests = 0
        self.statuses: list[str] = []
        self.settled = asyncio.Event()
        self._global_run_state = None

    async def wait_for_idle(self) -> _Activity:
        return _Activity(self.agent)

    def generate_reply(self, **_: Any) -> _Speech:
        error = self._outcomes[self.requests]
        self.requests += 1
        return _Speech(self.requests, error)

    def _tool_execution_updated(self, ev: ToolExecutionUpdatedEvent) -> None:
        assert isinstance(ev.update, ToolReplyUpdated)
        self.statuses.append(ev.update.status)
        if ev.update.status != "scheduled":
            self.settled.set()


async def _deliver(outcomes: list[BaseException | None]) -> _Session:
    session = _Session(outcomes)
    executor = _ToolExecutor()
    output = FunctionCallOutput(call_id="call_1", output="30 contacts", is_error=False)
    executor._pending_updates.append(
        _PendingUpdate(ctx=None, items=[output], target=session.agent)  # type: ignore[arg-type]
    )
    executor._schedule_delivery(session)  # type: ignore[arg-type]
    await asyncio.wait_for(session.settled.wait(), 1)
    return session


async def test_a_reply_refused_for_an_active_response_is_asked_for_again() -> None:
    session = await _deliver([_REFUSED, None])

    assert session.requests == 2
    assert session.statuses == ["scheduled", "scheduled", "completed"]


async def test_a_provider_that_keeps_refusing_is_given_up_on() -> None:
    session = await _deliver([_REFUSED] * (_MAX_REPLY_REFUSALS + 1))

    assert session.requests == _MAX_REPLY_REFUSALS + 1
    assert session.statuses[-1] == "skipped"


async def test_any_other_failure_is_not_retried() -> None:
    session = await _deliver([llm.RealtimeError("generate_reply timed out")])

    assert session.requests == 1
    assert session.statuses == ["scheduled", "skipped"]
