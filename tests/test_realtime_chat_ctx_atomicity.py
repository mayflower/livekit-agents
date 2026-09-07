"""RealtimeSession.update_chat_ctx_with must not lose a concurrent append.

``update_chat_ctx`` is declarative: it takes the whole conversation and the
session removes whatever the submitted context leaves out. A caller that only
wants to *add* something therefore has to read the current context first, and
anything that lands between that read and the write is dropped as though its
removal had been requested. Nobody asks for that — the appending callers only
ever add — so it is a lost update, not an intended deletion.

``update_chat_ctx_with`` closes it by running the caller's producer under the
same lock as the write, against the context as it is at that moment.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents.llm import ChatContext, ChatItem

from .fake_realtime import FakeRealtimeModel, FakeRealtimeSession

pytestmark = pytest.mark.unit


def _appending(item: ChatItem):
    def _produce(current: ChatContext) -> ChatContext:
        chat_ctx = current.copy()
        chat_ctx.items.append(item)
        return chat_ctx

    return _produce


def _message(id: str) -> ChatItem:
    return ChatContext.empty().add_message(role="user", content=id, id=id)


class _SlowSession(FakeRealtimeSession):
    """A session whose write takes a round-trip, like a real one's does.

    The gap between reading the context and the write landing is where a
    concurrent append is lost, so a test that cannot open that gap cannot tell
    the two versions of this code apart.
    """

    def __init__(self, model: FakeRealtimeModel) -> None:
        super().__init__(model)
        self.in_flight = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def update_chat_ctx(self, chat_ctx: ChatContext) -> None:
        self.in_flight.set()
        await self.release.wait()
        await super().update_chat_ctx(chat_ctx)


def _ids(session: FakeRealtimeSession) -> list[str]:
    return [item.id for item in session.chat_ctx.items]


async def test_an_append_that_lands_during_another_write_is_not_lost() -> None:
    session = _SlowSession(FakeRealtimeModel())
    await session.update_chat_ctx(ChatContext([_message("first")]))

    session.release.clear()
    slow = asyncio.create_task(session.update_chat_ctx_with(_appending(_message("slow"))))
    await session.in_flight.wait()

    # a second appender arrives while the first write is still in flight
    second = asyncio.create_task(session.update_chat_ctx_with(_appending(_message("second"))))
    await asyncio.sleep(0)

    session.release.set()
    await asyncio.gather(slow, second)

    assert _ids(session) == ["first", "slow", "second"]


async def test_concurrent_appends_all_survive() -> None:
    session = FakeRealtimeSession(FakeRealtimeModel())
    await session.update_chat_ctx(ChatContext([_message("first")]))

    await asyncio.gather(
        *(session.update_chat_ctx_with(_appending(_message(f"item{n}"))) for n in range(8))
    )

    assert _ids(session) == ["first", *[f"item{n}" for n in range(8)]]


async def test_update_chat_ctx_still_means_set_exactly_this() -> None:
    """The declarative API keeps its meaning — some callers do intend a removal.

    ``AgentActivity`` uses one to drop assistant items the user never heard.
    """
    session = FakeRealtimeSession(FakeRealtimeModel())
    await session.update_chat_ctx(ChatContext([_message("first"), _message("second")]))

    await session.update_chat_ctx(ChatContext([_message("first")]))

    assert _ids(session) == ["first"]


async def test_an_output_read_back_from_the_session_survives_tool_filtering() -> None:
    """A result the model was already given must not be filtered out as foreign.

    `ChatContext.copy(tools=...)` drops function calls and outputs that are not
    the agent's. It judged an output by its own `name` — but a provider's wire
    format for an *output* has no name field, so every output read back from a
    realtime session has lost it (`openai_item_to_livekit_item` builds one
    without). Any caller deriving an update from the session's own context
    therefore submitted a context missing every earlier result, and the diff
    deleted them: the model was told the weather, then had it taken away, and
    said the lookup was still running.
    """
    from livekit.agents.llm import FunctionCall, FunctionCallOutput, function_tool

    @function_tool
    async def web_search(query: str) -> str:
        """Search the web."""
        return "sunny"

    chat_ctx = ChatContext(
        [
            FunctionCall(id="c", call_id="call_1", name="web_search", arguments="{}"),
            # as it comes back from the session: no name, only the call_id
            FunctionCallOutput(id="o", call_id="call_1", output="18C", is_error=False),
            # an output whose call is not the agent's stays filtered
            FunctionCallOutput(id="x", call_id="call_9", output="?", is_error=False),
        ]
    )

    kept = [item.id for item in chat_ctx.copy(tools=[web_search]).items]

    assert kept == ["c", "o"]
