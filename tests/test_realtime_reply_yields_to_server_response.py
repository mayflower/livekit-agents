"""A reply that waits for the provider's own response is heard after it.

A deferred tool reply is queued as the caller stops speaking. OpenAI's server VAD starts
its own response to the caller in the same moment, and its speech is queued behind the
reply. The reply then waits for that response to be generated before it can be issued, so
it is generated second, but playout follows the queue: the caller heard it first (local
voicebot call 2026-10-14, a filler "das Wetter wird gerade abgerufen" before the answer to
the weather question).
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from livekit.agents import Agent, AgentSession
from livekit.agents.voice.events import ConversationItemAddedEvent

from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel, FakeRealtimeSession
from .test_realtime_agent_state_during_tool import _generation

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


async def _requested_reply(model: FakeRealtimeModel) -> asyncio.Future:
    for _ in range(100):
        if model.active_session._reply_futs:
            return model.active_session._reply_futs[0]
        await asyncio.sleep(0.05)
    raise AssertionError("no reply was requested")


async def _heard(*, provider_busy: bool) -> list[str]:
    """Queue a reply, then the provider's own response; return what played, in order."""
    model = FakeRealtimeModel()
    heard: list[str] = []

    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(Agent(instructions="You are a helpful assistant."))

        @session.on("conversation_item_added")
        def _added(ev: ConversationItemAddedEvent) -> None:
            if ev.item.type == "message" and ev.item.role == "assistant":
                heard.append(ev.item.text_content or "")

        rt = model.active_session
        rt.busy = provider_busy  # type: ignore[attr-defined]
        reply = session.generate_reply(instructions="give the tool result")
        server = _generation(response_id="server", text="answer to the caller", audio_duration=0.5)
        rt.emit("generation_created", dataclasses.replace(server, user_initiated=False))

        # the provider is done generating its own response
        await asyncio.sleep(0.2)
        rt.busy = False  # type: ignore[attr-defined]
        (await _requested_reply(model)).set_result(
            _generation(response_id="reply", text="the tool result", audio_duration=0.5)
        )
        await asyncio.wait_for(reply.wait_for_playout(), timeout=10)
        await asyncio.sleep(1.0)

    return heard


@pytest.fixture(autouse=True)
def _busy_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        FakeRealtimeSession,
        "has_active_generation",
        property(lambda self: getattr(self, "busy", False)),
    )


async def test_a_reply_that_waited_for_the_server_response_plays_after_it() -> None:
    assert await _heard(provider_busy=True) == ["answer to the caller", "the tool result"]


async def test_a_reply_to_a_free_provider_keeps_its_place() -> None:
    assert await _heard(provider_busy=False) == ["the tool result", "answer to the caller"]
