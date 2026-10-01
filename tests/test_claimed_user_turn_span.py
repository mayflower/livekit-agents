"""A claimed user turn (text input) takes ``user_state`` without a ``user_speaking`` span.

``_claim_user_turn`` pins ``user_state`` to ``"speaking"`` while a text input is handled. The
span is the user's voice, so it used to open and close around the claim, 1 ms long, for a
turn nobody spoke. Speech that starts while a turn is claimed still gets its own span."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents import Agent, AgentSession, UserStateChangedEvent
from livekit.agents.telemetry import set_tracer_provider, tracer
from livekit.agents.voice.room_io.types import TextInputEvent, _default_text_input_cb
from livekit.agents.voice.transcription.synchronizer import _SyncedAudioOutput

from .fake_io import FakeAudioInput
from .fake_session import FakeActions, create_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


@pytest.fixture
def span_exporter() -> Iterator[InMemorySpanExporter]:
    original_provider = tracer._tracer_provider
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    set_tracer_provider(provider)
    try:
        yield exporter
    finally:
        set_tracer_provider(original_provider)
        provider.shutdown()


def _spans(exporter: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    return [s for s in exporter.get_finished_spans() if s.name == name]


async def _start_session(actions: FakeActions) -> AgentSession:
    session = create_session(actions)
    await session.start(Agent(instructions="You are a helpful assistant."))
    audio_input = session.input.audio
    assert isinstance(audio_input, FakeAudioInput)
    audio_input.push(0.1)
    await asyncio.sleep(0.1)
    return session


async def _close_session(session: AgentSession) -> None:
    audio_output = session.output.audio
    await session.aclose()
    if isinstance(audio_output, _SyncedAudioOutput):
        await audio_output._synchronizer.aclose()


async def test_text_input_opens_no_user_speaking_span(
    span_exporter: InMemorySpanExporter,
) -> None:
    actions = FakeActions()
    actions.add_llm("Hi there.", input="hello from chat")
    actions.add_tts(0.5)
    session = await _start_session(actions)
    states: list[str] = []
    session.on("user_state_changed", lambda ev: states.append(ev.new_state))

    await _default_text_input_cb(session, TextInputEvent(text="hello from chat"))
    await asyncio.sleep(2.0)
    await _close_session(session)

    # the claim still holds the turn
    assert states == ["speaking", "listening"]
    assert not _spans(span_exporter, "user_speaking")
    assert len(_spans(span_exporter, "agent_turn")) == 1


async def test_speech_inside_a_claimed_turn_keeps_its_span(
    span_exporter: InMemorySpanExporter,
) -> None:
    session = await _start_session(FakeActions())
    activity = session._activity
    assert activity is not None
    states: list[UserStateChangedEvent] = []
    session.on("user_state_changed", states.append)

    async with session._claim_user_turn():
        await asyncio.sleep(0.2)
        started_at = time.time()
        activity.on_start_of_speech(None, speech_start_time=started_at)
    await asyncio.sleep(0.5)
    activity.on_end_of_speech(None)
    await _close_session(session)

    assert [ev.new_state for ev in states] == ["speaking", "listening"]
    [span] = _spans(span_exporter, "user_speaking")
    assert span.start_time == int(started_at * 1_000_000_000)
    assert span.end_time is not None
    assert (span.end_time - span.start_time) / 1e9 == pytest.approx(0.5, abs=1e-3)


async def test_each_utterance_inside_a_claimed_turn_gets_its_own_span(
    span_exporter: InMemorySpanExporter,
) -> None:
    session = await _start_session(FakeActions())
    activity = session._activity
    assert activity is not None

    starts: list[float] = []
    async with session._claim_user_turn():
        for _ in range(2):
            await asyncio.sleep(0.5)
            starts.append(time.time())
            activity.on_start_of_speech(None, speech_start_time=starts[-1])
            await asyncio.sleep(0.3)
            activity.on_end_of_speech(None)
        await asyncio.sleep(1.0)
    await _close_session(session)

    spans = sorted(_spans(span_exporter, "user_speaking"), key=lambda s: s.start_time or 0)
    assert [s.start_time for s in spans] == [int(t * 1_000_000_000) for t in starts]
    for span in spans:
        assert span.start_time is not None and span.end_time is not None
        assert (span.end_time - span.start_time) / 1e9 == pytest.approx(0.3, abs=1e-3)
