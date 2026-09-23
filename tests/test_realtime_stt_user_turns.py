"""A session STT next to a realtime model that owns turn detection (``realtime_llm``).

The model detects the turn and replies on its own, so nothing ever commits the STT's
transcript: every final was appended to one buffer and the whole call surfaced as a
single user turn when the session closed. Each final now closes its own turn."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents import Agent, AgentSession, llm
from livekit.agents.telemetry import set_tracer_provider, trace_types, tracer
from livekit.agents.voice.events import ConversationItemAddedEvent

from .fake_io import FakeAudioInput
from .fake_realtime import FakeRealtimeModel, fake_capabilities
from .fake_stt import FakeSTT, FakeUserSpeech

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

_FIRST = "where is my parcel"
_SECOND = "it was sent on monday"


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


async def _run_call(*, user_transcription: bool) -> tuple[list[llm.ChatMessage], Agent]:
    stt = FakeSTT(
        fake_user_speeches=[
            FakeUserSpeech(start_time=0.1, end_time=1.0, transcript=_FIRST, stt_delay=0.2),
            FakeUserSpeech(start_time=3.0, end_time=4.0, transcript=_SECOND, stt_delay=0.2),
        ]
    )
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(user_transcription=user_transcription, audio_output=False)
    )
    agent = Agent(instructions="test")
    added: list[llm.ChatMessage] = []
    session = AgentSession(
        llm=model, stt=stt, vad=None, turn_handling={"turn_detection": "realtime_llm"}
    )

    def _on_item(ev: ConversationItemAddedEvent) -> None:
        if ev.item.type == "message" and ev.item.role == "user":
            added.append(ev.item)

    session.on("conversation_item_added", _on_item)
    audio_input = FakeAudioInput()
    session.input.audio = audio_input
    await session.start(agent)
    audio_input.push(5.0)
    await stt.fake_user_speeches_done
    await asyncio.sleep(0.1)

    # both turns are in before the session closes, each on its own
    before_close = list(added)
    await session.aclose()
    assert added == before_close
    return added, agent


async def test_each_final_closes_its_own_user_turn(span_exporter: InMemorySpanExporter) -> None:
    added, agent = await _run_call(user_transcription=False)

    assert [m.text_content for m in added] == [_FIRST, _SECOND]
    turns = [s for s in span_exporter.get_finished_spans() if s.name == "user_turn"]
    assert [(s.attributes or {}).get(trace_types.ATTR_USER_TRANSCRIPT) for s in turns] == [
        _FIRST,
        _SECOND,
    ]
    # the message is dated to the turn's start, which the first interim opened
    assert [s.start_time for s in turns] == [int(m.created_at * 1e9) for m in added]
    assert turns[0].start_time < turns[1].start_time
    assert all(m.metrics["started_speaking_at"] == m.created_at for m in added)
    # the model heard the audio itself: sending the transcript on would repeat the caller
    assert not [i for i in agent.chat_ctx.items if i.type == "message" and i.role == "user"]


async def test_model_transcript_wins_over_the_stt(span_exporter: InMemorySpanExporter) -> None:
    # a model that transcribes the caller reports the turn itself; the STT only traces it
    added, _ = await _run_call(user_transcription=True)

    assert added == []
    turns = [s for s in span_exporter.get_finished_spans() if s.name == "user_turn"]
    assert len(turns) == 2
