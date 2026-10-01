"""One ``agent_turn`` span per speech handle.

A reply that calls a tool runs two generations (LLM steps) in two tasks; they used to be two
``agent_turn`` spans linked only by ``lk.parent_generation_id``. The speech handle now owns a
single span for its whole life: each generation is an event on it, tool and inference spans
nest under it, and it ends with the speech."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit import rtc
from livekit.agents import Agent, AgentSession, RunContext, function_tool, utils
from livekit.agents.llm import (
    FunctionCall,
    FunctionToolCall,
    GenerationCreatedEvent,
    MessageGeneration,
)
from livekit.agents.telemetry import set_tracer_provider, trace_types, tracer

from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel, fake_capabilities
from .fake_session import FakeActions, create_session, run_session
from .test_realtime_agent_state_during_tool import _generation
from .trace_schema import assert_trace_well_formed

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


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


def _children(
    exporter: InMemorySpanExporter, parent: ReadableSpan, name: str
) -> list[ReadableSpan]:
    return [
        s
        for s in _spans(exporter, name)
        if s.parent is not None and s.parent.span_id == parent.context.span_id
    ]


class _WeatherAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful assistant.")

    @function_tool
    async def get_weather(self, context: RunContext, location: str) -> str:
        return f"sunny in {location}"


async def _next_reply(model: FakeRealtimeModel) -> asyncio.Future[GenerationCreatedEvent]:
    for _ in range(500):
        if pending := [f for f in model.active_session._reply_futs if not f.done()]:
            return pending[0]
        await asyncio.sleep(0.01)
    raise AssertionError("no generate_reply issued")


def _weather_call_generation() -> GenerationCreatedEvent:
    return _generation(
        response_id="first",
        text="Let me check.",
        audio_duration=0.5,
        function_calls=[
            FunctionCall(call_id="1", name="get_weather", arguments='{"location": "Tokyo"}')
        ],
    )


async def test_tool_call_is_one_agent_turn(span_exporter: InMemorySpanExporter) -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.0, "What's the weather in Tokyo?")
    actions.add_llm(
        content="",
        tool_calls=[
            FunctionToolCall(name="get_weather", arguments='{"location": "Tokyo"}', call_id="1")
        ],
    )
    actions.add_llm(content="It is sunny in Tokyo.", input="sunny in Tokyo")
    actions.add_tts(1.0)

    session = create_session(actions, speed_factor=2.0)
    await asyncio.wait_for(run_session(session, _WeatherAgent(), drain_delay=1.0), timeout=60)

    [root] = _spans(span_exporter, "agent_session")
    turns = _spans(span_exporter, "agent_turn")
    assert len(turns) == 1, [(t.attributes or {}).get(trace_types.ATTR_SPEECH_ID) for t in turns]
    [turn] = turns
    assert turn.parent is not None and turn.parent.span_id == root.context.span_id

    attrs = turn.attributes or {}
    speech_id = attrs[trace_types.ATTR_SPEECH_ID]
    assert attrs[trace_types.ATTR_GENERATION_COUNT] == 2
    assert attrs[trace_types.ATTR_AGENT_TURN_ID] == f"{speech_id}_2"
    generations = [e for e in turn.events if e.name == "generation"]
    assert [(e.attributes or {})[trace_types.ATTR_AGENT_TURN_ID] for e in generations] == [
        f"{speech_id}_1",
        f"{speech_id}_2",
    ]
    assert trace_types.ATTR_AGENT_PARENT_TURN_ID not in (generations[0].attributes or {})
    assert (generations[1].attributes or {})[trace_types.ATTR_AGENT_PARENT_TURN_ID] == (
        f"{speech_id}_1"
    )

    # both generations' inference, the tool between them, and the speech all nest under it
    assert len(_children(span_exporter, turn, "llm_node")) == 2
    [tool] = _children(span_exporter, turn, "function_tool")
    [tts] = _children(span_exporter, turn, "tts_node")
    [speaking] = _children(span_exporter, turn, "agent_speaking")
    assert tool.start_time < tts.start_time
    # and the turn covers everything, ending with the speech rather than with the first step
    for child in (tool, tts, speaking):
        assert child.end_time is not None and turn.end_time is not None
        assert turn.start_time <= child.start_time and child.end_time <= turn.end_time
    # the whole tree, not just the edges this test names (tests/trace_schema.py)
    assert_trace_well_formed(span_exporter.get_finished_spans())


async def test_turn_text_is_what_every_generation_said(
    span_exporter: InMemorySpanExporter,
) -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.0, "What's the weather in Tokyo?")
    actions.add_llm(
        content="Let me check.",
        tool_calls=[
            FunctionToolCall(name="get_weather", arguments='{"location": "Tokyo"}', call_id="1")
        ],
    )
    actions.add_tts(0.5)
    actions.add_llm(content="It is sunny in Tokyo.", input="sunny in Tokyo")
    actions.add_tts(1.0)

    session = create_session(actions, speed_factor=2.0)
    await asyncio.wait_for(run_session(session, _WeatherAgent(), drain_delay=1.0), timeout=60)

    [turn] = _spans(span_exporter, "agent_turn")
    assert (turn.attributes or {})[trace_types.ATTR_RESPONSE_TEXT] == (
        "Let me check.\nIt is sunny in Tokyo."
    )


@pytest.mark.virtual_time
async def test_realtime_turn_text_is_what_every_generation_said(
    span_exporter: InMemorySpanExporter,
) -> None:
    # the tool reply is requested by the framework, on the same speech handle
    model = FakeRealtimeModel(capabilities=fake_capabilities(auto_tool_reply_generation=False))

    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(_WeatherAgent())

        reply = session.generate_reply()
        (await _next_reply(model)).set_result(_weather_call_generation())
        (await _next_reply(model)).set_result(
            _generation(response_id="second", text="It is sunny in Tokyo.", audio_duration=0.5)
        )
        await asyncio.wait_for(reply, timeout=10)

    [turn] = _spans(span_exporter, "agent_turn")
    assert len(_children(span_exporter, turn, "realtime_inference")) == 2
    assert (turn.attributes or {})[trace_types.ATTR_RESPONSE_TEXT] == (
        "Let me check.\nIt is sunny in Tokyo."
    )


@pytest.mark.virtual_time
async def test_realtime_playout_position_counts_every_generation(
    span_exporter: InMemorySpanExporter,
) -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(auto_tool_reply_generation=False))
    audio_output = FakeAudioOutput()

    async with AgentSession(llm=model) as session:
        session.output.audio = audio_output
        await session.start(_WeatherAgent())

        reply = session.generate_reply()
        (await _next_reply(model)).set_result(_weather_call_generation())  # 0.5 s, played out
        (await _next_reply(model)).set_result(
            _generation(response_id="second", text="It is sunny in Tokyo.", audio_duration=2.0)
        )
        while audio_output._started_at is None:
            await asyncio.sleep(0)
        await asyncio.sleep(0.3)
        reply.interrupt(force=True)
        await asyncio.wait_for(reply, timeout=10)

    [turn] = _spans(span_exporter, "agent_turn")
    attrs = turn.attributes or {}
    assert attrs[trace_types.ATTR_SPEECH_INTERRUPTED] is True
    # the whole first generation and 0.3 s of the second, not the second alone
    assert attrs[trace_types.ATTR_PLAYOUT_POSITION] == pytest.approx(0.8, abs=0.05)


async def test_plain_reply_is_one_generation(span_exporter: InMemorySpanExporter) -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 1.5, "Hello", stt_delay=0.1)
    actions.add_llm("Hi there", ttft=0.1, duration=0.2)
    actions.add_tts(0.5, ttfb=0.1, duration=0.2)

    session = create_session(actions, speed_factor=2.0)
    await asyncio.wait_for(
        run_session(session, Agent(instructions="test"), drain_delay=1.0), timeout=60
    )

    [turn] = _spans(span_exporter, "agent_turn")
    attrs = turn.attributes or {}
    assert attrs[trace_types.ATTR_GENERATION_COUNT] == 1
    assert attrs[trace_types.ATTR_AGENT_TURN_ID] == f"{attrs[trace_types.ATTR_SPEECH_ID]}_1"
    assert len([e for e in turn.events if e.name == "generation"]) == 1
    assert trace_types.ATTR_AGENT_PARENT_TURN_ID not in attrs
    # the whole tree, not just the edges this test names (tests/trace_schema.py)
    assert_trace_well_formed(span_exporter.get_finished_spans())


def test_discarded_preemptive_generation_hands_its_turn_to_the_successor(
    span_exporter: InMemorySpanExporter,
) -> None:
    """A preemptive attempt discarded for the real reply (or a newer attempt) must not leave a
    second agent_turn behind: the successor continues the span, the discarded speech ends
    without touching it."""
    from livekit.agents.voice.agent_activity import _agent_turn, _continue_discarded_turn
    from livekit.agents.voice.speech_handle import SpeechHandle

    root = tracer.start_span("agent_session")
    root_ctx = trace.set_span_in_context(root)
    attempt = SpeechHandle.create(allow_interruptions=True)
    with _agent_turn(attempt, root_context=root_ctx, agent_label="a"):
        pass  # the attempt's first generation ran here

    reply = SpeechHandle.create(allow_interruptions=True)
    _continue_discarded_turn(attempt, reply)
    attempt._mark_done()  # the cancelled attempt finishes: the span must survive it
    assert _spans(span_exporter, "agent_turn") == []

    with _agent_turn(reply, root_context=root_ctx, agent_label="a"):
        pass
    reply._mark_done()
    root.end()

    [turn] = _spans(span_exporter, "agent_turn")
    attrs = turn.attributes or {}
    assert attrs[trace_types.ATTR_SPEECH_ID] == reply.id
    assert attrs[trace_types.ATTR_GENERATION_COUNT] == 1  # the reply's own step count
    events = [e.name for e in turn.events]
    assert events == ["generation", "preemptive_generation_discarded", "generation"]
    [discarded] = [e for e in turn.events if e.name == "preemptive_generation_discarded"]
    assert (discarded.attributes or {})[trace_types.ATTR_SPEECH_ID] == attempt.id

    # nothing to hand over: a plain successor is untouched
    _continue_discarded_turn(None, reply)
    _continue_discarded_turn(reply, reply)


def test_llm_failure_stored_on_the_handle_fails_the_turn(
    span_exporter: InMemorySpanExporter,
) -> None:
    """The pipeline stores an LLM failure on the handle and marks it done without one; the
    turn must still end as failed."""
    from opentelemetry.trace import StatusCode

    from livekit.agents.voice.agent_activity import _agent_turn
    from livekit.agents.voice.speech_handle import SpeechHandle

    handle = SpeechHandle.create(allow_interruptions=True)
    with _agent_turn(handle, root_context=None, agent_label="a"):
        handle._error = RuntimeError("llm down")
    handle._mark_done()

    [turn] = _spans(span_exporter, "agent_turn")
    assert turn.status.status_code == StatusCode.ERROR
    assert [e.name for e in turn.events if e.name == "exception"] == ["exception"]


def test_turn_duration_metric_is_recorded_when_the_span_is_sampled_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    from livekit.agents.telemetry import otel_metrics
    from livekit.agents.voice.speech_handle import SpeechHandle

    record = MagicMock()
    monkeypatch.setattr(otel_metrics, "record_invoke_agent_duration", record)
    handle = SpeechHandle.create(allow_interruptions=True)
    handle._agent_turn_span = trace.NonRecordingSpan(trace.INVALID_SPAN_CONTEXT)
    handle._agent_turn_started_at = 1.0
    handle._agent_turn_agent_name = "a"
    handle._mark_done()
    record.assert_called_once()
    assert record.call_args.kwargs == {"agent_name": "a"}


def test_sampled_out_turn_is_still_handed_to_the_successor() -> None:
    """The successor adopts a non-recording turn too, so the duration metric keeps the
    discarded attempt's start time."""
    from livekit.agents.voice.agent_activity import _continue_discarded_turn
    from livekit.agents.voice.speech_handle import SpeechHandle

    attempt = SpeechHandle.create(allow_interruptions=True)
    attempt._agent_turn_span = trace.NonRecordingSpan(trace.INVALID_SPAN_CONTEXT)
    attempt._agent_turn_started_at = 1.0
    attempt._agent_turn_agent_name = "a"

    reply = SpeechHandle.create(allow_interruptions=True)
    _continue_discarded_turn(attempt, reply)
    assert attempt._agent_turn_span is None
    assert reply._agent_turn_span is not None
    assert reply._agent_turn_started_at == 1.0
    assert reply._agent_turn_agent_name == "a"


@pytest.mark.virtual_time
async def test_follow_up_cut_before_it_plays_keeps_the_earlier_position(
    span_exporter: InMemorySpanExporter,
) -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(auto_tool_reply_generation=False))

    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(_WeatherAgent())

        reply = session.generate_reply()
        (await _next_reply(model)).set_result(_weather_call_generation())  # 0.5 s, played out
        follow_up = await _next_reply(model)
        # the caller starts talking: the follow-up waits for silence before it may play
        assert session._activity is not None
        session._activity._user_silence_event.clear()
        follow_up.set_result(
            _generation(response_id="second", text="It is sunny in Tokyo.", audio_duration=2.0)
        )
        await asyncio.sleep(0.1)
        reply.interrupt(force=True)
        await asyncio.wait_for(reply, timeout=10)

    [turn] = _spans(span_exporter, "agent_turn")
    attrs = turn.attributes or {}
    assert attrs[trace_types.ATTR_SPEECH_INTERRUPTED] is True
    assert attrs[trace_types.ATTR_PLAYOUT_POSITION] == pytest.approx(0.5, abs=0.05)


@pytest.mark.virtual_time
async def test_realtime_turn_cut_before_its_first_word_has_no_position(
    span_exporter: InMemorySpanExporter,
) -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities())

    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(Agent(instructions="test"))

        reply = session.generate_reply()
        message_ch = utils.aio.Chan[MessageGeneration]()
        function_ch = utils.aio.Chan[FunctionCall]()
        text_ch = utils.aio.Chan[str]()
        audio_ch = utils.aio.Chan[rtc.AudioFrame]()  # no frame arrives before the cut
        modalities = asyncio.Future[list[str]]()
        modalities.set_result(["audio", "text"])
        message_ch.send_nowait(
            MessageGeneration(
                message_id="message",
                text_stream=text_ch,
                audio_stream=audio_ch,
                modalities=modalities,
            )
        )
        message_ch.close()
        (await _next_reply(model)).set_result(
            GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=True,
                response_id="response",
            )
        )
        await asyncio.sleep(0.1)
        reply.interrupt(force=True)
        for ch in (function_ch, text_ch, audio_ch):
            ch.close()
        await asyncio.wait_for(reply, timeout=10)

    [turn] = _spans(span_exporter, "agent_turn")
    attrs = turn.attributes or {}
    assert attrs[trace_types.ATTR_SPEECH_INTERRUPTED] is True
    assert trace_types.ATTR_PLAYOUT_POSITION not in attrs
