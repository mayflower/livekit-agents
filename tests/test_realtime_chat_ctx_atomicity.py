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


class _BusySession(FakeRealtimeSession):
    """A session that is mid-response until told otherwise.

    Stands in for the provider keeping one response in flight — OpenAI answers
    a second `response.create` with `conversation_already_has_active_response`,
    and the framework does not retry a reply that gets that.
    """

    def __init__(self, model: FakeRealtimeModel) -> None:
        super().__init__(model)
        self.busy = True

    @property
    def has_active_generation(self) -> bool:
        return self.busy


async def test_a_reply_waits_for_the_provider_to_finish_its_response() -> None:
    session = _BusySession(FakeRealtimeModel())

    waiting = asyncio.create_task(session.wait_for_response_slot())
    for _ in range(5):
        await asyncio.sleep(0)
    assert not waiting.done(), "issued a reply while the provider was generating"

    session.busy = False
    assert await waiting is True


async def test_waiting_for_the_slot_gives_up_rather_than_dropping_the_reply() -> None:
    """A provider stuck generating must not cost the reply entirely.

    Issuing it and having it refused is recoverable and visible; never issuing
    it is the silent loss this whole path exists to avoid.
    """
    session = _BusySession(FakeRealtimeModel())

    assert await session.wait_for_response_slot(timeout=0.05) is False


async def test_a_free_session_does_not_wait() -> None:
    session = _BusySession(FakeRealtimeModel())
    session.busy = False

    assert await session.wait_for_response_slot(timeout=0) is True


# --- the signals a realtime turn leaves unset, and the reply nobody reports ---


async def test_realtime_speech_marks_the_user_as_speaking() -> None:
    """The gates that hold a reply back read these, and a realtime turn set neither.

    `AgentSession` builds a default VAD but the activity de-wires it when the
    model does its own turn taking, so `AudioRecognition._speaking` stays False
    for the whole call and `_user_silence_event` is never cleared — and a
    deferred tool reply is then spoken over a talking caller.
    """
    from livekit.agents.llm import InputSpeechStartedEvent, InputSpeechStoppedEvent

    activity = _speech_signal_activity()

    activity._on_input_speech_started(InputSpeechStartedEvent())
    assert not activity._user_silence_event.is_set()
    assert activity._audio_recognition._speaking is True

    activity._on_input_speech_stopped(
        InputSpeechStoppedEvent(user_transcription_enabled=False)
    )
    assert activity._user_silence_event.is_set()
    assert activity._audio_recognition._speaking is False


async def test_an_unpaired_speech_start_releases_itself() -> None:
    """Owning the latch means owning its failure mode.

    A start whose stop never arrives would mute the agent for the rest of the
    call: playout waits on `_user_silence_event` and `_deliver_reply` waits on
    an untimed `wait_for_idle()`. `_vad_task` guards the same shape in its
    `finally`, but it does not run when the VAD is de-wired.
    """
    from livekit.agents.voice import agent_activity as aa
    from livekit.agents.llm import InputSpeechStartedEvent

    activity = _speech_signal_activity()
    original = aa._UNPAIRED_SPEECH_TIMEOUT
    aa._UNPAIRED_SPEECH_TIMEOUT = 0.02
    try:
        activity._on_input_speech_started(InputSpeechStartedEvent())
        assert not activity._user_silence_event.is_set()
        await asyncio.sleep(0.05)
    finally:
        aa._UNPAIRED_SPEECH_TIMEOUT = original

    assert activity._user_silence_event.is_set()
    assert activity._audio_recognition._speaking is False


async def test_a_client_vad_keeps_ownership_of_the_signals() -> None:
    """When a real VAD is driving, its stream is authoritative and this stays out."""
    from livekit.agents.llm import InputSpeechStartedEvent

    from .fake_vad import FakeVAD

    activity = _speech_signal_activity()
    # a VAD set on the agent, which is what makes it client-side rather than
    # the session's default
    activity._agent._vad = FakeVAD()
    activity._user_silence_event.set()

    activity._on_input_speech_started(InputSpeechStartedEvent())

    assert activity._user_silence_event.is_set()


def _speech_signal_activity() -> Any:
    """A real activity in the production shape: realtime turn detection, VAD de-wired.

    Built rather than stubbed, because what is being tested is which of the
    framework's own signals a realtime turn leaves untouched — a hand-made
    object would only prove the test agrees with itself.
    """
    from livekit.agents import Agent, AgentSession
    from livekit.agents.voice.agent_activity import AgentActivity
    from livekit.agents.voice.audio_recognition import AudioRecognition
    from livekit.agents.voice.endpointing import BaseEndpointing

    session = AgentSession(llm=FakeRealtimeModel())
    agent = Agent(instructions="test")
    activity = AgentActivity(agent, session)
    agent._activity = activity
    activity._audio_recognition = AudioRecognition(
        session,
        hooks=activity,
        endpointing=BaseEndpointing(min_delay=0.4, max_delay=6.0),
        stt=None,
        vad=None,
        interruption_detection=None,
        turn_detection=None,
    )
    return activity
