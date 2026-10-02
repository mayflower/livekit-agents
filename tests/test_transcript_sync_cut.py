"""A cut keeps the words its played audio holds.

The synchronizer paces the transcript at a default speech rate until both the text and the
audio of a segment have ended, and counts a word only once its whole paced delay has passed.
A cut took that lagging progress as the segment's text: a realtime voice speaking faster than
the default rate lost the words played since, and a cut inside the first word kept nothing.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, tokenize, utils
from livekit.agents.llm import FunctionCall, GenerationCreatedEvent, MessageGeneration
from livekit.agents.voice.io import PlaybackFinishedEvent
from livekit.agents.voice.transcription.synchronizer import TranscriptSynchronizer

from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel, fake_capabilities

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time]

SAMPLE_RATE = 24000

# 22 hyphens in 3.5s: 6.3/s against the default 3.83/s
REPLY = "Sure, the weekend in Berlin stays sunny and dry, with highs of up to twenty-six degrees."
RATE = 22 / 3.5

# the shape of a cut measured on a realtime call: 22 hyphens in 3.76s, then the rest of an
# answer of 10s in all
WEATHER = (
    "Klar, in Würzburg wird es am Wochenende sonnig und trocken, mit Temperaturen bis zu 26 "
    "Grad. Am Sonntag ziehen am Nachmittag ein paar Wolken auf, es bleibt aber trocken und "
    "warm. Soll ich dir auch noch die Vorhersage für die kommende Woche heraussuchen?"
)
WEATHER_RATE = 22 / 3.76


def _frame(duration: float) -> rtc.AudioFrame:
    samples = round(duration * SAMPLE_RATE)
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=samples,
    )


def _hyphens(word: str) -> int:
    return len(tokenize.basic.hyphenate_word(word))


class _Turn:
    def __init__(self) -> None:
        self.sink = FakeAudioOutput()
        self.sync = TranscriptSynchronizer(next_in_chain_audio=self.sink, next_in_chain_text=None)
        self.audio, self.text = self.sync.audio_output, self.sync.text_output
        self.finished: list[PlaybackFinishedEvent] = []
        self.audio.on("playback_finished", self.finished.append)

    async def push(self, text: str, audio: float) -> None:
        await self.text.capture_text(text)
        await self.audio.capture_frame(_frame(audio))

    async def stream(self, text: str, *, in_step: bool) -> None:
        first = True

        async def push(word: str, seconds: float) -> None:
            nonlocal first
            await self.push(word, audio=seconds)
            if first and in_step:
                self.audio.mark_transcript_in_step()
            first = False

        await _stream(text, rate=RATE, push=push)
        self.end_generation()

    def end_generation(self) -> None:
        self.text.flush()
        self.audio.flush()

    def cut_mid_generation(self) -> None:
        # the order a cancelled generation tears its forwarding down in
        self.audio.flush()
        self.audio.clear_buffer()
        self.text.flush()
        self.audio.clear_buffer()

    def transcript(self) -> str:
        assert len(self.finished) == 1
        transcript = self.finished[0].synchronized_transcript
        assert transcript is not None
        return transcript


async def _stream(text: str, *, rate: float, push: Callable[[str, float], Awaitable[None]]) -> None:
    """Push a reply word by word, each word's text just ahead of its audio, at 2.8x real time,
    as a realtime model generates them."""
    for i, word in enumerate(text.split(" ")):
        seconds = _hyphens(word) / rate
        await push(word if i == 0 else f" {word}", seconds)
        await asyncio.sleep(seconds / 2.8)


async def _cut_mid_stream(text: str, *, at: float, in_step: bool) -> str:
    turn = _Turn()
    streaming = asyncio.create_task(turn.stream(text, in_step=in_step))
    try:
        while turn.sink._started_at is None:
            await asyncio.sleep(0)
        await asyncio.sleep(at)
        assert not streaming.done(), "the reply finished streaming before the cut"
        streaming.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await streaming
        turn.cut_mid_generation()
        return turn.transcript()
    finally:
        await turn.sync.aclose()


async def test_a_cut_mid_stream_keeps_the_words_its_audio_played() -> None:
    transcript = await _cut_mid_stream(REPLY, at=1.0, in_step=True)

    # 6.3 hyphens played: "Berlin" has started, "stays" has not
    assert REPLY.startswith(transcript)
    assert abs(len(transcript.split()) - len("Sure, the weekend in Berlin".split())) <= 1, (
        transcript
    )


async def test_a_cut_inside_the_first_word_keeps_that_word() -> None:
    reply = "Certainly, let me look that up for you."  # "Certainly," is 3 hyphens, 0.48s

    assert await _cut_mid_stream(reply, at=0.2, in_step=True) == "Certainly,"


async def test_a_transcript_far_ahead_of_its_audio_counts_no_faster_than_speech() -> None:
    # early on, a transcript three words ahead of 0.25s of audio reads as 16 hyphens/s
    turn = _Turn()
    try:
        await turn.push("Klar, in Würzburg wird", audio=0.25)
        turn.audio.mark_transcript_in_step()
        await asyncio.sleep(0.2)
        turn.cut_mid_generation()

        transcript = turn.transcript()
        assert abs(len(transcript.split()) - len("Klar, in".split())) <= 1, transcript
    finally:
        await turn.sync.aclose()


async def test_a_cut_without_an_in_step_transcript_keeps_the_paced_text() -> None:
    # an LLM's text runs ahead of its TTS: the whole reply has arrived with 1s of audio, so
    # the text and audio pushed are no measure of what played
    turn = _Turn()
    try:
        await turn.push(REPLY, audio=1.0)
        turn.text.flush()
        await asyncio.sleep(0.5)
        turn.cut_mid_generation()

        # the default pace has released at most "Sure,"; the pushed text would give 11 hyphens
        assert len(turn.transcript().split()) <= 1
    finally:
        await turn.sync.aclose()


@pytest.mark.parametrize("cut_at_the_end", [False, True])
async def test_a_turn_played_to_its_end_keeps_its_whole_text(cut_at_the_end: bool) -> None:
    reply = "I can help with that."  # 5 hyphens in 1s
    turn = _Turn()
    try:
        await turn.push(reply, audio=1.0)
        turn.audio.mark_transcript_in_step()
        turn.end_generation()

        if cut_at_the_end:
            await asyncio.sleep(0.95)  # "that." has started
            turn.audio.clear_buffer()
        else:
            await asyncio.sleep(1.2)

        assert turn.transcript() == reply
    finally:
        await turn.sync.aclose()


async def _realtime_cut(*, at: float, server_cancels_first: bool, in_step: bool = True) -> str:
    """Cut a realtime reply ``at`` seconds into its playout while the model still streams it."""
    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_transcript_in_step=in_step))
    sink = FakeAudioOutput()
    sync = TranscriptSynchronizer(next_in_chain_audio=sink, next_in_chain_text=None)
    text_ch, audio_ch = utils.aio.Chan[str](), utils.aio.Chan[rtc.AudioFrame]()

    async def push(word: str, seconds: float) -> None:
        text_ch.send_nowait(word)
        audio_ch.send_nowait(_frame(seconds))

    async with AgentSession(llm=model) as session:
        session.output.audio = sync.audio_output
        session.output.transcription = sync.text_output
        await session.start(Agent(instructions="You are a helpful assistant."))

        reply = session.generate_reply()
        while not model.active_session._reply_futs:
            await asyncio.sleep(0)
        message_ch = utils.aio.Chan[MessageGeneration]()
        function_ch = utils.aio.Chan[FunctionCall]()
        modalities = asyncio.Future[list[str]]()
        modalities.set_result(["audio", "text"])
        message_ch.send_nowait(
            MessageGeneration(
                message_id="message-id",
                text_stream=text_ch,
                audio_stream=audio_ch,
                modalities=modalities,
            )
        )
        message_ch.close()
        function_ch.close()
        model.active_session._reply_futs[0].set_result(
            GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=True,
                response_id="response-id",
            )
        )

        streaming = asyncio.create_task(_stream(WEATHER, rate=WEATHER_RATE, push=push))
        while sink._started_at is None:
            await asyncio.sleep(0)
        await asyncio.sleep(at)
        assert not streaming.done(), "the reply finished streaming before the cut"
        streaming.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await streaming

        if server_cancels_first:
            # the server's own turn detection cancels the response: its response.done closes
            # the streams before the local interruption cancels the forwarding
            text_ch.close()
            audio_ch.close()
            await asyncio.sleep(0.05)
        reply.interrupt(force=True)
        await asyncio.wait_for(reply, timeout=10)
    await sync.aclose()

    # a cut that kept no text stores no message
    texts = [item.text_content for item in reply.chat_items if item.type == "message"]
    return texts[0] or "" if texts else ""


@pytest.mark.parametrize("server_cancels_first", [False, True])
@pytest.mark.parametrize(
    ("at", "played"),
    [
        (0.2, "Klar, in"),  # 1.2 hyphens: "in" started at 0.17s
        (2.8, "Klar, in Würzburg wird es am Wochenende sonnig und trocken, mit Temperaturen"),
    ],
)
async def test_a_realtime_cut_mid_stream_keeps_the_words_its_audio_played(
    at: float, played: str, server_cancels_first: bool
) -> None:
    transcript = await _realtime_cut(at=at, server_cancels_first=server_cancels_first)

    assert WEATHER.startswith(transcript)
    assert abs(len(transcript.split()) - len(played.split())) <= 1, transcript


@pytest.mark.parametrize(
    ("at", "paced"),
    [
        (0.2, ""),  # "Klar," takes 0.26s at the default pace
        (2.8, "Klar, in Würzburg wird es am Wochenende sonnig"),  # 10.7 hyphens, 4 words short
    ],
)
async def test_a_realtime_model_without_an_in_step_transcript_keeps_the_paced_text(
    at: float, paced: str
) -> None:
    transcript = await _realtime_cut(at=at, server_cancels_first=False, in_step=False)

    assert WEATHER.startswith(transcript)
    assert len(transcript.split()) <= len(paced.split()), transcript
