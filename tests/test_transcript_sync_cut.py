"""A cut keeps the words its played audio holds.

The synchronizer paces the transcript at a default speech rate until both the text and the
audio of a segment have ended, and counts a word only once its whole paced delay has passed.
A cut took that lagging progress as the segment's text: a realtime voice speaking faster than
the default rate lost the words played since, and a cut inside the first word kept nothing.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from livekit import rtc
from livekit.agents import tokenize
from livekit.agents.voice.io import PlaybackFinishedEvent
from livekit.agents.voice.transcription.synchronizer import TranscriptSynchronizer

from .fake_io import FakeAudioOutput

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time]

SAMPLE_RATE = 24000

# 22 hyphens in 3.5s: 6.3/s against the default 3.83/s
REPLY = "Sure, the weekend in Berlin stays sunny and dry, with highs of up to twenty-six degrees."
RATE = 22 / 3.5


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
        """Push a reply word by word, its text just ahead of its audio, at 2.8x real time."""
        for i, word in enumerate(text.split(" ")):
            seconds = _hyphens(word) / RATE
            await self.push(word if i == 0 else f" {word}", audio=seconds)
            if i == 0 and in_step:
                self.audio.mark_transcript_in_step()
            await asyncio.sleep(seconds / 2.8)
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
