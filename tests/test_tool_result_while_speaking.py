"""A model that takes tool results while it speaks gets each one as its tool returns.

The caller asks one question, then a second while the first tool runs. The first tool
returns while the reply to the second is still playing. Held until that reply played, the
first result lands behind the second one, and Gemini answers only the result it got first.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, function_tool
from livekit.agents.llm import FunctionCall

from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel, fake_capabilities
from .test_realtime_agent_state_during_tool import _generation

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


def _sent_outputs(model: FakeRealtimeModel) -> list[str]:
    return [
        item.call_id
        for item in model.active_session.chat_ctx.items
        if item.type == "function_call_output"
    ]


async def _next_reply_fut(model: FakeRealtimeModel, index: int) -> asyncio.Future:
    for _ in range(500):
        if len(model.active_session._reply_futs) > index:
            return model.active_session._reply_futs[index]
        await asyncio.sleep(0)
    raise AssertionError("no reply was requested")


async def _first_result_sent_while_the_second_reply_plays(*, takes_results: bool) -> bool:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(takes_tool_results_while_speaking=takes_results)
    )
    release_first = asyncio.Event()

    class TwoToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You are a helpful assistant.")

        @function_tool
        async def training_dates(self) -> str:
            """Look up the training dates."""
            await release_first.wait()
            return "testing on 26 November"

        @function_tool
        async def contact_search(self) -> str:
            """Search the contacts."""
            await asyncio.sleep(30.0)
            return "Erika Muster"

    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(TwoToolAgent())

        first = session.generate_reply()
        (await _next_reply_fut(model, 0)).set_result(
            _generation(
                response_id="first",
                text="let me look up the dates",
                audio_duration=0.5,
                function_calls=[
                    FunctionCall(call_id="dates", name="training_dates", arguments="{}")
                ],
            )
        )
        await asyncio.sleep(1.0)

        second = session.generate_reply()
        (await _next_reply_fut(model, 1)).set_result(
            _generation(
                response_id="second",
                text="searching the contacts now",
                audio_duration=10.0,
                function_calls=[
                    FunctionCall(call_id="contacts", name="contact_search", arguments="{}")
                ],
            )
        )
        await asyncio.sleep(1.0)
        release_first.set()
        await asyncio.sleep(2.0)

        sent = "dates" in _sent_outputs(model)
        await asyncio.wait_for(first.wait_for_playout(), timeout=60)
        await asyncio.wait_for(second.wait_for_playout(), timeout=60)
        return sent


async def test_a_model_that_takes_results_while_speaking_gets_them_at_once() -> None:
    assert await _first_result_sent_while_the_second_reply_plays(takes_results=True)


async def test_other_models_get_results_once_every_speech_has_played() -> None:
    assert not await _first_result_sent_while_the_second_reply_plays(takes_results=False)
