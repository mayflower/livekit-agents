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


async def _second_result_sent(
    *, model_answers_the_first: bool, interrupt_the_second: bool = False
) -> tuple[bool, bool]:
    """Whether the second result went out while the answer to the first played, and after."""
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(takes_tool_results_while_speaking=True)
    )
    release = {"dates": asyncio.Event(), "contacts": asyncio.Event()}

    class TwoToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You are a helpful assistant.")

        @function_tool
        async def training_dates(self) -> str:
            """Look up the training dates."""
            await release["dates"].wait()
            return "testing on 26 November"

        @function_tool
        async def contact_search(self) -> str:
            """Search the contacts."""
            await release["contacts"].wait()
            return "Erika Muster"

    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(TwoToolAgent())

        handles = []
        for index, (call_id, name) in enumerate(
            [("dates", "training_dates"), ("contacts", "contact_search")]
        ):
            handles.append(session.generate_reply())
            (await _next_reply_fut(model, index)).set_result(
                _generation(
                    response_id=call_id,
                    text="one moment",
                    audio_duration=0.5,
                    function_calls=[FunctionCall(call_id=call_id, name=name, arguments="{}")],
                )
            )
            await asyncio.sleep(1.0)

        release["dates"].set()
        await asyncio.sleep(0.5)
        assert _sent_outputs(model) == ["dates"]
        if model_answers_the_first:
            answer = _generation(response_id="answer", text="the dates are", audio_duration=10.0)
            answer.user_initiated = False
            model.active_session.emit("generation_created", answer)
            await asyncio.sleep(0.5)

        release["contacts"].set()
        if interrupt_the_second:
            await asyncio.sleep(0.1)
            # the answer to the first opening interrupts the speech that waits to send
            handles[1].interrupt(force=True)
        await asyncio.sleep(2.0)
        during = "contacts" in _sent_outputs(model)
        await asyncio.sleep(10.0)
        return during, "contacts" in _sent_outputs(model)


async def test_the_next_result_waits_until_the_answer_to_the_last_has_played() -> None:
    assert await _second_result_sent(model_answers_the_first=True) == (False, True)


async def test_the_next_result_goes_out_when_the_last_gets_no_answer() -> None:
    during, after = await _second_result_sent(model_answers_the_first=False)
    assert after


async def test_the_next_result_survives_an_interruption_of_its_speech() -> None:
    assert await _second_result_sent(model_answers_the_first=True, interrupt_the_second=True) == (
        False,
        True,
    )
