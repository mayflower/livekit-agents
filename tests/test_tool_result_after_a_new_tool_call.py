"""The caller corrects a lookup: the model calls a new tool instead of answering the last
result, and Gemini 3.8 says nothing until that new tool's result arrives.

Local call 2026-10-16 (dev-martin-vds-web-call-1792162690670): the weather result came
back at 14:58:57 and reached the model only at 14:59:12, after the caller asked again.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, function_tool, utils
from livekit.agents.llm import ChatContext, FunctionCall, MessageGeneration
from livekit.agents.voice.speech_handle import INTERRUPTION_TIMEOUT

from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel, fake_capabilities
from .test_realtime_agent_state_during_tool import _generation
from .test_tool_result_while_speaking import _next_reply_fut, _sent_outputs

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


async def _results_after_a_new_tool_call(
    *, reaction_text: str, dates_asked_before: bool = False, interrupt_the_send: bool = False
) -> tuple[list[str], list[str]]:
    """What was sent 0.5 s after the reaction's tool returned, and once the model answered."""
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(takes_tool_results_while_speaking=True)
    )
    release = {"contacts": asyncio.Event(), "dates": asyncio.Event()}

    class ThreeToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You are a helpful assistant.")

        @function_tool
        async def contact_search(self) -> str:
            """Search the contacts."""
            await release["contacts"].wait()
            return "no contact found"

        @function_tool
        async def training_dates(self) -> str:
            """Look up the training dates."""
            await release["dates"].wait()
            return "testing on 26 November"

        @function_tool
        async def weather(self) -> str:
            """Look up the weather."""
            await asyncio.sleep(3.0)
            return "rain on Saturday"

    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(ThreeToolAgent())

        calls = [("contacts", "contact_search")]
        if dates_asked_before:
            calls.append(("dates", "training_dates"))
        for index, (call_id, name) in enumerate(calls):
            session.generate_reply()
            (await _next_reply_fut(model, index)).set_result(
                _generation(
                    response_id=call_id,
                    text="one moment",
                    audio_duration=0.5,
                    function_calls=[FunctionCall(call_id=call_id, name=name, arguments="{}")],
                )
            )
            await asyncio.sleep(1.0)

        release["contacts"].set()
        await asyncio.sleep(0.5)
        assert _sent_outputs(model) == ["contacts"]
        release["dates"].set()

        # the caller asks for the weather: the model reacts with a new tool call, which
        # is no answer to the contact result, and then waits for the weather result
        reaction = _generation(
            response_id="reaction",
            text=reaction_text,
            audio_duration=1.0,
            function_calls=[FunctionCall(call_id="weather", name="weather", arguments="{}")],
        )
        reaction.user_initiated = False
        if not reaction_text:
            # No zero-sample audio frame: the provider closes an empty stream.
            empty_messages = utils.aio.Chan[MessageGeneration]()
            empty_messages.close()
            reaction.message_stream = empty_messages
        rt_session = model.active_session
        if interrupt_the_send:
            update_chat_ctx = rt_session.update_chat_ctx

            async def slow_update_chat_ctx(chat_ctx: ChatContext) -> None:
                # longer than an interrupted speech may take to finish before it is cancelled
                await asyncio.sleep(INTERRUPTION_TIMEOUT + 1.0)
                await update_chat_ctx(chat_ctx)

            rt_session.update_chat_ctx = slow_update_chat_ctx  # type: ignore[method-assign]
        handles = []
        session.on("speech_created", lambda ev: handles.append(ev.speech_handle))
        rt_session.emit("generation_created", reaction)
        if interrupt_the_send:
            # the caller cuts in while the weather result is on its way to the model
            await asyncio.sleep(3.2)
            handles[-1].interrupt(force=True)
            await asyncio.sleep(0.3)
        else:
            await asyncio.sleep(3.5)
        soon = _sent_outputs(model)

        answer = _generation(response_id="answer", text="it will rain", audio_duration=1.0)
        answer.user_initiated = False
        model.active_session.emit("generation_created", answer)
        await asyncio.sleep(10.0)
        return soon, _sent_outputs(model)


async def test_a_new_tool_calls_result_is_sent_while_the_last_result_has_no_answer() -> None:
    soon, after = await _results_after_a_new_tool_call(reaction_text="")
    assert soon == ["contacts", "weather"]
    assert after == ["contacts", "weather"]


async def test_a_new_tool_call_spoken_with_text_gets_its_result_at_once() -> None:
    soon, after = await _results_after_a_new_tool_call(reaction_text="Jetzt prüfe ich das Wetter")
    assert soon == ["contacts", "weather"]
    assert after == ["contacts", "weather"]


async def test_a_result_asked_for_before_the_held_one_still_waits_for_an_answer() -> None:
    soon, after = await _results_after_a_new_tool_call(
        reaction_text="Jetzt prüfe ich das Wetter", dates_asked_before=True
    )
    assert soon == ["contacts", "weather"]
    assert after == ["contacts", "weather", "dates"]


async def test_a_new_tool_calls_result_survives_an_interruption_of_its_send() -> None:
    _, after = await _results_after_a_new_tool_call(
        reaction_text="Jetzt prüfe ich das Wetter", interrupt_the_send=True
    )
    assert after == ["contacts", "weather"]
