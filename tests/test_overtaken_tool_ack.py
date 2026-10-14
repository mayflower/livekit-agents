"""A realtime turn does not announce a background tool that has already returned.

The turn's tool reply announces the wait of a tool released to the background,
and it waits for the speech before it to play first. A short tool can return in
that time; its result is then queued for the deferred reply, and the turn's
`interrupt()` would cancel that reply as it starts.
"""

from __future__ import annotations

import pytest

from livekit.agents import llm
from livekit.agents.voice.agent_activity import _only_acks_of_returned_tools
from livekit.agents.voice.generation import ToolExecutionOutput
from livekit.agents.voice.tool_executor import _RunningTasks

pytestmark = pytest.mark.unit

_NON_BLOCKING = "__livekit_agents_tool_non_blocking"


class _Session:
    """Only a key for `_RunningTasks`."""


def _output(call_id: str, *, released: bool, reply: bool = True) -> ToolExecutionOutput:
    call = llm.FunctionCall(call_id=call_id, name="web_search", arguments="{}")
    if released:
        call.extra[_NON_BLOCKING] = True
    out = llm.FunctionCallOutput(
        call_id=call_id, name="web_search", output="running", is_error=False
    )
    out.reply_required = reply
    return ToolExecutionOutput(
        fnc_call=call, fnc_call_out=out, agent_task=None, raw_output=None, raw_exception=None
    )


def test_the_ack_of_a_returned_tool_needs_no_announcement() -> None:
    session = _Session()

    assert _only_acks_of_returned_tools(session, [_output("c1", released=True)])  # type: ignore[arg-type]


def test_the_ack_of_a_tool_still_running_is_announced() -> None:
    session = _Session()
    _RunningTasks[session] = {"c1": object()}  # type: ignore[index, dict-item]

    assert not _only_acks_of_returned_tools(session, [_output("c1", released=True)])  # type: ignore[arg-type]


def test_a_tool_that_returned_its_result_inline_is_answered() -> None:
    assert not _only_acks_of_returned_tools(
        _Session(),  # type: ignore[arg-type]
        [_output("c1", released=False)],
    )


def test_one_output_that_still_needs_a_reply_keeps_it() -> None:
    session = _Session()

    assert not _only_acks_of_returned_tools(
        session,  # type: ignore[arg-type]
        [_output("c1", released=True), _output("c2", released=False)],
    )


def test_outputs_that_ask_for_no_reply_do_not_count() -> None:
    session = _Session()

    assert _only_acks_of_returned_tools(
        session,  # type: ignore[arg-type]
        [_output("c1", released=True), _output("c2", released=False, reply=False)],
    )
    assert not _only_acks_of_returned_tools(
        session,  # type: ignore[arg-type]
        [_output("c2", released=False, reply=False)],
    )
