# This fork

A fork of [livekit/agents](https://github.com/livekit/agents) carrying fixes that
[mayflower/voice-demo-solution](https://github.com/mayflower/voice-demo-solution)
needs before they land upstream.

## Layout

| Branch | What it is |
|---|---|
| `main` | upstream's `main`, plus this file and the rebase workflow. Nothing else — no fixes live here. |
| `mayflower/<version>` | upstream release tag `livekit-agents@<version>`, plus the fixes. **Open upstream PRs from here.** |
| `build/<version>` | `mayflower/<version>` plus the local version marker. **This is what gets pinned.** |

The patch branch is cut from a **release tag**, never from `main`, so the code we
run differs from a published release only by our own commits — and a PR opened
from it against upstream carries exactly those.

The version marker is kept on a separate branch rather than on the patch branch
for two reasons: it edits `version.py`, so it would show up in the diff of every
upstream PR, and every release bumps that same file, so it would conflict on
every rebase. Split off, the patch branch rebases clean and PRs carry only the
fix.

The maintenance workflow deliberately lives on `main` rather than on the patch
branch. GitHub only fires `schedule:` for workflows on the default branch, and
keeping it there is also what keeps it out of the diff an upstream PR would show.

## Consuming it

`voice-demo-solution` pins the head of **`build/<version>`** by rev in
`livekit-agent/pyproject.toml` under `[tool.uv.sources]`. Rev, not branch name:
a moving ref would break `uv sync --locked`.

Only `livekit-agents` and `livekit-plugins-google` are changed. uv resolves every
sibling plugin out of this repo's workspace rather than PyPI — a workspace member
wins over an `{ index = "pypi" }` source — which is what carries the Google fixes,
and is harmless for the rest, since they are the monorepo's own unmodified release.

## Moving to a newer upstream release

`Rebase onto the newest upstream release` runs weekly, and on demand from the
Actions tab. It rebases the newest `mayflower/*` branch onto the newest release
tag, pushes it along with a matching `build/<new>`, and opens an issue with the
rev to pin — or, if the rebase conflicts, an issue saying so and nothing pushed.
By hand it is:

```bash
git checkout -B mayflower/<new> mayflower/<old>
git rebase --onto livekit-agents@<new> livekit-agents@<old>

git checkout -B build/<new> mayflower/<new>
# set __version__ to "<new>+mayflower.1", commit, and pin that rev
```

A clean rebase is not a passing test. Run upstream's unit suite here against the
pristine tag and the rebased branch, then `voice-demo-solution`'s agent suite, as
described under "Verifying a port" below.

Three things about a fork: Actions are disabled until someone enables them, GitHub
suspends scheduled workflows after 60 days without repository activity, and Issues
are off by default — the workflow reports through them, so `gh issue create` fails
with "Resource not accessible by integration" until they are switched on in the
repository settings.

## What is patched

Eleven commits against six defect classes: **a finished tool result never
reaching the caller** (1–4), **a Gemini session dropped for no reason**
(5–6), **a finished reply the session will not let out** (7), **a caller
transcript that is not the one the session asked for** (8, 11), **a billed
token nobody counts** (9), and **a request the provider is given nothing to
answer** (10). Read this before
porting them to a new release — a clean `git rebase` says the text still
applies, not that the reasoning does.

### 1. Concurrent chat-context writers lose each other's items

*Files: `llm/realtime.py`, `voice/agent.py`, `voice/agent_activity.py`,
`voice/tool_executor.py`, `llm/chat_context.py`*

`RealtimeSession.update_chat_ctx(ctx)` is declarative: the caller submits the
whole conversation and the session removes whatever the submission leaves out.
Callers that only want to *add* something therefore read the current context,
append, and submit — and anything that lands between that read and the write is
deleted as though its removal had been requested. The plugin's own
`_update_chat_ctx_lock` cannot prevent it: it covers the write, not the read.

It costs a finished async tool its result. `_ToolExecutor._enqueue_reply` writes
the synthetic `{call_id}_final` pair, and the turn that dispatched the tool then
re-syncs the whole conversation from a snapshot taken before that write landed.

The fix adds `update_chat_ctx_with(producer)` at three levels —
`RealtimeSession`, `AgentActivity`, `Agent` — running the producer under the
same lock as the write, and converts the four framework writers to it.
`update_chat_ctx` keeps its declarative meaning for the one caller that means
"make it exactly this" (the never-played-item cleanup), which now goes through a
producer that ignores the current context so it serialises rather than races.

It also carries a second, subtler half. A provider's wire format for a function
*output* has no name field, so an output read back from a session has lost it
(`openai_item_to_livekit_item` rebuilds one without). `ChatContext.copy(tools=…)`
judged an output by that name and dropped it as foreign — so any update derived
from the session's context was missing every earlier result. An output is now
judged by its call. **Do not drop this half when porting**: deriving the update
from the session is what makes the first half work, and it is exactly what
exposes the name loss.

*Porting checks:* does `update_chat_ctx` still remove what a submission omits?
Do the four writers still read the context themselves? Does
`openai_item_to_livekit_item` still build a `FunctionCallOutput` without a
`name`? If any answer is no, the upstream shape has changed — re-read before
carrying the commit.

### 2. A reply is issued while the provider is still generating

*Files: `llm/realtime.py`, `voice/agent_activity.py`*

Several providers keep one response in flight and refuse a second; OpenAI
answers `response.create` with `conversation_already_has_active_response`. The
reply that speaks a finished tool result is issued when the framework's *local*
speech queue is empty, which says nothing about the provider — so it races the
acknowledgement the same tool triggered, and loses.

Losing is terminal: `_realtime_reply_task` logs, marks the speech done and
returns, and `_deliver_reply` has already cleared its pending updates. Upstream
leaves a `TODO(long): reschedule interrupted replies?` where a retry would go.

`has_active_generation` already existed on the OpenAI plugin; the fix lifts it to
the base with a `False` default (so a plugin that cannot tell keeps today's
behaviour), adds `wait_for_response_slot`, and waits before issuing —
interruptibly, and giving up rather than raising, because a refused reply is
visible while an unissued one is silent.

*Porting checks:* is the refusal still terminal upstream (is the `TODO` still
there)? Has `has_active_generation` moved to the base or gained a real event? If
upstream adds a retry or an await, prefer theirs and drop this.

### 3. A realtime turn never records that the user is speaking

*Files: `voice/agent_activity.py`*

`AgentSession` builds a default VAD, but `AgentActivity` de-wires it when the
model does its own turn taking — so `AudioRecognition._speaking`, fed only by the
VAD and STT streams, stays False for the whole call, and `_user_silence_event` is
never cleared. `_on_input_speech_started` updates `user_state` and calls
`_on_start_of_speech`, but that helper sets neither.

Two gates read exactly those two and both go inert:
`wait_for_idle(wait_for_user=True)` returns while the caller is mid-sentence, so
a deferred tool reply is spoken over them; and playout authorization waits on
`_user_silence_event`, which is what stops a queued reply talking over a new user
turn in the STT pipeline.

The handlers now set both, behind the guard they already use for the rest of
their bodies. Owning the signals means owning the latch: a start whose stop never
arrives would mute the agent for the rest of the call, so there are two backstops
— `session_reconnected`, and a timeout for anything else.

*Porting checks:* does `AgentActivity` still de-wire the VAD for realtime turn
detection? Is `AudioRecognition._speaking` still a settable property, and does
`_on_start_of_speech` still leave it alone? Do the two gates still read these
signals? If upstream starts feeding them itself, drop this.

### 4. A reply that cannot even be requested is invisible

*Files: `voice/tool_executor.py`*

`_deliver_reply` calls `session.generate_reply(...)` with no `try`, and
`_create_speech_task` attaches no error handler — so a raise there surfaces only
as asyncio's "Task exception was never retrieved", which structured logging does
not show. `_pending_updates` is already cleared, so there is nothing to retry
with.

Every in-tree plugin reports failure through the returned future, where
`AgentActivity` catches it, which is why upstream has never felt this. An
out-of-tree plugin that raised synchronously cost two finished tool calls in one
call, silently.

The results survive regardless — `_enqueue_reply` puts them in the chat context
first — so what the log names is which calls the caller did not hear. It
deliberately does not retry: a retry competes for the same floor as the next
acknowledgement and loses.

*Porting check:* is the call still unguarded, and does `_create_speech_task`
still attach no handler? If upstream adds either, drop this.

### 5. A rebuilt tool reconnects a session the API sees no change in

*Files: `livekit-plugins-google/.../realtime/realtime_api.py`*

The Gemini realtime session cannot mutate its tools in place, so `update_tools`
reconnects — gated on `ToolContext.__eq__`, which compares tools by **object
identity**: the right check for a session that can push a diff, the wrong one
here. The caller cannot work around it either. A `@function_tool`-decorated
*method* is a new object on every attribute access, so an agent that assembles
its tool list per push pays a reconnect every time for a set the API cannot
tell from the running one.

The fix compares what is actually sent: the `types.Tool` list
`_build_connect_config` puts in the connect config. Equal declarations adopt
the new objects — their handlers are the ones the caller wants run — and return
without a restart. Ultravox already compares tool names rather than identity,
so Google is the outlier. Widening `ToolContext.__eq__` instead has the bigger
blast radius: `_sync_flattened` depends on its `id()` semantics.

*Porting check:* does `update_tools` still gate the restart on `ToolContext`
equality alone? If upstream starts comparing declarations, or the Live API
gains a tool update, drop this.

### 6. A reconnect that fails once ends the call

*Files: `livekit-plugins-google/.../realtime/realtime_api.py`*

`_main_task` treats any failure raised before the socket is up as fatal:
`if not session` — an unconnected session means bad parameters, not worth a
retry. That holds for the first connect only. Every later one is a *reconnect*
of a session that already worked, so the parameters are proven, the failure is
the server's, and giving up costs the whole conversation.

Gemini answered a reconnect for a tool update with `1011 Internal error`, the
plugin raised `APIConnectionError`, and `AgentSession` closed mid-call. The fix
records whether a connect ever succeeded and lets the existing bounded retry
handle the rest — three attempts, 4.1s of silence at worst.

It does not cover everything a reconnect can fail on: the retry re-sends the
same `_session_resumption_handle`, so a handle the server has stopped accepting
fails identically three times and the call ends anyway. Dropping the handle on
the last attempt would trade the history for the call; nothing has needed it yet.

*Porting check:* does the guard still read `if not session or max_retries == 0`?
If upstream starts telling the first connect apart itself, drop this.

### 7. A playout interrupt is booked as a caller turn

*Files: `llm/realtime.py`, `voice/agent_activity.py`,
`livekit-plugins-google/.../realtime/realtime_api.py`*

`input_speech_started` is the only channel a realtime session has for "stop
what you are saying", so the Gemini plugin emits it before every
agent-initiated generation — the branch where `_pending_generation_fut` is
unset. Nothing was heard from the caller. Fix 3 above made the session record
that event as a turn, which closes the user-speaking latch, and the matching
stop comes from `_mark_current_generation_done` — it lands when *that
generation* completes. Playout authorization waits on the latch, so the
agent's own reply waits out its own generation before it may play.

Measured on one dev call: 6.7s, 10.4s and 2.8s of silence before three tool
answers, with the audio probe recording no audible bucket and Deepgram no
speech onset for the whole window. A fourth answer was cut off after 1.07s,
because a second tool result opened a second generation whose synthetic event
interrupted the first — the caller heard "Ich habe unter dem" and then a fresh
answer saying nothing was found.

`InputSpeechStartedEvent` gains `speech_detected`, default True. Every other
plugin emits this event only from real detection — OpenAI from
`InputAudioBufferSpeechStartedEvent`, hugging-voice from `SpeechStartedEvent` —
so the default is what they already mean, and only Google passes False. The
handler then interrupts and records nothing; the stop mirrors it, so it never
ends a turn nobody started, and the two backstops from fix 3 clear the pairing
flag with the latch.

This is worth upstreaming ahead of fix 3: without the latch the event still
mislabels `user_state`, which is what a trace and a Langfuse span read.

*Porting checks:* does the plugin still emit `input_speech_started` on the
`else` branch of the pending-generation check? Does anything else now emit it
without detection? If upstream gives the plugins a real interrupt channel, drop
this and the plugin's synthetic emit with it.

### 9. A realtime session's reasoning tokens are billed but never counted

*Files: `metrics/base.py`, `metrics/usage.py`,
`livekit-plugins-google/.../realtime/realtime_api.py`*

Gemini reports what a session spent on hidden reasoning in
`usage_metadata.thoughts_token_count`, and leaves it out of
`response_token_count`. It is billed all the same.
`_handle_usage_metadata` read the response count and nothing else, so the
reasoning left no trace in anything downstream: a consumer summing the reported
usage undercounts the output by exactly the thinking.

Probed against the Live API from a running pod on 2026-09-21, one turn with a
tool call, every `usage_metadata` frame logged:

```
gemini-3.8-live-extended-thinking   response=46   thoughts=140
                                    response=246  thoughts=496
gemini-3.8-live                     response=14   thoughts=124
                                    response=172  thoughts=111
```

A second probe, three one-word turns in a single `gemini-3.8-live` session, added
three more frames: `response` stayed flat at 28/25/26 and `thoughts` at 60/66/66
while only `prompt` grew, which is the context growing and not an accumulating
counter — the collector is right to sum frames with `+=`.

Note the second model. **Plain `gemini-3.8-live` thinks and bills for it with no
`thinking_config` at all** — Extended Thinking is a matter of degree, not of
kind. Nothing here may key on a model name or on whether thinking was
configured; the field is read whenever the provider sends it.

Three layers had to carry it, and none did. `RealtimeModelMetrics.OutputTokenDetails`
had buckets for text, audio and image only. `ModelUsageCollector` fills
`LLMModelUsage.output_reasoning_tokens` in the `LLMMetrics` branch alone; the
realtime branch never touched it. And the plugin never read the field.

`OutputTokenDetails` gains `reasoning_tokens`, the realtime branch of the
collector maps it onto `output_reasoning_tokens`, and the plugin fills it. The
plugin also adds the thinking into `output_tokens`, because reasoning is
documented — and asserted in `tests/test_metrics_usage.py` for the LLM path — as
a *subset* of `output_tokens`, never an addition to it; a provider whose response
count excludes thinking has to add it in before reporting. The probe confirms the
exclusion is real and not merely documented: in two of the frames above
`thoughts` exceeds `response` outright (140 > 46, 124 > 14), which it could not
if the response count already contained it. It stays out of `text_tokens` and
`audio_tokens`: nobody heard or read it.

Nothing here is derived from `total_token_count`, deliberately. The SDK documents
it as the sum of prompt, candidates, tool-use and thoughts, but across two probes
that held in only one of seven frames — in the other six `prompt + response`
already equalled `total` and the thinking sat outside it, so anything reading
`total` as the sum of its parts loses the thinking entirely. Whatever that
inconsistency is, reading only `response_token_count` and `thoughts_token_count`
is unaffected by it.

This is not Gemini-specific. OpenAI's realtime plugin accepts a `reasoning`
config for models like `gpt-realtime-2` and reports no reasoning usage either, so
the same three layers now carry it for any realtime provider that fills the field.

*Porting checks:* does `UsageMetadata` still separate `thoughts_token_count` from
`response_token_count`? Does the realtime branch of `ModelUsageCollector` still
skip `output_reasoning_tokens`? Has anything started keying reasoning on a model
name or on `thinking_config`? If upstream begins reporting reasoning usage for
realtime sessions itself, drop this; if `OutputTokenDetails` gains an official
reasoning field, keep the plugin's read and drop the rest.

## Verifying a port

A clean rebase is not a passing test.

1. `uv sync --all-extras --dev && uv run pytest -p no:randomly --unit` here.
   Compare the failures against the same command on the *pristine* release tag —
   about a dozen fail either way because they need a LiveKit server or network
   access. What matters is that the two sets are identical.
   `tests/test_realtime_chat_ctx_atomicity.py` is this fork's own and must pass.
2. In `voice-demo-solution`: pin the new rev, `uv lock`, and run the agent suite
   (`cd livekit-agent && uv run --group test pytest`). Nothing there guards the
   framework's source any more, so re-read "What is patched" against the new
   release rather than trusting a green rebase.
3. `e2e-tests/tests/realtime-tool-result-retained.test.ts` drives a real realtime
   session and is the only test that exercises both fixes together. It is written
   to fail without them; its header says how that was checked.
