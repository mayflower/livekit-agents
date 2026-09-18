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

Six commits against two defect classes: **a finished tool result never reaching
the caller** (1–4) and **a Gemini session dropped for no reason** (5–6). Read
this before porting them to a new release — a clean `git rebase` says the text
still applies, not that the reasoning does.

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
