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

Only `livekit-agents` is changed. uv nonetheless resolves the sibling plugins out
of this repo's workspace rather than PyPI — a workspace member wins over an
`{ index = "pypi" }` source — which is harmless, since they are the monorepo's own
unmodified release.

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

A clean rebase is not a passing test. `voice-demo-solution` hashes the source of
every framework function it wraps (`lib/framework_patches.py`), so check those and
run both suites before pinning.

Two things about Actions on a fork: they are disabled until someone enables them,
and GitHub suspends scheduled workflows after 60 days without repository activity.

## What is patched

Two commits, both against one defect class: **a finished tool result never
reaching the caller.** Read this before porting them to a new release — a clean
`git rebase` says the text still applies, not that the reasoning does.

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

## Verifying a port

A clean rebase is not a passing test.

1. `uv sync --all-extras --dev && uv run pytest -p no:randomly --unit` here.
   Compare the failures against the same command on the *pristine* release tag —
   about a dozen fail either way because they need a LiveKit server or network
   access. What matters is that the two sets are identical.
   `tests/test_realtime_chat_ctx_atomicity.py` is this fork's own and must pass.
2. In `voice-demo-solution`: `lib/framework_patches.py` hashes the source of every
   framework function it wraps, so a bump that touches one fails loudly. Re-read
   the upstream source before re-recording a hash.
3. `e2e-tests/tests/realtime-tool-result-retained.test.ts` drives a real realtime
   session and is the only test that exercises both fixes together. It is written
   to fail without them; its header says how that was checked.
