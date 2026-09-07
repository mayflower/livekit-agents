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

See the commits on the patch branch. Each is written to be cherry-picked into an
upstream PR unchanged — which is the whole point of keeping the version marker
(`<version>+mayflower.1`) on `build/*` instead. That marker exists so a running
worker's log names which build it is; the lock file records the exact rev anyway.
