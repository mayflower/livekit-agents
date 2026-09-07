# This fork

A fork of [livekit/agents](https://github.com/livekit/agents) carrying fixes that
[mayflower/voice-demo-solution](https://github.com/mayflower/voice-demo-solution)
needs before they land upstream.

## Layout

| Branch | What it is |
|---|---|
| `main` | upstream's `main`, plus this file and the rebase workflow. Nothing else — no fixes live here. |
| `mayflower/<version>` | upstream release tag `livekit-agents@<version>`, plus our commits. This is what gets consumed. |

The patch branch is cut from a **release tag**, never from `main`, so the code we
run differs from a published release only by our own commits — and a PR opened
from it against upstream carries exactly those.

The maintenance workflow deliberately lives on `main` rather than on the patch
branch. GitHub only fires `schedule:` for workflows on the default branch, and
keeping it there is also what keeps it out of the diff an upstream PR would show.

## Consuming it

`voice-demo-solution` pins the branch head by rev in
`livekit-agent/pyproject.toml` under `[tool.uv.sources]`. Rev, not branch name:
a moving ref would break `uv sync --locked`.

Only `livekit-agents` is changed. uv nonetheless resolves the sibling plugins out
of this repo's workspace rather than PyPI — a workspace member wins over an
`{ index = "pypi" }` source — which is harmless, since they are the monorepo's own
unmodified release.

## Moving to a newer upstream release

`Rebase onto the newest upstream release` runs weekly, and on demand from the
Actions tab. It rebases the newest `mayflower/*` branch onto the newest release
tag, pushes `mayflower/<new>`, and opens an issue with the rev to pin — or, if the
rebase conflicts, an issue saying so and nothing pushed. By hand it is:

```bash
# the version marker is rewritten, not rebased: it edits version.py, which
# every release also bumps, so carrying it across conflicts every time
git checkout -B mayflower/<new> mayflower/<old>^
git rebase --onto livekit-agents@<new> livekit-agents@<old>
# then set __version__ to "<new>+mayflower.1" and commit it afresh
```

That means the marker has to stay the **tip** commit of the patch branch; the
workflow refuses to run if it is not.

A clean rebase is not a passing test. `voice-demo-solution` hashes the source of
every framework function it wraps (`lib/framework_patches.py`), so check those and
run both suites before pinning.

Two things about Actions on a fork: they are disabled until someone enables them,
and GitHub suspends scheduled workflows after 60 days without repository activity.

## What is patched

See the commits on the patch branch. Each is written to be cherry-picked into an
upstream PR unchanged; the only commit that should not go upstream is the local
version marker (`<version>+mayflower.1`), which exists so the lock file and the
worker's startup log name which build is running.
