## Never build an image from a working tree

This service is being deployed to the homelab, so this rule applies here from the start.

Build from a commit, in CI, on a machine that fetched that commit itself. Anything else is banned and gets deleted on sight — no discussion, no exceptions.

Banned: copying, tarring, rsyncing or scp-ing this source tree onto a build host; building an image from anything the builder did not pull from git itself; any build that needs a laptop, a checkout, or an SSH session to run; build scratch directories left on the VMs.

Do not copy `scripts/build-image.sh` from the openconv repo. It is the pattern this rule exists to kill: it tars the working tree, untracked files included, and ships it over SSH to a build node, producing images whose source nobody can identify.

Why: an image built from a working tree has no known source. It contains whatever was on someone's disk, records no commit, and cannot be reproduced, bisected, or audited.

Instead: build in CI from a commit, tag `YYYY.MM.DD.N`, push to the homelab registry, then file that tag into the homelab's `service-versions.auto.tfvars.json` so Atlantis deploys it.

### Nothing builds until you push to gitea

Merging a pull request on GitHub builds nothing and publishes nothing. No mirror, no poller, no schedule, no webhook. GitHub (`origin`) is the development and review surface and knows nothing about images. Nothing that happens on GitHub reaches this network. Images come from exactly two deliberate acts, both performed by a human or an agent, and the first is the one that matters:

    git push gitea master          # the normal path: publishes the dated tag and :latest

    # ...and dispatching the workflow by hand on any ref, which publishes that
    # ref's dated tag. It cannot move :latest — only master does that.

Rehearse the moment, because it is coming: the PR is merged, the checks are green, and you think *"the image should land in a few minutes — I'll go check the registry."* Nothing is coming. No run ever started. You can wait all afternoon and find an empty registry. After every merge to master, push to gitea, or there is no build. The remote is already in this clone — `gitea` → `ssh://git@gitea.sanctuary.gdn:2222/brandon-fryslie/elvenspeak.git` — and it is a build remote only; nothing ever flows back from it to GitHub. Because the trigger is a deliberate push, gitea is allowed to sit behind GitHub's master: that gap is not a mess to tidy up, it means nobody has asked for a build of those commits yet.

The workflow is `.gitea/workflows/publish-image.yaml`, and it runs on every ref you push. Three of its four jobs run on every ref. `reachability` proves the *builder* — that it fetched the commit itself, that its CPU can run the wheels the image installs (x86-64-v2), that jq and python3 are present, that a BuildKit-capable daemon is reachable, and that the registry both answers and accepts a write. `tests` runs the full pytest suite. Once both pass, `prove` takes each row of the `engine` matrix, builds the real Dockerfile with `docker build --load` into the runner's local Docker store as `elvenspeak-<engine>:proof-<run id>`, runs `smoke.py` against that image, and removes it. That name carries no registry host, so nothing can be pushed under it: `prove` never pushes, never spends a dated tag, and cannot move `:latest`. Only `refs/heads/master` and a manual `workflow_dispatch` run the fourth job, `publish`, which needs `reachability`, `tests` and `prove` all green. It resolves the dated tag, builds again with provenance labels, smokes that build again, pushes, then verifies the registry digest and the labels. So a publishing run builds and smokes every image twice. That is not duplication to trim: `publish` pushes its own build, not `prove`'s, and the second smoke is what proves the exact bytes it pushes.

Read a green branch run as exactly what it proves and no more. It proves that every engine's image, built from that commit by the real Dockerfile, boots, answers `/health` with 200, passes its own HEALTHCHECK inside the container, and passes `speaks.py` conformance, real synthesis included — so a broken `RUN`, a missing apt package or a failed asset download turns the branch red at `prove`, before anything lands. It does not prove the steps only `publish` runs: resolving the tag, the push, verifying the digest, verifying the labels. Those first run when master is pushed or the workflow is dispatched, so the `publish` run is the one to read for them, never the branch's green. And do not credit `reachability` with any of it: its BuildKit check builds a synthetic busybox image, not the real Dockerfile.

To prove a Dockerfile change before it lands, push the branch to gitea: `git push gitea <branch>`. That runs `prove` on every engine and spends no tag, but it is not free — it costs runner time, and chatterbox dominates it. Measured on this 2-cpu runner on 2026-09-10, per engine leg: a passing smoke took chatterbox 624s, kokoro 178s, piper 45s and router 10s, and a change to any layer at or above the Dockerfile's bake step cost the chatterbox build 582s, 538.8s of it a single `chown -R` (filed as piper-build-7uh). So a push that touches `elvenspeak/` costs the chatterbox leg roughly twenty minutes. Rehearse the moment, because twenty minutes invites it: the push touches nothing that could reach an image, and you think *"a path filter would spare the runner this — skip `prove` when the Dockerfile and the code are untouched."* No path filter or conditional skip applies, and none gets added. This project decided against checks that can be skipped, because a skip is indistinguishable from a pass: the day the filter guesses wrong, the run is exactly as green as one that proved something. Dispatching the workflow on a branch is still a real publish, and what it spends depends on how the smoke ends. When the smoke passes, the dispatch consumes a dated tag that is then spent forever. When the smoke refuses the image, nothing is pushed and no tag is spent — observed on gitea runs 55 and 57 on 2026-09-10, which both went red at the smoke and left the registry holding only 2026.09.10.1, its `:latest` digests unchanged. A dispatch builds and smokes from the dispatched ref's own tree, so a fix to a CI script is proven by dispatching a branch based on it. It does not move `:latest` — the tag step only adds that alias when the ref is `refs/heads/master`, so an unreviewed build cannot become what a break-glass `nomad job run` with no `-var` picks up. That guard is enforced in the workflow rather than asked for here, because during this pipeline's own development a branch dispatch published every image and moved every `:latest` alias while this file claimed that was impossible.

A publish produces one image per row of the `engine` matrix in `.gitea/workflows/publish-image.yaml`, because an image bakes that engine's assets and installs its Python extra. Today that matrix reads `[piper, kokoro, chatterbox, router]`, so a publish is four images — `elvenspeak-piper:YYYY.MM.DD.N`, `elvenspeak-kokoro:YYYY.MM.DD.N`, `elvenspeak-chatterbox:YYYY.MM.DD.N`, `elvenspeak-router:YYYY.MM.DD.N`, `:latest` only from master. Count them off the matrix rather than off this sentence: `tests/test_workflow.py` holds that list equal to `elvenspeak.engines.ENGINES`, so the matrix is the authority and the four names here are only its shadow — the day a fifth engine is added, this sentence is the thing that went stale, not the pipeline. That is not a hypothetical: it said three until #34, because chatterbox landed in #33 and nothing here is held equal to anything. The router rides the matrix even though it synthesizes nothing — its Python extra is deliberately empty, because its backends are other elvenspeak deployments. The engine is in the image *name*; the tag is a pure date sequence, N read from what the registry has already published, never from a run number.

Every image's tag then goes into home-infra's `service-versions.auto.tfvars.json`, one service key per image — `elvenspeak-piper`, `elvenspeak-kokoro`, `elvenspeak-chatterbox`, `elvenspeak-router` — and Atlantis deploys. CI holds no home-infra credential, so that last step is deliberately a human's.

Move all of those keys in one commit. The four images are not independently deployable, and the router is the one that cannot lead: since `piper-routing-7e2.17` it asks each backend which models its voices speak, and a backend image older than that answers nothing, so `remote._voice` raises `ConfigError` while the router is still constructing itself. It never reaches the point of serving a request — it exits. Rehearse the moment, because it is the shape a careful deploy takes: you have the diff open, the router's own code did not change this cycle, and you think *"I'll bump the engine keys and leave the router pinned — deploy only what changed."* That instinct is right nearly everywhere and wrong here, because the tags are not four independent versions; they are one fleet-wide handshake, and half of it does not answer. This is what a split rollout looks like:

    WRONG — the router left behind:

      "elvenspeak-piper":      "2026.09.14.1",   moved
      "elvenspeak-kokoro":     "2026.09.14.1",   moved
      "elvenspeak-chatterbox": "2026.09.14.1",   moved
      "elvenspeak-router":     "2026.09.02.5",   pinned — an old router fronting new
                                                 backends, running none of the routing
                                                 the tag it is pinned to predates

    RIGHT — one commit, one tag, every key:

      "elvenspeak-piper":      "2026.09.14.1",
      "elvenspeak-kokoro":     "2026.09.14.1",
      "elvenspeak-chatterbox": "2026.09.14.1",
      "elvenspeak-router":     "2026.09.14.1",

Observed, not theorised: at 2026.09.02.5 the router's first allocation exited 2 and restarted once, and it recovered only because all three keys had moved together.

<!-- BEGIN LIT INTEGRATION -->
## lit Agent-Native Workflow

This repository uses `lit` for agent-native issue tracking.

Start by running `lit quickstart` to load the workflow instructions. It prints how tickets are found, created, updated, and closed here, so running it first means the rest of your work follows the conventions this repo expects. It's a quick, read-only command — no need to check in before running it.

<!-- END LIT INTEGRATION -->
