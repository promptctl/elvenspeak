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

The workflow is `.gitea/workflows/publish-image.yaml`, and it runs on every ref you push. A branch push runs only the `reachability` job, which proves the *builder* — that it fetched the commit itself, that its CPU can run the wheels the image installs, that jq and a BuildKit-capable daemon are present, and that the registry both answers and accepts a write. Only `refs/heads/master` and a manual `workflow_dispatch` publish.

Read a green branch run as exactly that and no more. `reachability` never builds the real Dockerfile: it reads line 1 for the `# syntax=` directive and checks the daemon covers the `--mount` kinds the Dockerfile uses, then builds a synthetic busybox image. A broken `RUN`, a missing apt package or a failed asset download is not caught there — the first real build of the Dockerfile is the one that publishes.

To prove a Dockerfile change before it lands, dispatch the workflow on the branch. Know what that costs: it is a real publish, and it consumes a dated tag that is then spent forever. It does not move `:latest` — the tag step only adds that alias when the ref is `refs/heads/master`, so an unreviewed build cannot become what a break-glass `nomad job run` with no `-var` picks up. That guard is enforced in the workflow rather than asked for here, because during this pipeline's own development a branch dispatch published every image and moved every `:latest` alias while this file claimed that was impossible.

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
