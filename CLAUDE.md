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

To prove a Dockerfile change before it lands, dispatch the workflow on the branch. Know what that costs: it is a real publish, and it consumes a dated tag that is then spent forever. It does not move `:latest` — the tag step only adds that alias when the ref is `refs/heads/master`, so an unreviewed build cannot become what a break-glass `nomad job run` with no `-var` picks up. That guard is enforced in the workflow rather than asked for here, because during this pipeline's own development a branch dispatch published both images and moved both `:latest` aliases while this file claimed that was impossible.

A publish produces two images, one per engine, because an image bakes that engine's assets and installs its Python extra: `elvenspeak-piper:YYYY.MM.DD.N` and `elvenspeak-kokoro:YYYY.MM.DD.N`, each also tagged `:latest`. The engine is in the image *name*; the tag is a pure date sequence, N read from what the registry has already published, never from a run number. Both tags then go into home-infra's `service-versions.auto.tfvars.json` under two service keys, one per image, and Atlantis deploys. CI holds no home-infra credential, so that last step is deliberately a human's.

<!-- BEGIN LIT INTEGRATION -->
## lit Agent-Native Workflow

This repository uses `lit` for agent-native issue tracking.

Start by running `lit quickstart` to load the workflow instructions. It prints how tickets are found, created, updated, and closed here, so running it first means the rest of your work follows the conventions this repo expects. It's a quick, read-only command — no need to check in before running it.

<!-- END LIT INTEGRATION -->
