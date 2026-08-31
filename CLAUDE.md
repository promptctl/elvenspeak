## Never build an image from a working tree

This service is being deployed to the homelab, so this rule applies here from the start.

Build from a commit, in CI, on a machine that fetched that commit itself. Anything else is banned and gets deleted on sight — no discussion, no exceptions.

Banned: copying, tarring, rsyncing or scp-ing this source tree onto a build host; building an image from anything the builder did not pull from git itself; any build that needs a laptop, a checkout, or an SSH session to run; build scratch directories left on the VMs.

Do not copy `scripts/build-image.sh` from the openconv repo. It is the pattern this rule exists to kill: it tars the working tree, untracked files included, and ships it over SSH to a build node, producing images whose source nobody can identify.

Why: an image built from a working tree has no known source. It contains whatever was on someone's disk, records no commit, and cannot be reproduced, bisected, or audited.

Instead: build in CI from a commit, tag `YYYY.MM.DD.N`, push to the homelab registry, then file that tag into the homelab's `service-versions.auto.tfvars.json` so Atlantis deploys it.

### Nothing builds until you push to gitea

Merging a pull request on GitHub builds nothing and publishes nothing. No mirror, no poller, no schedule, no webhook. GitHub (`origin`) is the development and review surface and knows nothing about images. The only thing in this system that causes an image to exist is a human or an agent running, by hand:

    git push gitea master

Rehearse the moment, because it is coming: the PR is merged, the checks are green, and you think *"the image should land in a few minutes — I'll go check the registry."* Nothing is coming. No run ever started. You can wait all afternoon and find an empty registry. After every merge to master, push to gitea, or there is no build. The remote is already in this clone — `gitea` → `ssh://git@gitea.sanctuary.gdn:2222/brandon-fryslie/elvenspeak.git` — and it is a build remote only; nothing ever flows back from it to GitHub. Because the trigger is a deliberate push, gitea is allowed to sit behind GitHub's master: that gap is not a mess to tidy up, it means nobody has asked for a build of those commits yet.

The workflow is `.gitea/workflows/publish-image.yaml`, and it runs on every ref you push. A branch push runs only the `reachability` job, which proves the builder without building — so a Dockerfile problem surfaces on a branch instead of at the far end of a cold build. Only `refs/heads/master` and a manual `workflow_dispatch` publish.

A publish produces two images, one per engine, because an image bakes that engine's assets and installs its Python extra: `elvenspeak-piper:YYYY.MM.DD.N` and `elvenspeak-kokoro:YYYY.MM.DD.N`, each also tagged `:latest`. The engine is in the image *name*; the tag is a pure date sequence, N read from what the registry has already published, never from a run number. Both tags then go into home-infra's `service-versions.auto.tfvars.json` under two service keys, one per image, and Atlantis deploys. CI holds no home-infra credential, so that last step is deliberately a human's.

<!-- BEGIN LIT INTEGRATION -->
## lit Agent-Native Workflow

This repository uses `lit` for agent-native issue tracking.

Start by running `lit quickstart` to load the workflow instructions. It prints how tickets are found, created, updated, and closed here, so running it first means the rest of your work follows the conventions this repo expects. It's a quick, read-only command — no need to check in before running it.

<!-- END LIT INTEGRATION -->
