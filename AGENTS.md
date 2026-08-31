# AGENTS

## Never build an image from a working tree

Build from a commit, in CI, on a machine that fetched that commit itself. Anything else is banned and gets deleted on sight — no discussion, no exceptions.

CLAUDE.md in this repo states the rule in full — what exactly is banned, why, and what to do instead. It is the only copy. This section deliberately does not restate it: the two files disagreed about the rule within a single commit of each other, so read CLAUDE.md rather than growing a second version here.

<!-- BEGIN LIT INTEGRATION -->
## lit Agent-Native Workflow

This repository uses `lit` for agent-native issue tracking.

Start by running `lit quickstart` to load the workflow instructions. It prints how tickets are found, created, updated, and closed here, so running it first means the rest of your work follows the conventions this repo expects. It's a quick, read-only command — no need to check in before running it.

<!-- END LIT INTEGRATION -->
