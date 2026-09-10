"""A workflow file, read the one way this suite reads one.

`tests/test_workflow.py` and `tests/test_merge_gate.py` both hold claims about
`.gitea/workflows/publish-image.yaml`, and both need the same two readings of
it: the YAML without the commentary that discusses it, and one job's part of
that YAML. The first was written out in each file. The second became necessary
when `prove` joined `publish` as a job with its own `needs:` and its own engine
matrix, since a reader of the whole file can no longer tell whose it found.
[LAW:one-source-of-truth]

Read with patterns rather than a YAML parser, because the suite has never had
one to lean on: `pyyaml` reaches the lockfile only as somebody else's
dependency, and a test that imports it would pass or fail on whether an extra
happened to pull it in.
"""

from __future__ import annotations

import re
from pathlib import Path

#: Where the jobs begin. Everything above it is `name:`, `on:` and `env:`, whose
#: own two-space keys (`push:`, `REGISTRY:`) would otherwise read as jobs.
_JOBS = re.compile(r"^jobs:\s*$", re.MULTILINE)

#: A job's key: two spaces in, under `jobs:`. Nothing else below `jobs:` sits at
#: that depth, because a job's own keys and every block scalar are indented
#: further than the key that owns them.
_JOB = re.compile(r"^  ([A-Za-z_][\w-]*):\s*$", re.MULTILINE)

#: A job's dependencies, written inline.
_NEEDS = re.compile(r"^\s*needs:\s*(.+)$", re.MULTILINE)


def without_prose(workflow: Path) -> str:
    """`workflow`'s YAML with its commentary stripped.

    These workflows discuss their own YAML at length, so a check that cannot
    tell YAML from prose about YAML matches the discussion and reports on a
    sentence — the mistake `tests/test_dockerfile.py` was bitten by first.
    """
    return "\n".join(
        line
        for line in workflow.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )


def job(workflow: Path, name: str) -> str:
    """The YAML of the job called `name`, from its key to the next job's.

    A job that is not there fails naming the jobs that are, rather than
    returning empty text that every pattern downstream would find nothing in
    and pass.
    """
    yaml = without_prose(workflow)
    start = _JOBS.search(yaml)
    assert start, f"{workflow.name} declares no `jobs:`"
    jobs = yaml[start.end() :]
    keys = list(_JOB.finditer(jobs))
    for key, following in zip(keys, [*keys[1:], None]):
        if key.group(1) == name:
            return jobs[key.start() : following.start() if following else len(jobs)]
    raise AssertionError(
        f"{workflow.name} has no job {name!r}; its jobs are {[key.group(1) for key in keys]}"
    )


def needs(workflow: Path, name: str) -> frozenset[str]:
    """The jobs the job called `name` depends on, as names.

    [LAW:parse-dont-validate] The list is parsed into its members rather than
    left as the text `"[reachability, tests]"` for callers to search. Asking
    `"tests" in text` is a weaker theorem than it looks: it is also true of
    `[reachability, integration-tests]`, so an assertion that a job depends on
    `tests` could pass on a workflow where that dependency had been dropped.

    A `needs:` written as a block sequence puts nothing on the line and matches
    nothing, which fails the count below by name instead of silently reading an
    empty dependency list as a satisfied one.
    """
    listed = _NEEDS.findall(job(workflow, name))
    assert len(listed) == 1, (
        f"the {name} job in {workflow.name} has {len(listed)} inline `needs:` lines "
        f"({listed}), so its dependencies cannot be read"
    )
    return frozenset(
        member.strip() for member in listed[0].strip().strip("[]").split(",") if member.strip()
    )
