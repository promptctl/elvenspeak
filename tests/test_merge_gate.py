"""The merge gate, checked for the two ways it can pass while testing nothing.

`.github/workflows/tests.yml` is what stops a commit that breaks a test from
reaching master. Its own failure mode is quiet in the way a gate's always is: a
gate that has stopped gating looks exactly like a gate nobody has tripped. The
check goes green, the pull request merges, and the first symptom is a broken
master that CI reported as fine.

[FRAMING:representation] Two edits make it green without making it true, and
neither announces itself. Narrowing the command -- a path, a `-k`, a `-m` --
leaves a check that runs and passes over a fraction of the suite. Silencing the
result -- `continue-on-error`, a `|| true`, a retry that reruns until a run comes
back clean -- leaves a check that cannot go red at all. The rest of the ways to
break this file are loud: a renamed job, a deleted trigger, a syntax error all
end with a required check that never reports and a pull request that blocks
forever, which needs no test to notice.

Read off the workflow file for the same reason `tests/test_workflow.py` reads off
`.gitea/workflows/publish-image.yaml`: the file is what GitHub executes, and a
run is a copy of the answer that goes stale in the direction that hides the bug.
"""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOW = Path(__file__).parent.parent / ".github" / "workflows" / "tests.yml"

#: The step that runs the suite. Matched against the file with its comment lines
#: removed, because this workflow's comments discuss retrying at length -- they
#: are the record of why nothing here retries -- and a check that cannot tell
#: YAML from prose about YAML would read that prose as the thing it forbids.
#: `tests/test_workflow.py` keeps its distance from the same mistake, which
#: `tests/test_dockerfile.py` was bitten by first.
_PYTEST = re.compile(r"^\s*run:\s*(.*\bpytest\b.*)$", re.MULTILINE)

#: Ways to make a failing suite stop failing the check, spelled as they appear in
#: a workflow. Not a guess at everything a future editor might reach for -- an
#: exhaustive list is not available -- but the three that have a name, and the
#: three anyone reaches for first when a required check goes red at random.
_SILENCERS = ("continue-on-error", "|| true", "|| :")

#: Selectors that shrink what `pytest` collects. A gate running a tenth of the
#: suite passes a tenth of the time it should fail.
_NARROWERS = (" -k", " -m", " --ignore", " tests/", " tests ")


def yaml_without_prose() -> str:
    """The workflow's YAML with its commentary stripped."""
    return "\n".join(
        line
        for line in WORKFLOW.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )


def pytest_commands() -> list[str]:
    """Every command in the workflow that invokes pytest."""
    return _PYTEST.findall(yaml_without_prose())


def test_the_gate_still_runs_the_suite():
    """Positive control: the reader still reads, and the gate still invokes pytest.

    Both failures land here and they are the same failure. A regex that quietly
    stopped matching would make the assertions below inspect an empty list --
    green, and meaning nothing. A workflow that stopped running the suite reads
    identically from here, which is the point: neither is allowed to be silent.
    """
    assert pytest_commands(), (
        "no pytest invocation in the merge gate — either the workflow stopped "
        "running the suite, or this regex stopped finding it"
    )


def test_the_gate_runs_every_test():
    """[LAW:no-silent-failure] A gate over part of the suite reports on the whole.

    Nothing distinguishes "all 357 passed" from "the 12 I still collect passed"
    in a green check, so the narrowing has to be refused here rather than noticed
    later.
    """
    for command in pytest_commands():
        found = [flag for flag in _NARROWERS if flag in command]
        assert not found, (
            f"the merge gate narrows what it collects with {found}: {command!r} — "
            "a check that runs part of the suite passes for the wrong reason"
        )


def test_nothing_silences_a_failing_suite():
    """[LAW:no-silent-failure] The gate is allowed to be red. That is its whole job.

    The pressure this resists is real and it is documented on ticket
    piper-tests-ona: roughly one full-suite run in five has aborted at
    interpreter teardown from ONNX Runtime, after every test already passed. The
    fix for that is the teardown. Reaching instead for a retry or a
    `continue-on-error` here would trade a visible intermittent failure for an
    invisible permanent one -- and it would take the real gate down with it,
    since a step that cannot fail cannot block a merge either.
    """
    yaml = yaml_without_prose()
    found = [silencer for silencer in _SILENCERS if silencer in yaml]
    assert not found, (
        f"the merge gate silences its own failures with {found} — fix what is "
        "failing; a check that cannot go red gates nothing"
    )
