"""The gates, checked for the two ways one can pass while testing nothing.

There are two, they stop different things, and both are checked here by the same
assertions because they are one kind of object:

    `.github/workflows/tests.yml`            the MERGE gate -- no red commit
                                             reaches master
    `.gitea/workflows/publish-image.yaml`    the PUBLISH gate -- no red commit
                                             becomes an image (7e2.14)

Neither subsumes the other. GitHub's runs on pull requests and cannot see a
`workflow_dispatch` on a branch, which is a publish that never opens a pull
request; gitea's runs on every ref it is pushed and knows nothing about merging.

A gate's own failure mode is quiet in the way a gate's always is: a gate that has
stopped gating looks exactly like a gate nobody has tripped. The check goes
green, the pull request merges or the image publishes, and the first symptom is a
broken master -- or a broken image in the cluster -- that CI reported as fine.

[FRAMING:representation] Two edits make it green without making it true, and
neither announces itself. Narrowing the command -- a path, a `-k`, a `-m` --
leaves a check that runs and passes over a fraction of the suite. Silencing the
result -- `continue-on-error`, a `|| true`, a retry that reruns until a run comes
back clean -- leaves a check that cannot go red at all. The rest of the ways to
break this file are loud: a renamed job, a deleted trigger, a syntax error all
end with a required check that never reports and a pull request that blocks
forever, which needs no test to notice.

Read off the workflow files for the same reason `tests/test_workflow.py` reads off
`.gitea/workflows/publish-image.yaml`: the file is what the forge executes, and a
run is a copy of the answer that goes stale in the direction that hides the bug.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent

MERGE_GATE = _ROOT / ".github" / "workflows" / "tests.yml"
PUBLISH_GATE = _ROOT / ".gitea" / "workflows" / "publish-image.yaml"

#: Both gates, as one parameter set. Every assertion below holds of a gate as
#: such, so a third one is an entry here and no new test
#: ([LAW:one-type-per-behavior]) -- and, more to the point, a gate added without
#: one cannot be quietly weaker than the gate it was copied from.
GATES = [
    pytest.param(MERGE_GATE, id="merge-gate"),
    pytest.param(PUBLISH_GATE, id="publish-gate"),
]

#: The step that runs the suite, captured as everything pytest itself is passed.
#: Matched against the file with its comment lines removed, because this
#: workflow's comments discuss retrying at length -- they are the record of why
#: nothing here retries -- and a check that cannot tell YAML from prose about
#: YAML would read that prose as the thing it forbids.
#: `tests/test_workflow.py` keeps its distance from the same mistake, which
#: `tests/test_dockerfile.py` was bitten by first.
#:
#: The capture starts after the word `pytest` so that the selectors below are
#: read as pytest's own. `uv run python -m pytest` is a legitimate way to spell
#: this step, and a check that searched the whole line for `-m` would call that
#: spelling a narrowed test run and refuse it.
_PYTEST = re.compile(r"^\s*run:\s*.*\bpytest\b(.*)$", re.MULTILINE)

#: Ways to make a failing suite stop failing the check, spelled as they appear in
#: a workflow. Not a guess at everything a future editor might reach for -- an
#: exhaustive list is not available -- but the three that have a name, and the
#: three anyone reaches for first when a required check goes red at random.
_SILENCERS = ("continue-on-error", "|| true", "|| :")

#: Selectors that shrink what `pytest` collects. A gate running a tenth of the
#: suite passes a tenth of the time it should fail.
#:
#: A named module or test is `.py` and `::` rather than a directory, because
#: `pytest tests/` collects the whole suite -- every test in this project lives
#: under that one directory -- and refusing that spelling would be refusing a
#: gate that does exactly what it should.
_NARROWERS = (" -k", " -m", " --ignore", " --deselect", ".py", "::")


def yaml_without_prose(workflow: Path) -> str:
    """`workflow`'s YAML with its commentary stripped."""
    return "\n".join(
        line
        for line in workflow.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )


def pytest_arguments(workflow: Path) -> list[str]:
    """What each pytest invocation in `workflow` passes to pytest."""
    return _PYTEST.findall(yaml_without_prose(workflow))


@pytest.mark.parametrize("workflow", GATES)
def test_the_gate_still_runs_the_suite(workflow):
    """Positive control: the reader still reads, and the gate still invokes pytest.

    Both failures land here and they are the same failure. A regex that quietly
    stopped matching would make the assertions below inspect an empty list --
    green, and meaning nothing. A workflow that stopped running the suite reads
    identically from here, which is the point: neither is allowed to be silent.
    """
    assert pytest_arguments(workflow), (
        f"no pytest invocation in {workflow.name} — either the workflow stopped "
        "running the suite, or this regex stopped finding it"
    )


@pytest.mark.parametrize("workflow", GATES)
def test_the_gate_runs_every_test(workflow):
    """[LAW:no-silent-failure] A gate over part of the suite reports on the whole.

    Nothing distinguishes "all 357 passed" from "the 12 I still collect passed"
    in a green check, so the narrowing has to be refused here rather than noticed
    later.
    """
    for arguments in pytest_arguments(workflow):
        found = [flag for flag in _NARROWERS if flag in arguments]
        assert not found, (
            f"{workflow.name} narrows what it collects with {found}: {arguments!r} — "
            "a check that runs part of the suite passes for the wrong reason"
        )


@pytest.mark.parametrize("workflow", GATES)
def test_nothing_silences_a_failing_suite(workflow):
    """[LAW:no-silent-failure] The gate is allowed to be red. That is its whole job.

    The pressure this resists is real and it is documented on ticket
    piper-tests-ona: roughly one full-suite run in five has aborted at
    interpreter teardown from ONNX Runtime, after every test already passed. The
    fix for that is the teardown. Reaching instead for a retry or a
    `continue-on-error` here would trade a visible intermittent failure for an
    invisible permanent one -- and it would take the real gate down with it,
    since a step that cannot fail cannot block a merge either.
    """
    yaml = yaml_without_prose(workflow)
    found = [silencer for silencer in _SILENCERS if silencer in yaml]
    assert not found, (
        f"{workflow.name} silences its own failures with {found} — fix what is "
        "failing; a check that cannot go red gates nothing"
    )


# ---------------------------------------------------- the publish gate is wired up


#: The whole `run:` value of the step that runs the suite, not just what follows
#: `pytest`. The two gates have to invoke the *same* suite, and the part that
#: decides which suite that is -- `--all-extras`, `--locked` -- is to the left of
#: the word the other pattern starts capturing at.
_SUITE_COMMAND = re.compile(r"^\s*run:\s*(.*\bpytest\b.*)$", re.MULTILINE)

#: Every `needs:` in the publish workflow. There is one, and the positive control
#: below fails if that stops being true rather than letting this quietly read the
#: wrong job's dependencies.
_NEEDS = re.compile(r"^\s*needs:\s*(.+)$", re.MULTILINE)


def suite_command(workflow: Path) -> str:
    found = {command.strip() for command in _SUITE_COMMAND.findall(yaml_without_prose(workflow))}
    assert len(found) == 1, (
        f"{workflow.name} runs the suite {len(found)} different ways ({found}) — "
        "a gate with two spellings of its own command has two answers to what passed"
    )
    return found.pop()


def test_both_gates_run_the_same_suite():
    """[LAW:one-source-of-truth] "The tests passed" has to name one fact.

    Two gates running two different commands is two different claims wearing one
    sentence, and the drift is silent in the direction that matters: the publish
    gate is the one no human reads before an image reaches the cluster, so it is
    the one that can quietly narrow and still look green.

    Compared rather than shared, because a workflow cannot import anything: the
    same line genuinely has to be typed in both files, and this is what stops the
    two copies diverging.
    """
    assert suite_command(MERGE_GATE) == suite_command(PUBLISH_GATE), (
        f"the gates run different suites:\n"
        f"  {MERGE_GATE.name}:  {suite_command(MERGE_GATE)}\n"
        f"  {PUBLISH_GATE.name}: {suite_command(PUBLISH_GATE)}"
    )


def test_the_publish_gate_is_actually_wired_to_the_publish():
    """The gate exists, and the thing it gates depends on it (7e2.14).

    This is the assertion the ticket is really about. A `tests` job that runs,
    goes red, and is not in `publish`'s `needs` is the worst of all the shapes
    here: it looks like a gate on the runs page, it costs the full suite every
    build, and the image publishes anyway. Nothing else in this repo would
    notice -- the run is red, the tag is there, and the two facts are on
    different screens.

    `reachability` is asserted alongside it because dropping it while adding
    `tests` is the plausible edit: `needs:` goes from a string to a list, and a
    list with one entry in it is a silent loss of the older gate.
    """
    listed = _NEEDS.findall(yaml_without_prose(PUBLISH_GATE))
    assert len(listed) == 1, (
        f"{PUBLISH_GATE.name} has {len(listed)} `needs:` lines ({listed}) — this "
        "check reads the publish job's dependencies and can no longer tell which "
        "of them it is looking at"
    )
    required = listed[0]
    for job in ("reachability", "tests"):
        assert job in required, (
            f"the publish job does not need {job!r} ({required!r}) — a gate the "
            "publish does not depend on gates nothing, however red it goes"
        )
