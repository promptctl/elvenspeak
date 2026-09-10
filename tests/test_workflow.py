"""What the publish workflow builds, proves and publishes, and in what order.

Claims about `.gitea/workflows/publish-image.yaml`, every one read off the file
because the file is what act_runner executes. The first is the engine set below.
The rest concern the two jobs that build the real Dockerfile: that `prove` does
it on every ref and can publish nothing, that `publish` cannot run past a proof
that failed, and that `publish` pushes nothing before it has run the image and
seen it serve. Those are guarantees made of position and of names, so they
cannot be held by anything except checks on position and on names.

The images CI builds, checked against the engines this package registers.

Each building job's matrix is another map of the engine set — after
`elvenspeak.engines.ENGINES`, which decides what `ELVENSPEAK_ENGINE` may name,
and `pyproject.toml`'s extras, which decide what can be installed.
`tests/test_packaging.py` already holds those two together; this file adds the
matrices, for the same reason and by the same means.

[FRAMING:representation] The failure it makes expressible is a quiet one. An
engine registered and packaged but missing from the matrix publishes no image and
reports nothing wrong: the workflow goes green, the registry gains the tags it
was asked for, and the only symptom is a service key nobody can fill months later
when someone tries to deploy the engine they thought shipped. Nothing in CI can
catch this — the workflow cannot know about an engine it was never told to build.

Read off the workflow file rather than off a run, for the reason
`tests/test_packaging.py` reads `pyproject.toml` rather than the installed
environment: the file is what act_runner executes, and anything else is a copy of
the answer that goes stale in the direction that hides the bug.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from workflows import job, needs, without_prose

from elvenspeak.engines import ENGINES

WORKFLOW = Path(__file__).parent.parent / ".gitea" / "workflows" / "publish-image.yaml"
SMOKE_ACTION = Path(__file__).parent.parent / ".gitea" / "actions" / "smoke-image" / "action.yml"
ENGINE_SOURCE = Path(__file__).parent.parent / "elvenspeak" / "chatterbox.py"
DEPLOY_INSTRUCTIONS = Path(__file__).parent.parent / "CLAUDE.md"

#: The jobs that build the real Dockerfile once per engine: `prove` on every
#: ref, `publish` on the refs that publish. Each declares its own matrix,
#: because a workflow cannot hand one job's matrix to another, so the engine
#: list is written once per job and every copy is held to the registry below.
BUILDERS = ("prove", "publish")

#: An image name, which is also its service key in home-infra — the two are one
#: string by construction, which is why one pattern finds both.
_SERVICE_KEY = re.compile(r"elvenspeak-([a-z]+)")

#: A build matrix's one axis. Matched against one job's YAML with its comments
#: removed — this workflow's comments discuss the engine list at length, and a
#: check that cannot tell YAML from prose about YAML is the substring-against-
#: prose mistake `tests/test_dockerfile.py` was already bitten by.
_MATRIX = re.compile(r"^\s*engine:\s*\[([^\]]*)\]\s*$", re.MULTILINE)


def matrix_engines(builder: str) -> list[str]:
    """Every engine the job called `builder` builds an image for."""
    return [
        engine.strip()
        for listing in _MATRIX.findall(job(WORKFLOW, builder))
        for engine in listing.split(",")
        if engine.strip()
    ]


@pytest.mark.parametrize("builder", BUILDERS)
def test_the_workflow_still_declares_a_build_matrix(builder):
    """Positive control: the reader still reads.

    A regex over YAML that quietly stopped matching would make the equivalence
    below compare two empty sets — green, and meaning nothing. That is how this
    suite's other static checks have failed before, so the vacuous case is a
    failure here rather than silence.
    """
    assert matrix_engines(builder), (
        f"parsed no matrix engines in {builder} — the regex is wrong, not the file"
    )


@pytest.mark.parametrize("builder", BUILDERS)
def test_ci_builds_an_image_for_every_registered_engine(builder):
    """[LAW:one-source-of-truth] The registry decides; every matrix follows.

    Stated as an equivalence so it fails from both sides. An engine registered
    without a `publish` entry is one no deployment can ever run, because no
    image of it exists; without a `prove` entry, its Dockerfile build goes
    unchecked until the run that publishes it. A matrix entry naming no engine
    is a build that installs an extra `uv` has never heard of and fails ten
    minutes in — loudly, but on the runner rather than here.
    """
    assert set(matrix_engines(builder)) == set(ENGINES)


@pytest.mark.parametrize("builder", BUILDERS)
def test_no_engine_is_built_twice(builder):
    """A duplicate would build two images under one name, second overwriting first.

    Harmless-looking, and the tell that the list was edited by appending rather
    than by reading — which is exactly how the equivalence above stops being the
    check it looks like.
    """
    engines = matrix_engines(builder)
    assert len(engines) == len(set(engines)), engines


def test_the_deploy_instructions_name_every_engine_and_no_others():
    """[LAW:one-source-of-truth] The registry decides; the instructions follow.

    `CLAUDE.md` tells an agent how many images a publish produces and which
    service keys move together, by name. It is prose, so it cannot compute the
    list, and it has now been wrong about it twice: #30 was merged to stop it
    telling agents to deploy two of three images, and it still said three after
    chatterbox landed in #33 — while containing a sentence predicting exactly
    that ("the day a fourth engine is added, this sentence is the thing that went
    stale"). Predicting the drift is not preventing it. This is what prevents it.

    The direction that matters is the silent one. An engine missing from that
    list is an image nobody deploys and a key nobody fills, and the deploy looks
    complete — which is the same failure the matrix check above exists for, one
    step further down the pipeline, where nothing else is watching.

    Stated as an equivalence so it fails from both sides: a name here that no
    longer names an engine sends someone looking for an image CI never built.
    Sets rather than counts, because how many times a name is written is a fact
    about the prose and none of this test's business.
    """
    named = set(_SERVICE_KEY.findall(DEPLOY_INSTRUCTIONS.read_text(encoding="utf-8")))
    assert named == set(ENGINES)


#: A smoke of a built image, which every building job runs through the one
#: composite action. The reachability job runs `smoke.py --help` directly to
#: prove the runner's interpreter parses it, and that run smokes no image — so
#: the action is what tells the proof apart from the proof that the prover works.
_SMOKE = re.compile(r"^\s*uses:\s*\./\.gitea/actions/smoke-image\s*$", re.MULTILINE)

#: Every push of a built image to the registry.
_PUSH = re.compile(r"^\s*docker push ", re.MULTILINE)


def test_no_image_is_pushed_before_it_has_been_proved_to_run():
    """[LAW:parse-dont-validate] The step order is the guarantee, so it is the thing to hold.

    `smoke.py` runs the image and proves it answers /health. That it runs at all
    is worth much less than *where*: above the push, a red smoke ends the leg
    with the registry untouched; below it, the same script reports the same fact
    about the same image after the bytes have landed and `:latest` has moved,
    leaving a deploy one `nomad job run` away from an artifact the run already
    knew was broken.

    The move that breaks it is a reasonable-sounding one, which is why prose will
    not hold it. Smoking the pushed reference reads like the more honest test —
    it is what a deploy actually pulls — and rejoining the build and push steps
    reads like tidying up two steps that were one until this check needed to
    stand between them. Either edit leaves a green suite, a green publish, and no
    proof left in the pipeline at all.

    Held of `publish`'s own smoke, not `prove`'s. `publish` pushes its own
    build, and a smoke of a different build of the same commit, in another job,
    proves nothing about the bytes this job is about to push.

    Positions rather than presence, and both patterns controlled for: a regex
    that quietly stopped matching would compare nothing against nothing and pass,
    which is how this suite's other static checks have failed before.
    """
    publish = job(WORKFLOW, "publish")
    smoked = [m.start() for m in _SMOKE.finditer(publish)]
    pushed = [m.start() for m in _PUSH.finditer(publish)]

    assert len(smoked) == 1, f"expected one smoke of a built image, found {len(smoked)}"
    assert pushed, "matched no `docker push` — the regex is wrong, not the file"
    assert smoked[0] < min(pushed), (
        "the publish job pushes before it smokes, so a broken image reaches the "
        "registry and `:latest` before anything starts the process"
    )


def test_the_proof_on_every_ref_can_publish_nothing():
    """A branch push spends no dated tag, and `prove` must not be what changes that.

    `prove` builds and runs the real Dockerfile on every ref, which is safe only
    because nothing in it reaches the registry: its image name carries no
    registry host, it pushes nothing, and it never reads `$REGISTRY`. The edit
    that breaks this sounds like an economy: "the image is already proven, push
    it from here and spare `publish` its rebuild". After that, every branch push
    takes a dated tag, and a branch deleted later leaves a published image whose
    commit no ref names.

    Positive control first: the slice is `prove` and it does smoke, so an empty
    slice cannot pass the two absences below.
    """
    prove = job(WORKFLOW, "prove")
    assert _SMOKE.search(prove), "prove runs no smoke — the slice or the regex is wrong"
    assert not _PUSH.search(prove), "prove pushes an image, so every branch push would publish"
    assert "REGISTRY" not in prove, "prove names the registry, so its image could be pushed there"


#: A job-level condition: four spaces in, directly under the job's key. A step's
#: `if:` sits deeper, and decides whether one step runs rather than the job.
_JOB_CONDITION = re.compile(r"^    if:", re.MULTILINE)


def test_the_proof_runs_on_every_ref():
    """[LAW:dataflow-not-control-flow] Nothing decides whether an image gets proved.

    A skipped check is indistinguishable from a passing one. A `prove` gated on
    the ref or on what a diff touched goes green by not running, and `publish`,
    which needs it, then waits on nothing. The twenty minutes a chatterbox leg
    costs is exactly the pressure that produces that edit, so it is held here
    rather than argued in a comment.

    `publish` carries a job-level `if:` by design, which makes it the positive
    control: if the pattern stopped finding a condition, it would stop finding
    one on `prove` too.
    """
    assert _JOB_CONDITION.search(job(WORKFLOW, "publish")), (
        "found no job-level `if:` on publish — the pattern is wrong, not the file"
    )
    assert not _JOB_CONDITION.search(job(WORKFLOW, "prove")), (
        "prove has a job-level `if:`, so some refs publish past a proof that never ran"
    )


def test_nothing_publishes_past_a_proof_that_failed():
    """`publish` needs `prove`, so a ref whose image did not serve publishes nothing.

    A proof is a gate only if the thing it gates depends on it. A `prove` that
    `publish` does not need goes red on the runs page while the image publishes
    anyway, with the two facts on different screens. The dependency also orders
    the two jobs: without it a master push starts both matrices at once, and two
    chatterbox smokes share one 7.9 GB runner.
    """
    assert "prove" in needs(WORKFLOW, "publish")


#: One row of the smoke's boot table: an engine, the memory ceiling measured for
#: it, then whatever else it needs before the image can answer at all.
_BOOT_TABLE = re.compile(r"^\s*([a-z][a-z0-9_]*)\)\s*memory=\d+m;\s*boot=\(", re.MULTILINE)

#: The synthesis width every smoked image is confined to.
_WIDTH = re.compile(r"ELVENSPEAK_CONCURRENT_SYNTHESES=(\d+)")


def test_the_smoke_boot_table_names_every_engine_and_nothing_else():
    """A row keyed to a name that is not an engine is a silent no-op; a missing one is red.

    Two of four legs went red on the first real run of the smoke step because the
    image was handed an empty environment and a setting with no default was
    missing -- a red that says nothing about the commit. The table that fixed it
    is keyed by engine name, and a name that stops matching stops applying,
    quietly: the leg falls to the refusing arm, with the table sitting right there
    looking like it covers the case.

    Both directions, since piper-build-b4h.scw. Every engine now needs a row,
    because every image is smoked under a memory ceiling and a ceiling is a
    per-engine measurement. An engine with no row is refused by the table's `*)`
    arm at run time, and this says so before a runner is spent finding out.
    """
    named = set(_BOOT_TABLE.findall(without_prose(SMOKE_ACTION)))
    assert named, "matched no boot-table entries — the regex is wrong, not the file"
    assert named == set(ENGINES), (
        f"boot table names non-engines {named - set(ENGINES)} "
        f"and has no ceiling for {set(ENGINES) - named}"
    )


def test_the_smoke_confines_every_image_to_the_width_conformance_exercises():
    """The width a ceiling is proven at is the width `speaks.py` actually drives.

    A ceiling means "this image fits in this much at this many syntheses at once".
    Named wider than [`speaks.CALLERS_AT_ONCE`], the width would be a claim no
    caller ever tested; named narrower, the fourth caller queues and the measured
    peak understates what the image costs at the width conformance asks for.
    """
    widths = _WIDTH.findall(without_prose(SMOKE_ACTION))
    assert widths == [str(speaks.CALLERS_AT_ONCE)], (
        f"smoke-image names widths {widths}; conformance drives {speaks.CALLERS_AT_ONCE}"
    )


#: The CPU row of `elvenspeak.chatterbox`'s measurement table, which owns both
#: figures every other file quotes: the RTF range, then resident and peak.
_MEASURED = re.compile(
    r"^\s*CPU \([^)]*\)\s+([\d.]+) - ([\d.]+)\s+([\d.]+) GiB, ([\d.]+) GiB peak\s*$",
    re.MULTILINE,
)

#: Every restatement of the peak, wherever it is quoted.
_PEAK = re.compile(r"([\d.]+) GiB peak")

#: Every restatement of the RTF range, in the two shapes this repository writes it.
_RTF = re.compile(r"(\d+)\s*(?:-|to)\s*(\d+)\s*(?:x real time|times real time)")

#: A line break inside a comment or docstring, with whatever marker and
#: indentation continue it. Collapsed before matching, because a quotation is
#: prose and prose wraps: the `Dockerfile`'s figure sat across "8-33x real" /
#: "# time" and no pattern anchored to contiguous text could ever have seen it —
#: which is how it survived a review round that was looking straight at it.
_WRAP = re.compile(r"\n[ \t]*(?:#:?|//|--)?[ \t]*")


def flowed(text: str) -> str:
    """`text` with wrapped lines rejoined, so a quotation reads as one string."""
    return _WRAP.sub(" ", text)


def quoting() -> list[Path]:
    """Every tracked file, because a restatement may live in any of them.

    [LAW:single-enforcer] The first version of this listed globs —
    `elvenspeak/*.py`, `tests/*.py`, `.gitea/workflows/*.yaml` — and review found
    a stale figure in the `Dockerfile` the same day, invisible to the check
    written to prevent exactly that. A list of places to look is the same
    hand-maintained map that lets a fact drift in the first place; it just moves
    the drift one level up, to the list.

    So there is no list. `git ls-files` is the repository's own answer to "what
    is in this repository", and a file added tomorrow is scanned without anyone
    remembering to say so. Read as text, never imported, so a stale `.pyc` cannot
    answer for a file (see the test below).
    """
    import subprocess

    listed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        check=True,
        text=True,
    )
    return sorted(
        Path(__file__).parent.parent / name
        for name in listed.stdout.split("\0")
        if name
    )


def test_every_quotation_of_chatterbox_cost_matches_what_it_measured():
    """[LAW:single-enforcer] One measurement, however many places quote it.

    Two figures travel: the CPU peak an operator checks free RAM against before
    `git push gitea master`, and the RTF range that justifies this engine having
    no default device. Both are quoted at people who act on them, in files that
    cannot compute — a YAML comment, a `ConfigError` message, half a dozen
    docstrings — so every quotation is a hand-copy that can drift.

    They already had, three ways. The peak was written 6.8, 6.83 and 6.68 in
    three files; the RTF low bound was 10 in `chatterbox.py` and 8 in every file
    quoting it, where 10 is neither measurement's low end but a splice of the
    12-thread low with the 4-thread high. Nobody would catch that reading any one
    file, and the dangerous direction is silent: a comment understating the peak
    reads exactly like a safe one.

    So the table in `elvenspeak/chatterbox.py` owns both figures and this holds
    every other copy equal to it, anywhere in the repository — see [`quoting`]
    for why it scans everything tracked rather than a list of likely files.

    Read as source text, never through `chatterbox.__doc__`: an edit preserving
    a row's byte length and landing in the same second as the last import left
    `__doc__` serving the previous numbers off a `.pyc` Python thought current.
    That cost a false green while this test was being written.
    """
    measured = _MEASURED.search(ENGINE_SOURCE.read_text(encoding="utf-8"))
    assert measured, "no CPU row found in chatterbox's measurement table"

    rtf_low, rtf_high, _resident, peak = measured.groups()
    # The table carries the raw measurement; prose rounds it to whole numbers.
    expected_rtf = (str(round(float(rtf_low))), str(round(float(rtf_high))))

    for path in quoting():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # a binary asset quotes nothing
        where = path.relative_to(Path(__file__).parent.parent)
        text = flowed(text)
        for quoted in _PEAK.findall(text):
            assert quoted == peak, f"{where} quotes a {quoted} GiB peak; measured {peak}"
        for quoted in _RTF.findall(text):
            assert quoted == expected_rtf, (
                f"{where} quotes {quoted[0]}-{quoted[1]}x real time; "
                f"measured {rtf_low}-{rtf_high}"
            )
