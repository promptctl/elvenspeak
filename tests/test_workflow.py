"""The images CI publishes, checked against the engines this package registers.

`.gitea/workflows/publish-image.yaml` builds one image per engine, and its build
matrix is a third map of the engine set — after `elvenspeak.engines.ENGINES`,
which decides what `ELVENSPEAK_ENGINE` may name, and `pyproject.toml`'s extras,
which decide what can be installed. `tests/test_packaging.py` already holds those
two together; this file adds the third, for the same reason and by the same
means.

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

from elvenspeak.engines import ENGINES

WORKFLOW = Path(__file__).parent.parent / ".gitea" / "workflows" / "publish-image.yaml"
ENGINE_SOURCE = Path(__file__).parent.parent / "elvenspeak" / "chatterbox.py"

#: The build matrix's one axis. Matched against the file with its comments
#: removed — this workflow's comments discuss the engine list at length, and a
#: check that cannot tell YAML from prose about YAML is the substring-against-
#: prose mistake `tests/test_dockerfile.py` was already bitten by.
_MATRIX = re.compile(r"^\s*engine:\s*\[([^\]]*)\]\s*$", re.MULTILINE)


def matrix_engines() -> list[str]:
    """Every engine the publish job is told to build an image for."""
    yaml = "\n".join(
        line
        for line in WORKFLOW.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )
    return [
        engine.strip()
        for listing in _MATRIX.findall(yaml)
        for engine in listing.split(",")
        if engine.strip()
    ]


def test_the_workflow_still_declares_a_build_matrix():
    """Positive control: the reader still reads.

    A regex over YAML that quietly stopped matching would make the equivalence
    below compare two empty sets — green, and meaning nothing. That is how this
    suite's other static checks have failed before, so the vacuous case is a
    failure here rather than silence.
    """
    assert matrix_engines(), "parsed no matrix engines — the regex is wrong, not the file"


def test_ci_publishes_an_image_for_every_registered_engine():
    """[LAW:one-source-of-truth] The registry decides; the matrix follows.

    Stated as an equivalence so it fails from both sides. An engine registered
    without a matrix entry is one no deployment can ever run, because no image of
    it exists. A matrix entry naming no engine is a build that installs an extra
    `uv` has never heard of and fails ten minutes in — loudly, but on the runner
    rather than here.
    """
    assert set(matrix_engines()) == set(ENGINES)


def test_no_engine_is_built_twice():
    """A duplicate would publish two images to one name, second overwriting first.

    Harmless-looking, and the tell that the list was edited by appending rather
    than by reading — which is exactly how the equivalence above stops being the
    check it looks like.
    """
    engines = matrix_engines()
    assert len(engines) == len(set(engines)), engines


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
