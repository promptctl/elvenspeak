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
