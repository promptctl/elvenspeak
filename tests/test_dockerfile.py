"""The Dockerfile's references to this repository, checked against the repository.

This file exists because of a specific failure. The package was renamed
`piper_server` -> `elvenspeak` and the Dockerfile's `COPY` was missed, so the
image could not build. The verification missed it the same way: the check was
`grep -rl --include=*.py --include=*.toml --include=*.md . Dockerfile`, where the
include-filters silently exclude the explicit file argument. It matched nothing
and read exactly like success.

[LAW:verifiable-goals] A check that shares an assumption with the thing it checks
can only confirm the assumption. So this reads the Dockerfile itself and resolves
each path against the filesystem — no pattern to get wrong, and it runs on every
commit rather than on whoever remembers.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
DOCKERFILE = REPO / "Dockerfile"

#: `COPY [--flags] <src>... <dest>`. Sources beginning with `--from=` name a
#: build stage or external image rather than a path in this repo, so a COPY
#: carrying one has no local source to resolve.
_COPY = re.compile(r"^\s*COPY\s+(.*)$", re.IGNORECASE)


def copy_sources() -> list[str]:
    """Every path this Dockerfile copies out of the build context."""
    sources: list[str] = []
    for line in DOCKERFILE.read_text().splitlines():
        match = _COPY.match(line)
        if not match:
            continue
        parts = match.group(1).split()
        if any(part.startswith("--from=") for part in parts):
            continue
        sources.extend(part for part in parts[:-1] if not part.startswith("--"))
    return sources


def test_dockerfile_exists():
    assert DOCKERFILE.exists()


def test_every_copied_path_is_in_the_repository():
    """The check that would have caught the rename, run automatically."""
    assert copy_sources(), "parsed no COPY sources — the regex is wrong, not the file"
    missing = [src for src in copy_sources() if not (REPO / src.rstrip("/")).exists()]
    assert not missing, f"Dockerfile copies paths that do not exist: {missing}"


def test_the_package_directory_is_copied():
    """Named explicitly, so deleting the COPY is a failure and not just silence."""
    assert any(src.rstrip("/") == "elvenspeak" for src in copy_sources())


@pytest.mark.parametrize("image", ["ghcr.io/astral-sh/uv", "python"])
def test_base_images_are_pinned(image):
    """[LAW:no-ambient-temporal-coupling] `:latest` makes one commit two artifacts."""
    text = DOCKERFILE.read_text()
    assert f"{image}:latest" not in text, f"{image} is unpinned"


def test_the_container_does_not_run_as_root():
    assert re.search(r"^\s*USER\s+\S+", DOCKERFILE.read_text(), re.MULTILINE)


def instructions() -> str:
    """The Dockerfile's instructions, one per line, with its comments removed.

    Asserting against the raw text matches the prose describing a pattern as
    readily as the pattern itself — the first version of the test below failed on
    the comment explaining the very interpolation it forbids. A check that cannot
    tell code from a description of code is the same substring-against-prose
    mistake in a new costume.

    Backslash continuations are joined here rather than by each caller, because
    a line is not what any check below actually reasons about: `RUN a \\` and its
    `&& b` are one instruction, and a per-line scan reads only as far as the
    first of them. Left to callers, every future check written here would be
    free to make that mistake independently — and one already did, passing while
    a continuation line went unexamined.
    """
    return "\n".join(
        line
        for line in DOCKERFILE.read_text().replace("\\\n", "").splitlines()
        if not line.lstrip().startswith("#")
    )


def test_no_build_arg_is_spliced_into_source_text():
    """Build args reach Python through the environment; interpolation was injectable.

    A single quote in a `--build-arg` once escaped into executable code, because
    the value was interpolated as `'${PIPER_VOICES}'.split(',')` — a
    caller-supplied string inside a Python literal inside a shell command.

    Asserts the property rather than one spelling of it. This previously also
    required `os.environ['PIPER_VOICES']` to appear, which pinned one particular
    way of reading the value: routing the same read through `Settings.from_env`
    satisfied the contract and failed the test anyway.

    Deliberately without a "parsed nothing" guard, unlike every other check here:
    the only `python -c` left in this file is the HEALTHCHECK's stdlib one-liner,
    and a Dockerfile carrying no Python source at all is the state where this
    property is vacuously true because it has been made unreachable, not the
    state where the regex has quietly stopped matching.
    """
    for block in re.findall(r'python -c "([\s\S]*?)"\s*$', instructions(), re.MULTILINE):
        assert "${" not in block, f"build arg interpolated into source: {block[:80]!r}"


def test_the_image_builds_on_the_python_this_project_targets():
    """[LAW:one-source-of-truth] Three declarations of one fact, tied together.

    The Python version is asserted independently in `.python-version`, in
    `pyproject.toml`'s `requires-python`, and in the Dockerfile's `FROM`. Nothing
    connected them, so bumping the first for local development would leave the
    image built and tested against a different interpreter than the one anyone
    develops on — a skew that produces no error at the moment it is introduced.

    Machine-checked rather than remembered: a build-time ARG would still need a
    literal default here, which moves the third copy rather than removing it.
    """
    pinned = (REPO / ".python-version").read_text().strip()

    requires = (REPO / "pyproject.toml").read_text()
    declared = re.search(r'requires-python\s*=\s*"[^0-9]*([0-9]+\.[0-9]+)', requires)
    assert declared, "pyproject.toml does not declare requires-python"

    from_tags = re.findall(
        r"^\s*FROM\s+python:([0-9]+\.[0-9]+)", DOCKERFILE.read_text(), re.MULTILINE
    )
    assert from_tags, "no python base image found in the Dockerfile"

    assert declared.group(1) == pinned, (
        f"pyproject requires-python {declared.group(1)} != .python-version {pinned}"
    )
    for tag in from_tags:
        assert tag == pinned, f"Dockerfile FROM python:{tag} != .python-version {pinned}"


def test_the_exposed_port_is_not_a_second_copy_of_the_configured_one():
    """EXPOSE and HEALTHCHECK both read PORT rather than repeating its value.

    Both were deliberate fixes — a literal here drifts from the port the server
    actually binds, and nothing else would notice. Pinned because the failure is
    silent: the image builds, the container starts, and only publishing or the
    healthcheck is wrong.
    """
    text = DOCKERFILE.read_text()

    expose = re.findall(r"^\s*EXPOSE\s+(.+)$", text, re.MULTILINE)
    assert expose, "no EXPOSE instruction found"
    for value in expose:
        assert "PORT" in value, f"EXPOSE {value!r} hardcodes a port instead of using ${{PORT}}"

    healthcheck = re.search(r"HEALTHCHECK[\s\S]*?(?=\n[A-Z]+\s|\Z)", text)
    assert healthcheck, "no HEALTHCHECK instruction found"
    body = healthcheck.group()
    assert "PORT" in body, "HEALTHCHECK does not read PORT from the environment"
    assert not re.search(r"127\.0\.0\.1:\d+", body), "HEALTHCHECK hardcodes a port"


def test_the_baked_default_voice_is_the_projects_default_voice():
    """[LAW:one-source-of-truth] The ARG default and DEFAULT_VOICE, tied together.

    A Dockerfile ARG cannot read a Python constant at parse time, so the
    duplication is not removable and is machine-checked instead — the same
    approach the Python version pin takes, for the same reason.

    Left to drift, "the default voice" means one thing for an image built with no
    --build-arg and another for `uv run main.py`, and the way that gets noticed is
    by listening to the wrong voice.
    """
    from elvenspeak.piper import DEFAULT_VOICE

    declared = re.search(r"^\s*ARG\s+PIPER_VOICES=(\S+)", DOCKERFILE.read_text(), re.MULTILINE)
    assert declared, "no ARG PIPER_VOICES found"
    assert declared.group(1) == DEFAULT_VOICE


def test_the_baked_default_engine_is_the_registrys_default_engine():
    """The same tie, for the setting that decides which engine gets baked.

    Compared against the registry's *order* rather than the string "piper",
    because the order is what actually decides the default — a maintainer who
    puts a new engine first has moved the default everywhere except this ARG,
    which keeps naming a real engine and so keeps building.

    Worth more than the voice version it mirrors: the image bakes one engine's
    assets and boots one engine, and this ARG is what makes those the same one.
    """
    from elvenspeak.engines import ENGINES

    declared = re.search(
        r"^\s*ARG\s+ELVENSPEAK_ENGINE=(\S+)", DOCKERFILE.read_text(), re.MULTILINE
    )
    assert declared, "no ARG ELVENSPEAK_ENGINE found"
    assert declared.group(1) == next(iter(ENGINES))


def test_substituted_env_values_are_quoted():
    """ENV splits on unescaped whitespace after substitution.

    `--build-arg PIPER_VOICES="a, b"` — the natural way to write a list, and the
    spacing `Settings.from_env` strips specifically to accept — produces a token
    with no `=` and breaks the instruction before the value reaches Python.
    """
    for line in DOCKERFILE.read_text().splitlines():
        for name, value in re.findall(r"(\w+)=(\$\{\w+\})", line):
            assert False, f"unquoted substitution {name}={value}; wrap it in quotes"


#: `RUN uv run python -m <module>`: how the image bakes its voices in. The step
#: used to be `python -c "<source>"` — Python written into this file, which no
#: import of this repo ever reached.
_RUN_MODULE = re.compile(r"^\s*RUN\s+uv\s+run\s+python\s+-m\s+(\S+)\s*$", re.MULTILINE)


def test_the_build_runs_no_python_source_of_its_own():
    """`piper-build-m8h`: the build's code lives in the package, not in this file.

    Source text in a RUN is invisible to every tool that reads this repository —
    not imported, not linted, not covered — and the only thing that executes it
    is a real image build, which by this repo's rules happens in CI after the
    merge. Two escaped defects in two consecutive pull requests came through
    that gap, both past a green suite.

    Asserted as the absence of the whole shape rather than as a check on the
    snippet's contents. Its predecessor here parsed that snippet, resolved its
    imports and bound its calls, and still could not catch the second defect: a
    function that still exists and still takes those arguments can always mean
    less than it used to. The snippet was the problem, so the snippet is gone.
    """
    for step in re.findall(r"^\s*RUN\s+(.*)$", instructions(), re.MULTILINE):
        assert "python -c" not in step, f"RUN executes Python source text: {step[:80]!r}"


def test_the_voice_bake_runs_a_module_that_exists_and_refuses_a_bad_environment():
    """[LAW:verifiable-goals] The exact invocation, executed rather than grepped.

    A bad `PORT` rather than a bad voice, so this needs no network and no model.
    Exit 2 with the problem on stderr proves the module the Dockerfile names is
    importable, that `python -m` reaches an entry point rather than importing
    something inert, and that it reports a bad environment the way every other
    entry point does.

    It stops there, before `bake` is called: what the bake guarantees, and that
    it still reaches `piper._install`, is `tests/test_bake.py`'s subject, which
    calls it directly — the whole reason this step was given a file.
    """
    import subprocess
    import sys

    module = _RUN_MODULE.search(instructions())
    assert module, "no `RUN uv run python -m <module>` bake step found"

    result = subprocess.run(
        [sys.executable, "-m", module.group(1)],
        cwd=REPO,
        env={"PATH": os.environ["PATH"], "PORT": "not-a-number"},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2, (
        f"`python -m {module.group(1)}` exited {result.returncode}, "
        f"not 2\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "PORT" in result.stderr
