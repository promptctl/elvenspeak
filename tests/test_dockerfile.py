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
    """The Dockerfile with its comments removed.

    Asserting against the raw text matches the prose describing a pattern as
    readily as the pattern itself — the first version of the test below failed on
    the comment explaining the very interpolation it forbids. A check that cannot
    tell code from a description of code is the same substring-against-prose
    mistake in a new costume.
    """
    return "\n".join(
        line for line in DOCKERFILE.read_text().splitlines()
        if not line.lstrip().startswith("#")
    )


def test_the_build_arg_is_not_spliced_into_source_text():
    """It reaches Python through the environment; interpolation was injectable."""
    assert "'${PIPER_VOICES}'" not in instructions()
    assert "os.environ['PIPER_VOICES']" in instructions()


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
