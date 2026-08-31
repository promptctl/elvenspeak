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


def test_no_build_arg_is_spliced_into_source_text():
    """Build args reach Python through the environment; interpolation was injectable.

    A single quote in a `--build-arg` once escaped into executable code, because
    the value was interpolated as `'${PIPER_VOICES}'.split(',')` — a
    caller-supplied string inside a Python literal inside a shell command.

    Asserts the property rather than one spelling of it. This previously also
    required `os.environ['PIPER_VOICES']` to appear, which pinned one particular
    way of reading the value: routing the same read through `Settings.from_env`
    satisfied the contract and failed the test anyway.
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


def test_the_voice_list_is_not_parsed_a_second_time():
    """[LAW:one-source-of-truth] The build reads voices the way the server does.

    This step used to re-implement the split-and-strip from `Settings.from_env`
    and carry a comment asserting the two agreed — the shape that makes a
    divergence hard to see rather than impossible, since the comment is what a
    reader trusts instead of checking.
    """
    text = DOCKERFILE.read_text()
    assert "Settings.from_env" in text
    assert "PIPER_VOICES'].split" not in text


def test_the_baked_default_voice_is_the_projects_default_voice():
    """[LAW:one-source-of-truth] The ARG default and DEFAULT_VOICE, tied together.

    A Dockerfile ARG cannot read a Python constant at parse time, so the
    duplication is not removable and is machine-checked instead — the same
    approach the Python version pin takes, for the same reason.

    Left to drift, "the default voice" means one thing for an image built with no
    --build-arg and another for `uv run main.py`, and the way that gets noticed is
    by listening to the wrong voice.
    """
    from elvenspeak.settings import DEFAULT_VOICE

    declared = re.search(r"^\s*ARG\s+PIPER_VOICES=(\S+)", DOCKERFILE.read_text(), re.MULTILINE)
    assert declared, "no ARG PIPER_VOICES found"
    assert declared.group(1) == DEFAULT_VOICE


def test_substituted_env_values_are_quoted():
    """ENV splits on unescaped whitespace after substitution.

    `--build-arg PIPER_VOICES="a, b"` — the natural way to write a list, and the
    spacing `Settings.from_env` strips specifically to accept — produces a token
    with no `=` and breaks the instruction before the value reaches Python.
    """
    for line in DOCKERFILE.read_text().splitlines():
        for name, value in re.findall(r"(\w+)=(\$\{\w+\})", line):
            assert False, f"unquoted substitution {name}={value}; wrap it in quotes"


#: `RUN uv run python -c "<source>"`, with the shell's backslash continuations
#: still in it. The image bakes voices in by calling into this package, so the
#: Dockerfile holds Python that no import of this repo ever reaches.
_RUN_PYTHON = re.compile(r'RUN\s+uv\s+run\s+python\s+-c\s+"(.*?)"', re.DOTALL)


def baked_python() -> str:
    """The Python source the image build executes, ready to parse."""
    joined = DOCKERFILE.read_text().replace("\\\n", "")
    match = _RUN_PYTHON.search(joined)
    assert match, "no `RUN uv run python -c` step found — the regex is wrong"
    return "\n".join(part.strip() for part in match.group(1).split(";"))


def test_the_baked_voice_step_calls_functions_that_exist():
    """The check that would have caught a deleted function, run automatically.

    `piper-engine-axb.2` moved `install` out of `elvenspeak.voices`, and this
    step went on importing it from there. Nothing failed: the test suite never
    executes the Dockerfile, so the only thing that would have reported it was an
    image build, which by this repo's rules happens in CI after the merge.

    [LAW:verifiable-goals] Resolved against the real modules rather than grepped
    for, because the failure has two shapes — a name that is gone, and a name
    that survived with different parameters — and only one of them is visible as
    text.
    """
    import ast
    import importlib
    import inspect

    tree = ast.parse(baked_python())
    imported: dict[str, object] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = importlib.import_module(node.module)
            for alias in node.names:
                # `from x import y` names an attribute of `x` when y is a class
                # or function, and a submodule when it is not — and a submodule
                # is invisible to `hasattr` until something imports it. Both are
                # valid imports, so both are resolved here; only a name that is
                # neither is the failure this test is for.
                found = getattr(module, alias.name, None)
                if found is None:
                    try:
                        found = importlib.import_module(f"{node.module}.{alias.name}")
                    except ImportError:
                        pytest.fail(
                            f"Dockerfile imports {alias.name!r} from {node.module}, "
                            f"which has no such attribute or submodule"
                        )
                imported[alias.asname or alias.name] = found

    assert imported, "parsed no imports — the extraction is wrong, not the file"

    calls = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            target = getattr(imported.get(func.value.id), func.attr, None)
        elif isinstance(func, ast.Name) and func.id in imported:
            target = imported[func.id]
        else:
            continue
        if target is None or not callable(target):
            continue
        calls += 1
        # Binds the call the Dockerfile actually writes, so a kwarg that no
        # longer exists, or a required one it stopped passing, is a failure here
        # rather than at `docker build`.
        inspect.signature(target).bind(
            *(object() for _ in node.args),
            **{kw.arg: object() for kw in node.keywords if kw.arg},
        )

    assert calls, "resolved no calls — the extraction is wrong, not the file"
