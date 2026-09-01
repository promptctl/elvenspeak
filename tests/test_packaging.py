"""What installing this package gets you, checked against what it claims.

`tests/test_encoding.py` proves the seam holds in the import graph: no module of
the ElevenLabs surface can reach an engine library. That proof is worth exactly
nothing to somebody who has not cloned this repository, because the thing they
interact with is the dependency metadata — and until this file existed, that
metadata said the opposite. Every engine's library was a hard requirement, so a
project bringing its own engine installed two ONNX runtimes, a phonemizer and an
espeak wheel it would never call, whose pins were also free to conflict with the
ones its own engine needed.

[FRAMING:representation] Two maps of one territory. The import graph and the
requirement list both answer "what does the reusable half need", and nothing
made them agree. This file is what makes the disagreement expressible as a
failure rather than as a 250 MB surprise.

Read off `pyproject.toml` rather than off the installed environment. The file is
what an installer resolves; a check against `importlib.metadata` would be reading
whatever this checkout last synced, which is a copy of the answer and goes stale
in exactly the direction that hides the bug.
"""

from __future__ import annotations

import ast
import re
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest
from conftest import ENGINE_LIBRARIES

import elvenspeak
from elvenspeak.engines import ENGINES

PYPROJECT = tomllib.loads(
    (Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8")
)

#: The one extra that installs no engine. Named here rather than filtered out by
#: a guess about which names look like engines, so adding an extra is a
#: deliberate act that shows up as a failure and not as a silent third category.
NOT_AN_ENGINE = frozenset({"dev"})

#: Where an engine's alias declarations ship. Read from the package's own
#: directory rather than from `voices._DECLARATIONS`, because what is under test
#: is which files this repository ships — a check that asked the module would go
#: green by agreeing with whatever the module had been changed to look at.
ALIAS_TABLES = Path(elvenspeak.__file__).parent / "aliases"

#: A requirement string's distribution name is everything before the first
#: character that starts an extras list, a version specifier, a URL or a marker.
_NAME = re.compile(r"^[A-Za-z0-9._-]+")


def distribution(requirement: str) -> str:
    """The name `pip` would install, normalised as PEP 503 compares them.

    `piper-tts[alignment]>=1.7.0` and `Piper_TTS` are the same distribution, and
    a comparison that missed it would pass this file's assertions by failing to
    recognise the thing it was looking for — the shape of green that means
    nothing.
    """
    match = _NAME.match(requirement.strip())
    assert match, f"{requirement!r} does not begin with a distribution name"
    return re.sub(r"[-_.]+", "-", match.group()).lower()


def named_in(section: list[str]) -> set[str]:
    return {distribution(requirement) for requirement in section}


EXTRAS: dict[str, list[str]] = PYPROJECT["project"]["optional-dependencies"]
CORE = named_in(PYPROJECT["project"]["dependencies"])

#: Every engine library, by the name that installs it rather than the name that
#: imports it. Resolved from the installed environment because that mapping is a
#: fact about the wheels — `piper` comes from `piper-tts` — that neither this
#: repository nor an import statement records anywhere.
ENGINE_DISTRIBUTIONS = {
    library: distribution(found[0])
    for library in sorted(ENGINE_LIBRARIES)
    for found in [packages_distributions().get(library, [])]
    if found
}


def test_every_engine_library_could_be_resolved():
    """Positive control: the mapping this file compares against is populated.

    `packages_distributions` returns nothing for a module that is not installed,
    so a suite run without an engine's extra would leave that engine unchecked
    below while every assertion still passed. Asserted here so the gap is a
    failure telling you to sync, rather than silence.
    """
    assert set(ENGINE_DISTRIBUTIONS) == ENGINE_LIBRARIES


def test_the_installable_extras_are_the_registered_engines():
    """[LAW:one-source-of-truth] `ELVENSPEAK_ENGINE=x` and `elvenspeak[x]`, one word.

    The correspondence is what lets the Dockerfile install an image's engine by
    interpolating the name it was already built for, and what makes the README's
    instruction derivable rather than remembered. Stated as an equivalence so it
    fails from both sides: an engine registered without an extra is one nobody
    can install the libraries for, and an extra naming no engine is one nothing
    can select.
    """
    assert set(EXTRAS) - NOT_AN_ENGINE == set(ENGINES)


@pytest.mark.parametrize("library", sorted(ENGINE_LIBRARIES))
def test_an_engines_library_is_installed_only_by_that_engines_extra(library: str):
    """The seam as an installer sees it: core carries no engine, and no engine
    carries another.

    This is the regression with history. `kokoro-onnx` was a hard requirement on
    the argument that an engine selectable by name must be importable — which was
    half true and is the reason to check rather than to trust: neither engine
    module imports its library at module scope, so the registry lists both with
    both extras absent, and only the engine actually asked to `open` needs its
    library present.
    """
    dist = ENGINE_DISTRIBUTIONS[library]
    assert dist not in CORE
    installs = {
        extra
        for extra, requirements in EXTRAS.items()
        if dist in named_in(requirements)
    }
    assert len(installs) == 1
    assert installs <= set(ENGINES)


def test_every_shipped_alias_table_is_named_after_a_real_engine():
    """[LAW:one-source-of-truth] The filename is the whole of the binding.

    An engine's alias declarations are the file named after its registry key, so
    the name is not a label on the table — it is the only thing that decides
    whether the table is ever read. A file named `pipe.toml`, or one left behind
    after an engine was renamed, is loaded by nobody and says nothing about it:
    `load_aliases` returns an empty table for an engine with no file, which is
    the correct answer for a supplied engine and indistinguishable from a typo.

    That is the same class of silent nothing this whole area was fixed for. The
    old shared table dropped its entries at INFO, which was at least a line in a
    log; a misnamed file does not even produce that.
    """
    tables = sorted(path.name for path in ALIAS_TABLES.glob("*.toml"))
    assert tables, f"no alias tables found in {ALIAS_TABLES}"
    assert {path.removesuffix(".toml") for path in tables} <= set(ENGINES), tables


@pytest.mark.parametrize(
    "table", sorted(ALIAS_TABLES.glob("*.toml")), ids=lambda path: path.stem
)
def test_a_shipped_alias_table_parses_and_declares_something(table: Path):
    """A table the server would refuse to boot on is one this suite should refuse.

    `Catalog.for_engine` reads these at startup, so a malformed edit is exit 2 on
    a real deployment. Caught here it is a red test instead, which is the same
    failure at the only moment it is still cheap — and the emptiness check is the
    positive control, since a table that parsed to nothing would satisfy every
    other assertion about it while answering for no id at all.
    """
    published = tomllib.loads(table.read_text(encoding="utf-8")).get("elevenlabs", {})
    assert published, f"{table.name} declares no aliases"


#: The two modules an outside engine implements against. Everything they define
#: is part of the obligation, which is why the package root has to name all of it.
SEAM = ("engine", "provisioning")


def defined_in(module: str) -> set[str]:
    """Every public name `elvenspeak/<module>.py` introduces.

    Read from the source rather than from `vars()` on the imported module,
    because at runtime a module's namespace also holds everything it imported —
    `Protocol`, `dataclass`, `Mapping` — and no filter tells those from the names
    the module is offering. The distinction this test is about is authorship, and
    authorship is a fact about the file.
    """
    tree = ast.parse(
        (Path(elvenspeak.__file__).parent / f"{module}.py").read_text(encoding="utf-8")
    )
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
    return {name for name in names if not name.startswith("_")}


def test_the_seam_modules_define_something_to_export():
    """Positive control: the reader still reads.

    A resolver that quietly stopped finding names would make the test below
    assert that a set contains the empty set, which is green and means nothing —
    the exact way this suite's other static check has already failed twice.
    """
    for module in SEAM:
        assert defined_in(module), module


@pytest.mark.parametrize("module", SEAM)
def test_the_package_root_exports_the_whole_seam(module: str):
    """[FRAMING:representation] `__all__` is a map of what this package offers.

    An outside engine's whole obligation is these two modules, and the package
    root claims to be where that obligation is spelled. A name added to either
    one and left unexported does not break anything — it just means the next
    engine author reads the root, concludes the package does not offer it, and
    imports the submodule instead, which is the fork this ticket exists to
    prevent starting one import at a time.
    """
    assert defined_in(module) <= set(elvenspeak.__all__)


def test_nothing_is_exported_twice_or_missing():
    """`__all__` names each thing once and every name resolves.

    A duplicate is harmless to Python and is the tell that the list was edited by
    appending rather than by reading, which is how the grouping above stops being
    true. A name that does not resolve is an `ImportError` for anyone doing the
    `from elvenspeak import *` the list invites.
    """
    assert len(elvenspeak.__all__) == len(set(elvenspeak.__all__))
    for name in elvenspeak.__all__:
        assert hasattr(elvenspeak, name), name


#: How the README tells a reader to start the server. Matched loosely on purpose
#: — what matters is that every spelling of it names an engine, not that there is
#: one spelling.
_README_RUN = re.compile(r"^\s*uv run (.*?)main\.py(?:\s|$)", re.MULTILINE)


def readme_run_commands() -> list[str]:
    return _README_RUN.findall(
        (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")
    )


def test_the_readme_still_documents_a_way_to_start_the_server():
    """Positive control, and it has already been needed.

    A regex over prose is the check most likely to stop matching and go quietly
    vacuous, and the section it reads has been rewritten twice in this PR alone.
    """
    assert readme_run_commands()


@pytest.mark.parametrize("flags", readme_run_commands() or [""])
def test_the_documented_way_to_start_the_server_installs_an_engine(flags: str):
    """The regression this file was extended for, and it shipped once.

    Moving the engines into extras made `uv run main.py` — the README's own
    quickstart, unchanged for the whole life of the project — install the API
    surface with no engine behind it. The default engine then fails to open with
    a `ModuleNotFoundError`, in a codebase whose entire configuration story is
    one clean list of problems and exit 2.

    Nothing caught it. The Dockerfile's references to this repository are checked
    against this repository; the README's were not, and prose is where a command
    goes stale without anything going red.
    """
    named = re.findall(r"--extra\s+(\S+)", flags)
    assert named, f"`uv run {flags}main.py` installs no engine"
    assert set(named) <= set(ENGINES), named
