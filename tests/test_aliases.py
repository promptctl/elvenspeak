"""Each engine's alias table, checked against the voices its image bakes.

Every image is built from this one repository, so an alias whose target no image
carries is visible statically — in the tree, before a build runs. That makes this
the earliest place the mistake can be caught: earlier than a router refusing at
boot, and far earlier than a caller hearing the wrong voice.

The mistake is not hypothetical and its shape is the reason this file exists.
Until `piper-routing-7e2.7` there was one shared `aliases.toml` naming Piper
voices, and `ARG PIPER_VOICES` baked a single voice. `load_aliases` drops a target
that is not installed, so the Kokoro image resolved none of the nine ElevenLabs
ids and the Piper image resolved none of them either — nine dead aliases in every
image ever published, reported only as one `INFO` line at startup. Everything
downstream was green: the images built, booted, healthchecked and synthesised
fluent audio, in the wrong voice, for every id the callers had been told to use.

[FRAMING:representation] Two maps of one territory. The alias table's values are
voice ids and the Dockerfile's `ARG <ENGINE>_VOICES` is the list of voice ids that
territory actually contains; nothing held them together, so they drifted, and the
drift was silent by construction because dropping is the table's documented
behaviour. This file is the synchronisation — both sides read off the tree, and
neither restated here.

[LAW:one-source-of-truth] In particular the *rule* is not restated. "Which aliases
survive" is `elvenspeak.voices.load_aliases`' question and it already answers it;
this file hands it the voices the image bakes and fails when it drops anything.
A reimplementation of the filter here would be a second answer to that question,
free to disagree with the one the running server uses — which is the class of bug
being checked for, committed by the check itself.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from elvenspeak.engine import Voice
from elvenspeak.engines import ENGINES
from elvenspeak.voices import _DECLARATIONS, load_aliases
from test_dockerfile import DOCKERFILE, instructions

#: `ARG <ENGINE>_VOICES=<comma-separated ids>`, matched against the Dockerfile
#: with its comments removed. The house pattern, for the house's reason: this
#: Dockerfile's comments discuss `ARG PIPER_VOICES` at length, and a check that
#: cannot tell an instruction from prose about the instruction is the
#: substring-against-prose mistake `tests/test_dockerfile.py` was bitten by.
_VOICES_ARG = re.compile(r"^\s*ARG\s+(\w+)_VOICES=(\S+)", re.MULTILINE)


def declaring_engines() -> list[str]:
    """Every engine that ships an alias table, named by its declaration file.

    Discovered rather than listed. An engine's declarations are a file named
    after it and nothing is registered centrally to add one, so a list here would
    be a third place to remember — and the one that fails open, since an engine
    missing from it is an engine whose table nobody checks.
    """
    return sorted(path.stem for path in _DECLARATIONS.glob("*.toml"))


def baked_voices(name: str) -> list[str]:
    """The voice ids `name`'s image installs, read off the Dockerfile.

    The ARG name is derived from the engine's registry key rather than looked up,
    so a third engine needs no edit here — the same relationship the Dockerfile
    itself relies on, spelled once.
    """
    wanted = f"{name.upper()}_VOICES"
    return [
        voice.strip()
        for arg, listing in _VOICES_ARG.findall(instructions())
        if f"{arg}_VOICES" == wanted
        for voice in listing.split(",")
        if voice.strip()
    ]


def declared_aliases(name: str) -> dict[str, str]:
    """Every foreign id `name`'s table claims, before anything is dropped.

    Read from the file rather than through `load_aliases`, which is the one thing
    this file cannot reuse: dropping the dead entries is precisely the behaviour
    under test, so the unfiltered table is what the comparison needs on the left.
    """
    with (_DECLARATIONS / f"{name}.toml").open("rb") as handle:
        return tomllib.load(handle).get("elevenlabs", {})


def test_the_dockerfile_still_declares_voice_lists():
    """Positive control: the reader still reads.

    A regex that quietly stopped matching would make every check below compare an
    empty set against an empty set — green, and asserting nothing. That is how
    this suite's static checks have failed before, so the vacuous case is a
    failure here rather than silence.
    """
    assert _VOICES_ARG.findall(instructions()), (
        f"parsed no `ARG *_VOICES` from {DOCKERFILE} — the regex is wrong, not the file"
    )


def test_some_engine_declares_aliases():
    """The other half of the control, over the side read from the filesystem."""
    assert declaring_engines(), f"found no alias tables under {_DECLARATIONS}"


@pytest.mark.parametrize("name", declaring_engines())
def test_every_alias_table_is_named_after_a_registered_engine(name):
    """A table named after nothing is data no deployment can ever load.

    `load_aliases` opens the file named after the engine that is running, so a
    table under any other name is never read: it does not fail, it does not warn,
    it simply answers for no one. A subset rather than an equivalence, because an
    engine legitimately declaring no aliases has no file and is not a defect — it
    answers for the ids it owns and for nothing else.
    """
    assert name in ENGINES, (
        f"{name}.toml names no engine in the registry: {sorted(ENGINES)}"
    )


@pytest.mark.parametrize("name", declaring_engines())
def test_every_engine_with_aliases_bakes_the_voices_to_answer_with(name):
    """An alias table over an image that installs no voices resolves nothing.

    Asserted separately from the check below so the two failures read differently.
    A missing `ARG <ENGINE>_VOICES` makes every one of that engine's aliases dead
    at once, and reporting that as nine dangling targets would describe the
    symptom while naming none of the cause.
    """
    assert baked_voices(name), (
        f"{name} declares aliases but {DOCKERFILE} has no ARG {name.upper()}_VOICES"
    )


@pytest.mark.parametrize("name", declaring_engines())
def test_no_alias_points_at_a_voice_the_image_does_not_bake(name):
    """[LAW:one-source-of-truth] The baked voices decide; the alias table follows.

    The check the nine dead aliases would have failed. Run through the server's
    own loader so that what counts as a live alias is asked once and answered in
    one place: anything `load_aliases` drops when handed exactly the voices the
    image installs is an entry that would be dropped in the running container,
    for the same reason, reported the same way — as an `INFO` line nobody reads.
    """
    baked = baked_voices(name)
    declared = declared_aliases(name)
    assert declared, f"{name}.toml declares no aliases — the parser is wrong, not the file"

    installed = {
        voice: Voice(id=voice, name=voice, description="") for voice in baked
    }
    live = load_aliases(name, installed)

    dead = sorted(f"{foreign} -> {declared[foreign]}" for foreign in set(declared) - set(live))
    assert not dead, (
        f"{name}.toml aliases voices the image does not bake "
        f"(ARG {name.upper()}_VOICES = {', '.join(baked)}):\n  " + "\n  ".join(dead)
    )
