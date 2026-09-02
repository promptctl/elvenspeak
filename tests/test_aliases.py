"""Which voice a request reaches, decided across engines rather than within one.

Two properties over the alias tables and the Dockerfile's per-engine voice ARGs:
every alias target is a voice its own image actually bakes, and no two engines
offer the same voice id. Every image is built from this one repository, so both
are visible statically — in the tree, before a build runs. That makes this the
earliest place either mistake can be caught: earlier than a router refusing at
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


def baked_by_engine() -> dict[str, list[str]]:
    """The voice ids each image installs, keyed by the engine that speaks them.

    The engine's key and its ARG name are one word in two cases, related by
    `PIPER_VOICES` <-> `piper` and spelled here once. Derived rather than looked
    up, so a third engine costs no edit in this file — the same relationship the
    Dockerfile itself relies on to bake the assets for the engine it installs.
    """
    return {
        arg.lower(): [voice.strip() for voice in listing.split(",") if voice.strip()]
        for arg, listing in _VOICES_ARG.findall(instructions())
    }


def baked_voices(name: str) -> list[str]:
    """The voice ids `name`'s image installs, or none if it bakes no list."""
    return baked_by_engine().get(name, [])


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


@pytest.mark.parametrize("name", sorted(baked_by_engine()))
def test_every_baked_voice_list_is_named_after_a_registered_engine(name):
    """An `ARG <ENGINE>_VOICES` naming no engine bakes assets nobody opens.

    The build succeeds and the image carries the models; the engine that boots
    reads its own variable, finds nothing, and falls back to its default voice.
    A typo here is therefore a working image speaking in a voice the operator
    did not choose, which is the same silent-substitution failure the alias
    check above exists for, arriving through the other door.
    """
    assert name in ENGINES, (
        f"ARG {name.upper()}_VOICES names no engine in the registry: {sorted(ENGINES)}"
    )


def test_no_two_engines_offer_the_same_voice_id():
    """[LAW:one-source-of-truth] One voice id, one engine that speaks it.

    A voice id is what a caller sends to reach one specific voice, and the
    router derives its map from what the engines advertise rather than holding a
    table of its own. So an id two engines both offer has two answers and no
    rule that picks between them — the ambiguity `piper-routing-7e2.5` decided
    to refuse rather than resolve by precedence, because first-registered-wins
    hands a live conversation an arbitrary engine silently, and openconv sends a
    bare voice id, so the operator would learn from audio that sounds wrong.

    Caught here as well as at the router's boot, and neither replaces the other:
    this sees one commit's tables and fails before an image exists, while boot
    sees a running fleet where a rolling deploy legitimately mixes image
    versions and a newer image's id can collide with an older one still serving
    — which a check over one tree structurally cannot see.

    Unqualified ids are the reason this can happen at all. Namespacing them as
    `engine/voice` was rejected because `voice_id` is a path segment, so a slash
    in it collides with the route shape.
    """
    baked = baked_by_engine()
    assert baked, "parsed no baked voice lists — the regex is wrong, not the file"

    # Engines rather than occurrences, so that one engine listing a voice twice
    # is reported by the check below and not misread here as a contest between
    # an engine and itself.
    speakers: dict[str, set[str]] = {}
    for name, voices in baked.items():
        for voice in voices:
            speakers.setdefault(voice, set()).add(name)

    contested = sorted(
        f"{voice} is offered by {', '.join(sorted(names))}"
        for voice, names in speakers.items()
        if len(names) > 1
    )
    assert not contested, (
        "two engines offer one voice id, which a router cannot resolve:\n  "
        + "\n  ".join(contested)
    )


@pytest.mark.parametrize("name", sorted(baked_by_engine()))
def test_no_engine_offers_a_voice_twice(name):
    """A duplicate is the tell that the list was edited by appending.

    Harmless on its own — the engine opens the model once either way — and worth
    failing on because it is how the equivalence above stops being the check it
    looks like: a list that grew by appending is a list nobody re-read.
    """
    voices = baked_voices(name)
    assert len(voices) == len(set(voices)), voices


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
