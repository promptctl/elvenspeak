"""Which engine a `model_id` names, tested without an engine.

`Directory` is a lookup over two sets of names, so it needs no model, no
network and no engine at all — the same reason `tests/test_voices.py` can test
voice resolution against constructed `Voice` values.

The property that matters here is the three-way answer. A deployment that could
only say "I serve this" and "I do not" has no way to distinguish the caller who
asked for another engine from the caller who sent an ElevenLabs model id nobody
maps, and the two want opposite answers: a refusal, and synthesis with the field
named back as ignored.
"""

from __future__ import annotations

import pytest

from elvenspeak import declarations as declarations_mod
from elvenspeak.models import Directory, Reach
from elvenspeak.provisioning import ConfigError

#: Every engine a build of this repository has. Two of them, which is what makes
#: [`Reach.ELSEWHERE`] reachable at all.
KNOWN = frozenset({"piper", "kokoro"})


def declare(directory, engine_name: str, body: str):
    """Writes one engine's declaration file, where the readers look for it."""
    (directory / f"{engine_name}.toml").write_text(body, encoding="utf-8")
    return directory


@pytest.fixture
def declared(tmp_path, monkeypatch):
    """A declarations directory of this test's own, with piper claiming one id."""
    declare(tmp_path, "piper", 'elevenlabs_models = ["eleven_flash_v2_5"]\n')
    monkeypatch.setattr(declarations_mod, "_DIRECTORY", tmp_path)
    return tmp_path


def test_the_engines_own_name_reaches_it(declared):
    """The unambiguous request, and the one a router will be asked most.

    `openconv` will forward the engine a conversation chose, by name — nothing
    translates it into an ElevenLabs id along the way, so the name has to be a
    legal `model_id` in its own right.
    """
    directory = Directory.for_engine("piper", KNOWN)
    assert directory.reach("piper") is Reach.SERVED


def test_a_declared_foreign_id_reaches_the_engine_that_declared_it(declared):
    assert Directory.for_engine("piper", KNOWN).reach("eleven_flash_v2_5") is (
        Reach.SERVED
    )


def test_another_engines_name_is_elsewhere_rather_than_unknown(declared):
    """The distinction the roster exists for.

    Collapsed into "unknown", this request would be answered in Piper — fluent,
    200, and in an engine the caller explicitly did not ask for.
    """
    assert Directory.for_engine("piper", KNOWN).reach("kokoro") is Reach.ELSEWHERE


def test_an_id_nobody_maps_names_no_engine_here(declared):
    """Reported as ignored rather than refused: a stock ElevenLabs client sends
    a `model_id` on its first request, and every one this deployment does not map
    would otherwise be a 422.
    """
    assert Directory.for_engine("piper", KNOWN).reach("eleven_turbo_v2") is (
        Reach.UNKNOWN
    )


def test_naming_no_model_is_the_same_situation_as_naming_an_unknown_one(declared):
    """Both mean the voice decides, so both are one state and not two."""
    assert Directory.for_engine("piper", KNOWN).reach(None) is Reach.UNKNOWN


def test_a_declaration_claiming_another_engine_refuses_to_boot(tmp_path, monkeypatch):
    """[LAW:no-silent-failure] Both readings are in the table; neither wins quietly.

    An engine listing another engine's name among its foreign ids would make the
    same request both SERVED and ELSEWHERE depending on which lookup ran first.
    Refused at construction, which is startup, so no deployment ever holds one.
    """
    declare(tmp_path, "piper", 'elevenlabs_models = ["kokoro"]\n')
    monkeypatch.setattr(declarations_mod, "_DIRECTORY", tmp_path)

    with pytest.raises(ConfigError) as raised:
        Directory.for_engine("piper", KNOWN)
    assert "kokoro" in str(raised.value)


def test_a_malformed_model_list_refuses_to_boot(tmp_path, monkeypatch):
    """The alternative is an engine that quietly answers for nothing.

    A table this deployment cannot read is exactly as broken as a table naming
    the wrong engine, and it fails the same way every other bad configuration
    does — at startup, with the file named.
    """
    declare(tmp_path, "piper", 'elevenlabs_models = "eleven_flash_v2_5"\n')
    monkeypatch.setattr(declarations_mod, "_DIRECTORY", tmp_path)

    with pytest.raises(ConfigError):
        Directory.for_engine("piper", KNOWN)


def test_an_engine_with_no_declarations_answers_for_its_own_name(tmp_path, monkeypatch):
    """Declaring foreign ids is optional; an engine supplied from outside this
    package has no file and is still reachable by the name that selected it.
    """
    monkeypatch.setattr(declarations_mod, "_DIRECTORY", tmp_path)
    directory = Directory.for_engine("brought-its-own", frozenset({"brought-its-own"}))

    assert directory.listed() == ("brought-its-own",)
    assert directory.reach("brought-its-own") is Reach.SERVED


def test_an_engine_reads_its_own_declarations_and_no_others(declared):
    """[LAW:one-source-of-truth] The declaration belongs to the engine.

    Kokoro's file is right there on disk and claims an id; a Piper deployment
    must not answer for it, because what Kokoro can do is Kokoro's to advertise
    and this process cannot speak a word of it.
    """
    declare(declared, "kokoro", 'elevenlabs_models = ["eleven_multilingual_v2"]\n')

    directory = Directory.for_engine("piper", KNOWN)
    assert directory.reach("eleven_multilingual_v2") is Reach.UNKNOWN
    assert directory.listed() == ("piper", "eleven_flash_v2_5")


def test_the_listing_leads_with_the_engine(declared):
    """The id an operator can rely on comes before the compatibility mappings."""
    declare(
        declared,
        "piper",
        'elevenlabs_models = ["eleven_turbo_v2_5", "eleven_flash_v2_5"]\n',
    )
    assert Directory.for_engine("piper", KNOWN).listed() == (
        "piper",
        "eleven_flash_v2_5",
        "eleven_turbo_v2_5",
    )
