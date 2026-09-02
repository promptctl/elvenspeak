"""Which engine a `model_id` names, tested without an engine.

Two units, and the split is the subject. `declared_by` reads one engine's own
declaration and says what a server running it answers to; `Directory` takes those
answers as they arrive on the voices and says how far a request gets. Neither
needs a model, a network or an engine at all — the same reason
`tests/test_voices.py` can test voice resolution against constructed `Voice`
values.

The property that matters is the three-way answer. A deployment that could only
say "I serve this" and "I do not" has no way to distinguish the caller who asked
for another engine from the caller who sent an ElevenLabs model id nobody maps,
and the two want opposite answers: a refusal, and synthesis with the field named
back as ignored.

The second thing tested here is what `piper-routing-7e2.17` found: with the served
set taken from the deployment's own engine name, a router — whose name declares
nothing — answered for itself alone while fronting two engines. So every question
below is asked twice where the answers differ, once of a single-engine deployment
and once of a fleet, because the whole point is that one rule covers both.
"""

from __future__ import annotations

import pytest

from elvenspeak import declarations as declarations_mod
from elvenspeak.models import Directory, Reach, declared_by
from elvenspeak.provisioning import ConfigError

#: Every engine a build of this repository has. More than one, which is what makes
#: [`Reach.ELSEWHERE`] reachable at all, and `router` because a build that has a
#: router is the case the derivation had wrong.
KNOWN = frozenset({"piper", "kokoro", "router"})


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


def one_engine(*served: str) -> Directory:
    """A deployment whose every voice is spoken by one server answering to `served`."""
    return Directory.over([frozenset(served)], KNOWN)


# ------------------------------------------------ what one engine answers to


def test_the_engines_own_name_is_among_what_it_answers_to(declared):
    """The unambiguous request, and the one a router will be asked most.

    `openconv` forwards the engine a conversation chose, by name — nothing
    translates it into an ElevenLabs id along the way, so the name has to be a
    legal `model_id` in its own right.
    """
    assert "piper" in declared_by("piper", KNOWN)


def test_a_declared_foreign_id_is_answered_for_by_the_engine_that_declared_it(declared):
    assert "eleven_flash_v2_5" in declared_by("piper", KNOWN)


def test_an_engine_reads_its_own_declarations_and_no_others(declared):
    """[LAW:one-source-of-truth] The declaration belongs to the engine.

    Kokoro's file is right there on disk and claims an id; a Piper deployment must
    not answer for it, because what Kokoro can do is Kokoro's to advertise and this
    process cannot speak a word of it.
    """
    declare(declared, "kokoro", 'elevenlabs_models = ["eleven_multilingual_v2"]\n')

    assert declared_by("piper", KNOWN) == {"piper", "eleven_flash_v2_5"}


def test_an_engine_with_no_declarations_answers_for_its_own_name(tmp_path, monkeypatch):
    """Declaring foreign ids is optional; an engine supplied from outside this
    package has no file and is still reachable by the name that selected it.
    """
    monkeypatch.setattr(declarations_mod, "_DIRECTORY", tmp_path)

    assert declared_by("brought-its-own", {"brought-its-own"}) == {"brought-its-own"}


def test_a_declaration_claiming_another_engine_refuses_to_boot(tmp_path, monkeypatch):
    """[LAW:no-silent-failure] Both readings are in the table; neither wins quietly.

    An engine listing another engine's name among its foreign ids would make the
    same request both SERVED and ELSEWHERE depending on which lookup ran first.
    Refused where the file is read, which is startup, so no deployment holds one.
    """
    declare(tmp_path, "piper", 'elevenlabs_models = ["kokoro"]\n')
    monkeypatch.setattr(declarations_mod, "_DIRECTORY", tmp_path)

    with pytest.raises(ConfigError) as raised:
        declared_by("piper", KNOWN)
    assert "kokoro" in str(raised.value)


def test_a_declaration_claiming_its_own_engine_refuses_to_boot(tmp_path, monkeypatch):
    """The other half of the same shape, and the one that duplicates.

    An engine's own name already reaches it, so declaring it again adds nothing but
    a second copy in the listing — a duplicate `model_id` in the one endpoint whose
    job is saying which ids are legal. Refused rather than filtered out: absorbing
    it silently would leave whoever wrote the table believing the list means
    something it does not.
    """
    declare(tmp_path, "piper", 'elevenlabs_models = ["piper"]\n')
    monkeypatch.setattr(declarations_mod, "_DIRECTORY", tmp_path)

    with pytest.raises(ConfigError) as raised:
        declared_by("piper", KNOWN)
    assert "piper" in str(raised.value)


def test_a_malformed_model_list_refuses_to_boot(tmp_path, monkeypatch):
    """The alternative is an engine that quietly answers for nothing.

    A table this deployment cannot read is exactly as broken as a table naming the
    wrong engine, and it fails the same way every other bad configuration does — at
    startup, with the file named.
    """
    declare(tmp_path, "piper", 'elevenlabs_models = "eleven_flash_v2_5"\n')
    monkeypatch.setattr(declarations_mod, "_DIRECTORY", tmp_path)

    with pytest.raises(ConfigError):
        declared_by("piper", KNOWN)


# ------------------------------------------------------- how far a request gets


def test_the_engine_speaking_this_voice_serves_its_own_name():
    assert one_engine("piper").reach("piper", frozenset({"piper"})) is Reach.SERVED


def test_a_foreign_id_the_speaking_engine_declares_is_served():
    here = frozenset({"piper", "eleven_flash_v2_5"})
    assert one_engine(*here).reach("eleven_flash_v2_5", here) is Reach.SERVED


def test_another_engines_name_is_elsewhere_rather_than_unknown():
    """The distinction the roster exists for.

    Collapsed into "unknown", this request would be answered in Piper — fluent,
    200, and in an engine the caller explicitly did not ask for.
    """
    assert one_engine("piper").reach("kokoro", frozenset({"piper"})) is Reach.ELSEWHERE


def test_an_id_nobody_maps_names_no_engine_here():
    """Reported as ignored rather than refused: a stock ElevenLabs client sends a
    `model_id` on its first request, and every one this deployment does not map
    would otherwise be a 422.
    """
    assert one_engine("piper").reach("eleven_turbo_v2", frozenset({"piper"})) is (
        Reach.UNKNOWN
    )


def test_naming_no_model_is_the_same_situation_as_naming_an_unknown_one():
    """Both mean the voice decides, so both are one state and not two."""
    assert one_engine("piper").reach(None, frozenset({"piper"})) is Reach.UNKNOWN


def test_the_listing_leads_with_the_engines_and_then_their_mappings():
    """The ids an operator can rely on come before the compatibility mappings."""
    directory = one_engine("piper", "eleven_turbo_v2_5", "eleven_flash_v2_5")

    assert directory.listed() == (
        "piper",
        "eleven_flash_v2_5",
        "eleven_turbo_v2_5",
    )


# ------------------------------------------------------------------- a fleet

#: What each backend of a two-engine router answers to, as its voices report it.
PIPER_SERVES = frozenset({"piper", "eleven_flash_v2_5"})
KOKORO_SERVES = frozenset({"kokoro", "eleven_multilingual_v2"})


def fleet() -> Directory:
    """A router fronting a piper and a kokoro, derived the way `api` derives it."""
    return Directory.over([PIPER_SERVES, KOKORO_SERVES], KNOWN)


def test_a_fleet_advertises_every_engine_it_fronts_and_what_they_honour():
    """`piper-routing-7e2.17`'s first symptom, at the unit that had it wrong.

    Measured against the running router at 2026.09.02.4: `GET /v1/models` answered
    `["router"]` while piper and kokoro sat behind it advertising five ids between
    them. The name it derived that from declares nothing and never will, so the
    union over what the voices carry is the only answer that can be right.
    """
    assert fleet().listed() == (
        "kokoro",
        "piper",
        "eleven_flash_v2_5",
        "eleven_multilingual_v2",
    )


def test_a_router_does_not_refuse_the_engines_it_is_running():
    """The second symptom: `model_id=piper` came back 422, served `["router"]`.

    An engine the fleet fronts is not absent, and a request naming it alongside one
    of its own voices is the ordinary way to choose an engine per request — which
    is the epic's stated outcome.
    """
    assert fleet().reach("piper", PIPER_SERVES) is Reach.SERVED
    assert fleet().reach("eleven_multilingual_v2", KOKORO_SERVES) is Reach.SERVED


def test_an_engine_that_is_running_but_not_speaking_this_voice_is_refused():
    """The lie a deployment-wide answer would have told instead.

    Widening the router's served set without asking *which* voice resolved would
    make this SERVED: the caller asks for piper, a kokoro voice speaks, and the
    header reports `model_id` as honoured. That is the silent wrong-engine answer
    this module exists to refuse, and it is only representable because the set the
    request is judged against comes from the voice.
    """
    assert fleet().reach("piper", KOKORO_SERVES) is Reach.ELSEWHERE
    assert fleet().reach("eleven_flash_v2_5", KOKORO_SERVES) is Reach.ELSEWHERE


def test_an_id_no_backend_serves_is_still_merely_unknown_behind_a_router():
    """A fleet does not turn a stock ElevenLabs id into a 422.

    The reason for `Reach.UNKNOWN` does not change with the number of engines: a
    client that sends `eleven_turbo_v2` to a router is not naming an engine, so it
    gets audio with the field reported ignored rather than a refusal.
    """
    assert fleet().reach("eleven_turbo_v2", PIPER_SERVES) is Reach.UNKNOWN


def test_a_build_engine_nobody_is_running_is_still_elsewhere():
    """`known` is the build's roster, so an engine present but unfronted refuses.

    A router that discovered only piper must still refuse `kokoro` rather than
    answering in piper — the fleet shrank, and that is exactly when the refusal
    matters.
    """
    piper_only = Directory.over([PIPER_SERVES], KNOWN)

    assert piper_only.reach("kokoro", PIPER_SERVES) is Reach.ELSEWHERE


def test_a_deployment_with_no_voices_advertises_nothing():
    """A router that discovered nothing, which `open()` deliberately allows.

    It has no voices, so `/health` is 503 and nothing is routed to it. The listing
    agreeing — rather than naming an engine no voice can reach — is what keeps the
    two answers from disagreeing about whether this process can speak.
    """
    assert Directory.over([], KNOWN).listed() == ()
