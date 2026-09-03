"""Voice resolution, tested without a voice.

`Catalog.resolve` is the substitution contract openconv depends on — it passes
ElevenLabs voice ids straight through precisely because this server guarantees a
response — and it was previously exercised only by tests that skip when no Piper
model is installed. It needs no model: the resolution logic is a lookup over
`Voice` values, so those are constructed directly here.
"""

from __future__ import annotations

import pytest

from elvenspeak.engine import Voice
from conftest import SERVES
from elvenspeak import declarations as declarations_mod
from elvenspeak.provisioning import ConfigError
from elvenspeak.voices import (
    Catalog,
    Substitution,
    VoiceNotInstalled,
    load_aliases,
)


def voice(key: str) -> Voice:
    """A stand-in voice, its language read off its key as Piper reads a real one.

    Piper keys are `<family>_<REGION>-<name>-<quality>`, so deriving the language
    here rather than passing it means a test naming `es_ES-...` gets a Spanish
    voice without saying so twice — and a key and a language cannot be set to
    disagree in a fixture in a way no real voice could.
    """
    return Voice(
        id=key,
        name=key.split("-")[1] if "-" in key else key,
        description=key,
        labels=(("quality", "medium"),),
        models=SERVES,
        language=key.split("_")[0] if "_" in key else "en",
    )


#: A fixed stand-in for a shipped declaration file, so these tests describe
#: resolution rather than the current contents of any engine's table. Reading a
#: live one meant that retargeting a voice failed tests about `Catalog`, which
#: has no opinion about which ids map where.
ALIASES = {"21m00Tcm4TlvDq8ikWAM": "en_US-hfc_female-medium"}

#: An engine name no declaration file is shipped for. The tests below are about
#: resolution over a table they were handed, so they ask for the one engine
#: whose table is empty — naming a real engine would quietly make them assert
#: against whatever that engine currently declares.
_UNDECLARED = "no-such-engine"


def catalog(
    *keys: str,
    fallback: str | None = None,
    aliases: dict[str, str] | None = None,
) -> Catalog:
    table = ALIASES if aliases is None else aliases
    return Catalog(
        voices={k: voice(k) for k in keys},
        fallback=fallback,
        # Filtered as `load_aliases` filters it, since an alias whose target is
        # not installed is dropped at load rather than carried into resolution.
        aliases={f: local for f, local in table.items() if local in keys},
    )


def test_exact_match_is_not_a_substitution():
    result = catalog("en_US-lessac-medium", fallback="en_US-lessac-medium").resolve(
        "en_US-lessac-medium"
    )
    assert result.voice.id == "en_US-lessac-medium"
    assert result.substituted is False


def test_unknown_id_falls_back_and_is_marked_substituted():
    """The contract openconv relies on, and the flag that keeps it honest."""
    result = catalog("en_US-lessac-medium", fallback="en_US-lessac-medium").resolve(
        "21m00Tcm4TlvDq8ikWAM"
    )
    assert result.voice.id == "en_US-lessac-medium"
    assert result.requested == "21m00Tcm4TlvDq8ikWAM"
    assert result.substituted is True


def test_alias_resolves_when_its_target_is_installed():
    """An aliased id reaches its target, and still reports the swap."""
    cat = catalog(
        "en_US-lessac-medium", "en_US-hfc_female-medium", fallback="en_US-lessac-medium"
    )
    result = cat.resolve("21m00Tcm4TlvDq8ikWAM")
    assert result.voice.id == "en_US-hfc_female-medium"
    assert result.substituted is True


def test_alias_is_dropped_when_its_target_is_not_installed():
    """An alias naming a voice that cannot speak is not an answer.

    This is why the alias table is inert under the default single-voice install,
    which the README now states explicitly — it was previously documented as
    though the nine ids always resolved.
    """
    cat = catalog("en_US-lessac-medium", fallback="en_US-lessac-medium")
    assert cat.aliases_for("en_US-lessac-medium") == ()
    assert cat.resolve("21m00Tcm4TlvDq8ikWAM").voice.id == "en_US-lessac-medium"


def test_live_aliases_are_reported_for_discovery():
    cat = catalog(
        "en_US-lessac-medium", "en_US-hfc_female-medium", fallback="en_US-lessac-medium"
    )
    assert "21m00Tcm4TlvDq8ikWAM" in cat.aliases_for("en_US-hfc_female-medium")


def test_no_fallback_means_an_unknown_id_is_refused():
    """Substitution off is a deployment choice, and then unknown ids are 404s."""
    cat = catalog("en_US-lessac-medium", fallback=None)
    with pytest.raises(VoiceNotInstalled) as raised:
        cat.resolve("21m00Tcm4TlvDq8ikWAM")
    assert raised.value.requested == "21m00Tcm4TlvDq8ikWAM"


def test_get_does_not_substitute():
    cat = catalog("en_US-lessac-medium", fallback="en_US-lessac-medium")
    assert cat.get("en_US-lessac-medium") is not None
    assert cat.get("21m00Tcm4TlvDq8ikWAM") is None


def test_installed_is_stable_order():
    cat = catalog("en_US-zzz-medium", "en_US-aaa-medium", fallback="en_US-aaa-medium")
    assert [v.id for v in cat.installed] == ["en_US-aaa-medium", "en_US-zzz-medium"]


def test_a_fallback_that_is_not_installed_is_refused_at_construction():
    """[LAW:parse-dont-validate] The bad state stops being constructible.

    `resolve()` reaches `self._voices[self.fallback]` on its last branch, so a
    fallback naming no installed voice turned every unrecognised id — precisely
    what the fallback is for — into a bare KeyError raised from inside synthesis,
    far from the configuration that caused it. Checked at construction, no
    Catalog anywhere can be in that state, so no caller has to ask.
    """
    with pytest.raises(ValueError, match="not among the installed voices"):
        Catalog(
            voices={"en_US-lessac-medium": voice("en_US-lessac-medium")},
            fallback="en_US-not-installed",
        )


def test_both_catalog_problems_are_reported_together():
    """One restart, the whole list — the promise `ConfigError` carries.

    This module was the last one raising on the first problem it found, which
    only became a contradiction when these checks started raising the type whose
    docstring promises the opposite. An operator with a bad fallback and a
    dangling alias would fix one, restart, and meet the other.
    """
    with pytest.raises(ConfigError) as raised:
        Catalog(
            voices={"en_US-lessac-medium": voice("en_US-lessac-medium")},
            fallback="en_US-not-installed",
            aliases={"dead": "en_US-also-not-installed"},
        )
    problems = raised.value.problems
    assert len(problems) == 2
    joined = " ".join(problems)
    assert "en_US-not-installed" in joined
    assert "en_US-also-not-installed" in joined


def test_an_alias_pointing_at_no_installed_voice_is_refused_at_construction():
    """The alias half of the same invariant the fallback already holds.

    `resolve()` indexes `_voices[aliased]` on its alias branch, so a dangling
    target is the same bare KeyError from inside a request that checking the
    fallback at construction was meant to make unreachable — the constructor was
    only enforcing it for one of its two parameters.

    Refused rather than filtered: dropping uninstalled targets is `load_aliases`'
    job and it logs how many it dropped, whereas a constructor that silently
    discarded a caller's entry would report nothing at all.
    """
    with pytest.raises(ValueError, match="alias targets are not among"):
        Catalog(
            voices={"en_US-lessac-medium": voice("en_US-lessac-medium")},
            fallback="en_US-lessac-medium",
            aliases={"21m00Tcm4TlvDq8ikWAM": "en_US-not-installed"},
        )


class _Engine:
    """An engine that offers voices and can do nothing else.

    [`Catalog`] is a lookup over values, so this is the whole of what it needs
    from an engine — which is the property being demonstrated as much as used.
    """

    def __init__(self, *keys: str) -> None:
        self._voices = tuple(voice(key) for key in keys)

    def voices(self) -> tuple[Voice, ...]:
        return self._voices


def test_a_catalog_is_built_from_whatever_the_engine_offers():
    cat = Catalog.for_engine(
        _UNDECLARED, _Engine("en_US-lessac-medium"), fallback=Substitution.OFF
    )
    assert [v.id for v in cat.installed] == ["en_US-lessac-medium"]


def test_an_unnamed_fallback_becomes_the_first_voice_the_engine_offers():
    """Where "whichever one you have" is finally answerable.

    The setting cannot name a voice on its own any more: the list belongs to the
    engine, so the operator who named none is asking a question only a loaded
    engine can answer. Answered here, once, and stored as an ordinary id — so
    everything downstream, the operator log included, reads a voice rather than
    an instruction.

    Offered first, not sorted first. The engine here lists its voices in an
    order that is deliberately *not* alphabetical, because with an already-sorted
    fixture this test passes whether the fallback follows the engine's order or
    quietly re-sorts — and a re-sort is precisely the bug that shipped: it hands
    the deployment a default voice its operator never chose.
    """
    cat = Catalog.for_engine(
        _UNDECLARED,
        _Engine("en_US-zzz-medium", "en_US-aaa-medium"),
        fallback=Substitution.FIRST_OFFERED,
    )
    assert cat.fallback == "en_US-zzz-medium"
    assert cat.resolve("some-elevenlabs-id").substituted


def test_switching_substitution_off_does_not_quietly_pick_a_voice():
    """[LAW:types-are-the-program] The two answers must not collapse again.

    While both were spelled `None`, "I named no voice" and "I want no
    substitution" were one value, and the only reason the closed deployment got
    what it asked for was that parsing resolved the first case early. It cannot
    now, so an enum carries the difference this far — and this is the test that
    goes red if either member is ever mapped onto the other.
    """
    cat = Catalog.for_engine(
        _UNDECLARED, _Engine("en_US-aaa-medium"), fallback=Substitution.OFF
    )
    assert cat.fallback is None
    with pytest.raises(VoiceNotInstalled):
        cat.resolve("some-elevenlabs-id")


def test_the_resolved_fallback_cannot_be_reassigned():
    """[LAW:types-are-the-program] The constructor check runs once, so it has to hold.

    `resolve()` indexes `_voices` with the fallback on its last branch, and the
    membership check that makes that safe happens at construction. A writable
    attribute would let later code put back the bare-KeyError-inside-synthesis
    state that check exists to make unreachable — and the constructor's comment
    would go on claiming no Catalog can reach it.
    """
    cat = catalog("en_US-lessac-medium", fallback="en_US-lessac-medium")
    with pytest.raises(AttributeError):
        cat.fallback = "en_US-not-installed"


def test_an_engine_offering_nothing_leaves_nothing_to_substitute_with():
    """The one case where "the first voice" has no answer.

    Not an error this function could report more usefully than the catalog
    already does: every id is refused by name, which is what an empty engine
    means whichever way the fallback was configured.
    """
    cat = Catalog.for_engine(
        _UNDECLARED, _Engine(), fallback=Substitution.FIRST_OFFERED
    )
    assert cat.fallback is None
    with pytest.raises(VoiceNotInstalled):
        cat.resolve("anything")


def declare(directory, engine_name: str, body: str):
    """Writes one engine's declaration file, where `load_aliases` looks for it."""
    (directory / f"{engine_name}.toml").write_text(body, encoding="utf-8")
    return directory


@pytest.mark.parametrize(
    ("body", "tells"),
    [
        # An unclosed `[elevenlabs`, which `tomllib` gives up on at column 12.
        pytest.param(b"[elevenlabs\nnot = valid", "line 1, column 12", id="bad-syntax"),
        # A file saved in some other encoding. It never reaches the parser at
        # all: `tomllib.load` decodes as UTF-8 first, so this used to be the
        # second door out of the same room -- a `UnicodeDecodeError` is no kind
        # of `TOMLDecodeError`, and a handler naming that type would have let
        # this one back out as the traceback the other case no longer is.
        pytest.param(b"\xff\xfe[elevenlabs]\n", "utf-8", id="bad-encoding"),
    ],
)
def test_a_malformed_alias_table_refuses_to_boot(tmp_path, monkeypatch, body, tells):
    """The reason aliases are read while the app is built, not on first use.

    Read lazily it surfaced on whichever synthesis call first needed an alias —
    invisible to a healthcheck that never touches resolution, and reported
    nowhere near the file that caused it.

    [LAW:behavior-not-structure] This used to assert that the escaping exception
    came from the `tomllib` module, which pinned the *plumbing*: it passed only
    while the failure stayed untranslated, and `settings.reported_or_exit`
    catches `ConfigError` and nothing else, so what it was really guarding was an
    operator getting a traceback. The contract is the one its valid-TOML
    neighbour already states — the file is named, in the list every other startup
    problem joins — and whatever `tomllib` said about where it stopped rides
    along, because a bad byte in a hundred-line table is not findable from the
    filename alone.

    [LAW:one-type-per-behavior] The two bodies are one behaviour's instances, not
    two tests: "this file could not be turned into a table" is the whole of what
    an operator is being told, and the two ways to earn that sentence differ only
    in which bytes provoke it.
    """
    (tmp_path / "piper.toml").write_bytes(body)
    monkeypatch.setattr(declarations_mod, "_DIRECTORY", tmp_path)

    with pytest.raises(ConfigError) as raised:
        Catalog.for_engine(
            "piper", _Engine("en_US-lessac-medium"), fallback=Substitution.OFF
        )
    assert "piper.toml" in str(raised.value)
    assert tells in str(raised.value)


def test_a_declaration_that_cannot_be_opened_refuses_to_boot(tmp_path, monkeypatch):
    """The third door out of the same room: the file never opens at all.

    Neither a parse nor a decode, so it reaches neither of the arms above --
    `OSError` is no kind of `ValueError` -- and it would have gone back to being
    the traceback the other two no longer are. Answered in its own sentence
    because it sends the operator somewhere else entirely: a mount or a
    permission bit, not the file's contents.

    A directory rather than a `chmod 000` file, and that is not fussiness: the
    gitea runner executes this suite as root (`user: root (uid 0)`, printed by
    the publish workflow), and root reads a mode-000 file happily. That test
    would pass here and quietly stop testing anything in the gate that matters.
    `open("rb")` on a directory refuses whoever asks.
    """
    (tmp_path / "piper.toml").mkdir()
    monkeypatch.setattr(declarations_mod, "_DIRECTORY", tmp_path)

    with pytest.raises(ConfigError) as raised:
        Catalog.for_engine(
            "piper", _Engine("en_US-lessac-medium"), fallback=Substitution.OFF
        )
    assert "piper.toml" in str(raised.value)
    assert "could not be opened" in str(raised.value)


def test_an_engine_reads_its_own_declarations_and_no_others(tmp_path, monkeypatch):
    """[LAW:one-source-of-truth] The table belongs to the engine, not the server.

    The whole of what went wrong before: one shared table named Piper voices, so
    the Kokoro image loaded nine ids pointing at voices it could not possibly
    have and dropped every one. A file per engine makes that unrepresentable —
    a table can only ever name voices its own engine was meant to offer.

    Both files are present here, which is the case that matters: the engine that
    is running must not see the other's entries even when they are right there
    on disk.
    """
    declare(tmp_path, "piper", '[elevenlabs]\n"shared-id" = "en_US-lessac-medium"\n')
    declare(tmp_path, "kokoro", '[elevenlabs]\n"shared-id" = "af_heart"\n')
    monkeypatch.setattr(declarations_mod, "_DIRECTORY", tmp_path)

    assert load_aliases("piper", {"en_US-lessac-medium": object()}) == {
        "shared-id": "en_US-lessac-medium"
    }
    assert load_aliases("kokoro", {"af_heart": object()}) == {"shared-id": "af_heart"}
    # The Piper entry is on disk and its target is installed, and it still does
    # not reach Kokoro — which is the claim, since dropping-by-target would give
    # an empty table here for the wrong reason.
    assert load_aliases("kokoro", {"en_US-lessac-medium": object()}) == {}


def test_aliases_pointing_at_uninstalled_voices_are_dropped(tmp_path, monkeypatch):
    """An answer that cannot be spoken is not an answer."""
    declare(
        tmp_path,
        "piper",
        '[elevenlabs]\n"live" = "en_US-lessac-medium"\n"dead" = "en_US-absent-medium"\n',
    )
    monkeypatch.setattr(declarations_mod, "_DIRECTORY", tmp_path)
    assert load_aliases("piper", {"en_US-lessac-medium": object()}) == {
        "live": "en_US-lessac-medium"
    }


@pytest.mark.parametrize(
    "body",
    [
        'elevenlabs = "oops"\n',
        '[elevenlabs]\n"live" = 3\n',
        '[elevenlabs]\n"live" = ["en_US-lessac-medium"]\n',
    ],
    ids=["not-a-table", "target-is-a-number", "target-is-a-list"],
)
def test_an_alias_table_of_the_wrong_shape_refuses_to_boot(tmp_path, monkeypatch, body):
    """The operator should read which file they mistyped, not a traceback.

    Distinct from the unparseable file above: this one is valid TOML saying
    something the table cannot mean, so it gets past `tomllib` and the shape is
    the only thing left that can catch it.

    Every shape here used to travel as far as `Catalog.for_engine` and die on a
    bare `AttributeError` or `TypeError`, which `reported_or_exit` does not catch
    because it catches `ConfigError` and nothing else — so a one-character typo
    came back as a stack trace rather than as the file's name. Held to the same
    standard as `elevenlabs_models`, for the same reason and in the same place.
    """
    declare(tmp_path, "piper", body)
    monkeypatch.setattr(declarations_mod, "_DIRECTORY", tmp_path)

    with pytest.raises(ConfigError) as raised:
        load_aliases("piper", {"en_US-lessac-medium": object()})
    assert "piper.toml" in str(raised.value)


def test_an_engine_with_no_declarations_gets_an_empty_table(tmp_path, monkeypatch):
    """Aliases are optional; their absence is not a failure to boot.

    The case a supplied engine arrives in — it answers for the ids it owns and
    for nothing else, and there is nothing to register anywhere to say so.
    """
    monkeypatch.setattr(declarations_mod, "_DIRECTORY", tmp_path)
    assert load_aliases("brought-its-own", {"en_US-lessac-medium": object()}) == {}


# --------------------------------------------------------------- language


def test_a_language_outranks_the_voice_id_that_was_named():
    """Asking for Spanish gets Spanish, even when the id names an English voice.

    This is the case openconv actually produces: an agent configured with an
    English default voice and `language: es`. The two cannot both be honoured —
    our voices are monolingual — and the voice is the one that gives, because a
    substitute voice is still an answer to "say this" while English phonemes over
    Spanish text is a different answer that sounds like a correct one.
    """
    cat = catalog("en_US-lessac-medium", "es_ES-davefx-medium", fallback="en_US-lessac-medium")

    spoken = cat.resolve("en_US-lessac-medium", "es")

    assert spoken.voice.id == "es_ES-davefx-medium"
    assert spoken.voice.language == "es"
    # Reported like every other substitution, so `x-elvenspeak-voice` names what
    # actually spoke rather than echoing what was asked for.
    assert spoken.substituted


def test_a_voice_that_already_speaks_the_language_is_not_a_substitution():
    """The narrowing must not turn a correct request into a substituted one.

    A caller that names a Spanish voice and says `es` agreed with itself, and a
    response claiming it was given something else would send a client looking for
    a substitution that never happened.
    """
    cat = catalog("en_US-lessac-medium", "es_ES-davefx-medium", fallback="en_US-lessac-medium")

    spoken = cat.resolve("es_ES-davefx-medium", "es")

    assert spoken.voice.id == "es_ES-davefx-medium"
    assert not spoken.substituted


def test_a_language_no_voice_speaks_steers_nothing():
    """An unanswerable language is reported, never refused.

    The rule `model_id` already set: an id this deployment does not map steers
    nothing and comes back in `x-elvenspeak-ignored`. A language nothing speaks is
    the same unanswerable ask, and refusing it would 404 a voice that is installed
    and was never the problem — breaking a stock client that sends
    `language_code` on every request.
    """
    cat = catalog("en_US-lessac-medium", fallback="en_US-lessac-medium")

    spoken = cat.resolve("en_US-lessac-medium", "ja")

    assert spoken.voice.id == "en_US-lessac-medium"
    assert not spoken.substituted


def test_a_language_is_answered_even_when_the_fallback_does_not_speak_it():
    """The fallback is a default, not a veto.

    A deployment's fallback is one configured id, so it speaks one language. If it
    outranked the narrowing, then every deployment whose fallback is English would
    answer every Spanish request in English — which is the whole failure this
    change exists to remove, reintroduced through the back door.
    """
    cat = catalog(
        "en_US-lessac-medium", "es_MX-claude-high", fallback="en_US-lessac-medium"
    )

    spoken = cat.resolve("no-such-voice", "es")

    assert spoken.voice.id == "es_MX-claude-high"
    assert spoken.substituted


def test_an_alias_is_not_followed_out_of_the_requested_language():
    """An alias maps an id onto a voice, and cannot override the language.

    The shipped tables point ElevenLabs' English speakers at English voices by
    design. Following one while Spanish was asked for would answer a Spanish
    request in English through a table that has no opinion about language at all.
    """
    cat = catalog(
        "en_US-hfc_female-medium",
        "es_MX-claude-high",
        fallback="en_US-hfc_female-medium",
    )

    spoken = cat.resolve("21m00Tcm4TlvDq8ikWAM", "es")

    assert spoken.voice.id == "es_MX-claude-high"


def test_a_language_cannot_steer_a_deployment_that_switched_substitution_off():
    """The false 404 that language narrowing introduced, and the rule that ends it.

    Narrowing is a substitution — it answers with a voice other than the one the
    caller named — and it arrived as the only one not answering to the switch that
    governs the rest. With no fallback configured the narrowed table did not hold
    the exact id, the alias step missed too, and `resolve` raised
    `VoiceNotInstalled` for a voice whose own message then listed it as available:
    a 404 for `en_US-lessac-high` on a deployment that had baked
    `en_US-lessac-high`.

    Two installed voices, not one, because that is what makes the narrowing bite:
    a catalog where nothing speaks Spanish falls back to the whole table via
    `speaking` and resolves correctly by accident.

    The answer is the id the caller named, unsubstituted. `api.py` then reports
    `language_code` in `x-elvenspeak-ignored`, since the voice does not speak it —
    which is the honest answer a 404 was not.
    """
    cat = catalog("en_US-lessac-high", "es_MX-claude-high", fallback=None)

    spoken = cat.resolve("en_US-lessac-high", "es")

    assert spoken.voice.id == "en_US-lessac-high"
    assert not spoken.substituted


def test_switching_substitution_off_still_refuses_an_id_nothing_installs():
    """The other half, which the fix above must not have bought at its expense.

    Ungating narrowing where there is no fallback could as easily have been done
    by ignoring the absent fallback, which would answer every unknown id with
    whatever spoke the language — turning the switch off into a substitution rule
    of its own. The refusal is the whole point of the setting and survives.
    """
    cat = catalog("en_US-lessac-high", "es_MX-claude-high", fallback=None)

    with pytest.raises(VoiceNotInstalled):
        cat.resolve("en_US-not-installed", "es")


def test_an_alias_reaches_its_voice_where_substitution_is_off():
    """The gate covers all three steps, and the alias step is why that is right.

    Narrowing the alias step alone — while leaving the exact-id step gated, which
    is what "narrow for aliases even with substitution off" would mean — puts the
    alias step back into the false 404 the gate was added to end: the alias
    resolves to a voice narrowing has removed, `aliased in speaking` misses, and
    `VoiceNotInstalled` names as *available* the very voice that would have
    answered.

    So the alias is followed and `language_code` comes back reported in
    `x-elvenspeak-ignored`, which differs from the substitution-on case above by
    design. A deployment with no fallback has not opted out of aliasing — it has
    opted out of having anywhere for a narrowed-away id to land.
    """
    cat = catalog(
        "en_US-hfc_female-medium",
        "es_MX-claude-high",
        fallback=None,
    )

    spoken = cat.resolve("21m00Tcm4TlvDq8ikWAM", "es")

    assert spoken.voice.id == "en_US-hfc_female-medium"
    assert spoken.substituted is True
