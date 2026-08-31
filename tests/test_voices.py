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
from elvenspeak import voices as voices_mod
from elvenspeak.voices import Catalog, VoiceNotInstalled, load_aliases


def voice(key: str) -> Voice:
    return Voice(
        id=key,
        name=key.split("-")[1] if "-" in key else key,
        description=key,
        labels=(("language", "en_US"), ("quality", "medium")),
    )


#: A fixed stand-in for `aliases.toml`, so these tests describe resolution rather
#: than the shipped table's current contents. That file is documented as
#: operator-editable — retargeting a voice without a release is the point of it —
#: and reading the live one meant a correct edit to it failed tests about
#: `Catalog`, which has no opinion about which ids map where.
ALIASES = {"21m00Tcm4TlvDq8ikWAM": "en_US-hfc_female-medium"}


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

    `resolve()` reaches `self._voices[self._fallback]` on its last branch, so a
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
    cat = Catalog.for_engine(_Engine("en_US-lessac-medium"), fallback=None)
    assert [v.id for v in cat.installed] == ["en_US-lessac-medium"]


def test_a_malformed_alias_table_refuses_to_boot(tmp_path, monkeypatch):
    """The reason aliases are read while the app is built, not on first use.

    `aliases.toml` is documented as operator-editable, so a malformed edit is a
    realistic event. Read lazily it surfaced as an uncaught TOMLDecodeError on
    whichever synthesis call first needed an alias — invisible to a healthcheck
    that never touches resolution, and reported nowhere near the file that caused
    it.
    """
    broken = tmp_path / "aliases.toml"
    broken.write_text("[elevenlabs\nnot = valid", encoding="utf-8")
    monkeypatch.setattr(voices_mod, "_ALIASES_FILE", broken)

    with pytest.raises(Exception) as raised:
        Catalog.for_engine(_Engine("en_US-lessac-medium"), fallback=None)
    assert "toml" in type(raised.value).__module__.lower()


def test_aliases_pointing_at_uninstalled_voices_are_dropped(tmp_path, monkeypatch):
    """An answer that cannot be spoken is not an answer."""
    table = tmp_path / "aliases.toml"
    table.write_text(
        '[elevenlabs]\n"live" = "en_US-lessac-medium"\n"dead" = "en_US-absent-medium"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(voices_mod, "_ALIASES_FILE", table)
    assert load_aliases({"en_US-lessac-medium": object()}) == {
        "live": "en_US-lessac-medium"
    }


def test_a_missing_alias_file_is_an_empty_table(tmp_path, monkeypatch):
    """Aliases are optional; their absence is not a failure to boot."""
    monkeypatch.setattr(voices_mod, "_ALIASES_FILE", tmp_path / "nothing-here.toml")
    assert load_aliases({"en_US-lessac-medium": object()}) == {}
