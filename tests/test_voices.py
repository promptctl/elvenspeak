"""Voice resolution, tested without a voice.

`Catalog.resolve` is the substitution contract openconv depends on — it passes
ElevenLabs voice ids straight through precisely because this server guarantees a
response — and it was previously exercised only by tests that skip when no Piper
model is installed. It needs no model: the resolution logic is a lookup over
`Voice` values, so those are constructed directly here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from elvenspeak.voices import Catalog, Voice, VoiceNotInstalled


def voice(key: str) -> Voice:
    return Voice(
        key=key,
        name=key.split("-")[1] if "-" in key else key,
        language="en_US",
        quality="medium",
        model_path=Path(f"/nonexistent/{key}.onnx"),
        sample_rate=22050,
        num_speakers=1,
    )


def catalog(*keys: str, fallback: str | None = None) -> Catalog:
    return Catalog(
        voices={k: voice(k) for k in keys},
        fallback=fallback,
        include_alignments=False,
    )


def test_exact_match_is_not_a_substitution():
    result = catalog("en_US-lessac-medium", fallback="en_US-lessac-medium").resolve(
        "en_US-lessac-medium"
    )
    assert result.voice.key == "en_US-lessac-medium"
    assert result.substituted is False


def test_unknown_id_falls_back_and_is_marked_substituted():
    """The contract openconv relies on, and the flag that keeps it honest."""
    result = catalog("en_US-lessac-medium", fallback="en_US-lessac-medium").resolve(
        "21m00Tcm4TlvDq8ikWAM"
    )
    assert result.voice.key == "en_US-lessac-medium"
    assert result.requested == "21m00Tcm4TlvDq8ikWAM"
    assert result.substituted is True


def test_alias_resolves_when_its_target_is_installed():
    """`21m00Tcm4TlvDq8ikWAM` is Rachel in aliases.toml, mapped to hfc_female."""
    cat = catalog(
        "en_US-lessac-medium", "en_US-hfc_female-medium", fallback="en_US-lessac-medium"
    )
    result = cat.resolve("21m00Tcm4TlvDq8ikWAM")
    assert result.voice.key == "en_US-hfc_female-medium"
    assert result.substituted is True


def test_alias_is_dropped_when_its_target_is_not_installed():
    """An alias naming a voice that cannot speak is not an answer.

    This is why the alias table is inert under the default single-voice install,
    which the README now states explicitly — it was previously documented as
    though the nine ids always resolved.
    """
    cat = catalog("en_US-lessac-medium", fallback="en_US-lessac-medium")
    assert cat.aliases_for("en_US-lessac-medium") == ()
    assert cat.resolve("21m00Tcm4TlvDq8ikWAM").voice.key == "en_US-lessac-medium"


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
    assert [v.key for v in cat.installed] == ["en_US-aaa-medium", "en_US-zzz-medium"]
