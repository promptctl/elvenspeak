"""Startup, and the failures it is supposed to turn into a refusal to boot.

`install()` exists to make a broken deployment one clean failure before the first
request, rather than an unbounded delay or a 500 inside somebody's synthesis. Its
failure paths had no coverage at all, which is the wrong way round: they are the
whole reason the function is separate from serving.

No Piper model is needed. `_describe` reads only the `.onnx.json` sidecar beside
the weights and never opens the ONNX file itself, so a placeholder `.onnx` plus a
real sidecar exercises the whole function.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from elvenspeak import voices as voices_mod
from elvenspeak.voices import install, load_aliases

KEY = "en_US-lessac-medium"


def make_voice(models_dir: Path, key: str = KEY, sample_rate: int = 22050) -> None:
    """A voice as far as `install()` can tell: both halves present."""
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / f"{key}.onnx").write_bytes(b"not a real model")
    (models_dir / f"{key}.onnx.json").write_text(
        json.dumps(
            {
                "dataset": key.split("-")[1],
                "language": {"code": "en_US"},
                "audio": {"sample_rate": sample_rate, "quality": "medium"},
                "num_speakers": 1,
            }
        ),
        encoding="utf-8",
    )


def test_an_installed_voice_becomes_a_catalog(tmp_path):
    make_voice(tmp_path)
    catalog = install(
        keys=(KEY,),
        models_dir=tmp_path,
        fallback=KEY,
        include_alignments=False,
        allow_download=False,
    )
    assert [v.key for v in catalog.installed] == [KEY]
    # Read from the sidecar rather than assumed, because it is the rate the
    # samples really have and the encoder is told it.
    assert catalog.get(KEY).sample_rate == 22050


def test_a_missing_voice_refuses_to_boot_when_downloading_is_off(tmp_path):
    """[LAW:no-silent-failure] Serving zero voices is not a degraded success."""
    with pytest.raises(FileNotFoundError, match="not completely installed"):
        install(
            keys=(KEY,),
            models_dir=tmp_path,
            fallback=None,
            include_alignments=False,
            allow_download=False,
        )


def test_weights_without_their_config_are_not_installed(tmp_path):
    """Both halves, because an interrupted download leaves one without the other.

    Checking only the `.onnx` treats a half-copied voice as installed and defers
    the failure to `_describe` or to the first synthesis — later, and further from
    the cause, than the boot this check exists to fail.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{KEY}.onnx").write_bytes(b"not a real model")
    with pytest.raises(FileNotFoundError, match="not completely installed"):
        install(
            keys=(KEY,),
            models_dir=tmp_path,
            fallback=None,
            include_alignments=False,
            allow_download=False,
        )


def test_a_fallback_naming_no_installed_voice_refuses_to_boot(tmp_path):
    make_voice(tmp_path)
    with pytest.raises(ValueError, match="not among the installed voices"):
        install(
            keys=(KEY,),
            models_dir=tmp_path,
            fallback="en_US-somebody-else",
            include_alignments=False,
            allow_download=False,
        )


def test_a_malformed_alias_table_refuses_to_boot(tmp_path, monkeypatch):
    """The reason aliases are read during startup rather than on first use.

    `aliases.toml` is documented as operator-editable, so a malformed edit is a
    realistic event. Read lazily it surfaced as an uncaught TOMLDecodeError on
    whichever synthesis call first needed an alias — invisible to a healthcheck
    that never touches resolution, and reported nowhere near the file that caused
    it.
    """
    make_voice(tmp_path)
    broken = tmp_path / "aliases.toml"
    broken.write_text("[elevenlabs\nnot = valid", encoding="utf-8")
    monkeypatch.setattr(voices_mod, "_ALIASES_FILE", broken)

    with pytest.raises(Exception) as raised:
        install(
            keys=(KEY,),
            models_dir=tmp_path,
            fallback=KEY,
            include_alignments=False,
            allow_download=False,
        )
    assert "toml" in type(raised.value).__module__.lower()


def test_aliases_pointing_at_uninstalled_voices_are_dropped(tmp_path, monkeypatch):
    """An answer that cannot be spoken is not an answer."""
    table = tmp_path / "aliases.toml"
    table.write_text(
        '[elevenlabs]\n"live" = "en_US-lessac-medium"\n"dead" = "en_US-absent-medium"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(voices_mod, "_ALIASES_FILE", table)
    assert load_aliases({KEY: object()}) == {"live": KEY}


def test_a_missing_alias_file_is_an_empty_table(tmp_path, monkeypatch):
    """Aliases are optional; their absence is not a failure to boot."""
    monkeypatch.setattr(voices_mod, "_ALIASES_FILE", tmp_path / "nothing-here.toml")
    assert load_aliases({KEY: object()}) == {}
