"""Startup, and the failures it is supposed to turn into a refusal to boot.

`piper.load()` exists to make a broken deployment one clean failure before the
first request, rather than an unbounded delay or a 500 inside somebody's
synthesis. Its failure paths had no coverage at all, which is the wrong way
round: they are the whole reason opening the engine is separate from serving.

No real Piper model is needed. `_describe` reads only the `.onnx.json` sidecar
beside the weights and never opens the ONNX file itself, and the session that
`load` builds from the weights is stubbed here — the ONNX runtime is not what
these tests are about.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from elvenspeak import piper
from elvenspeak.engine import Prosody

KEY = "en_US-lessac-medium"


def make_voice(models_dir: Path, key: str = KEY, sample_rate: int = 22050) -> None:
    """A voice as far as `load()` can tell: both halves present."""
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


class _StubVoice:
    """A loaded model that produces no audio.

    Enough for every test here: what is under test is which files `load` demands
    and what it reads out of them, not what the ONNX graph does with them.
    """

    @staticmethod
    def load(path: str, include_alignments: bool = False) -> "_StubVoice":
        return _StubVoice()

    def synthesize(self, text: str, **kwargs):
        return iter(())


@pytest.fixture(autouse=True)
def stub_sessions(monkeypatch):
    """Stands in for the ONNX loader, so a placeholder `.onnx` is loadable.

    Patched in `sys.modules` because `load` imports the symbol inside the
    function body, so it resolves the module at call time rather than at import.
    """
    monkeypatch.setitem(
        sys.modules, "piper", type("M", (), {"PiperVoice": _StubVoice})
    )


def load(models_dir: Path, allow_download: bool = False) -> piper.PiperEngine:
    return piper.load(
        keys=(KEY,),
        models_dir=models_dir,
        allow_download=allow_download,
        timings=False,
    )


def test_an_installed_voice_becomes_an_engine(tmp_path):
    make_voice(tmp_path)
    engine = load(tmp_path)
    assert [v.id for v in engine.voices()] == [KEY]
    # Read from the sidecar rather than assumed, because it is the rate the
    # samples really have and the encoder is told it. Reported with the audio
    # rather than on the voice, which is where a caller can act on it.
    spoken = engine.speak(engine.voices()[0], "hello", Prosody())
    assert spoken.sample_rate == 22050


def test_a_missing_voice_refuses_to_boot_when_downloading_is_off(tmp_path):
    """[LAW:no-silent-failure] Serving zero voices is not a degraded success."""
    with pytest.raises(FileNotFoundError, match="not completely installed"):
        load(tmp_path)


def test_weights_without_their_config_are_not_installed(tmp_path):
    """Both halves, because an interrupted download leaves one without the other.

    Checking only the `.onnx` treats a half-copied voice as installed and defers
    the failure to `_describe` or to the first synthesis — later, and further from
    the cause, than the boot this check exists to fail.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{KEY}.onnx").write_bytes(b"not a real model")
    with pytest.raises(FileNotFoundError, match="not completely installed"):
        load(tmp_path)


def test_a_minimal_config_derives_its_metadata_from_the_key(tmp_path):
    """The fallback branch: what a sidecar omits is read back out of the key.

    Every other fixture here writes a complete sidecar, so this derivation and
    the three-part key check gating it were never exercised.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{KEY}.onnx").write_bytes(b"not a real model")
    (tmp_path / f"{KEY}.onnx.json").write_text(
        json.dumps({"audio": {"sample_rate": 16000}}), encoding="utf-8"
    )
    voice = load(tmp_path).voices()[0]
    assert voice.name == "lessac"
    assert voice.labels["language"] == "en_US"
    assert voice.labels["quality"] == "medium"


def test_an_explicitly_null_section_reads_as_an_absent_one(tmp_path):
    """`.get(key, {})` substitutes only for a missing key, not for a null value.

    A hand-edited or half-written sidecar carrying `"language": null` made the
    chained lookup raise AttributeError instead of falling back to the key —
    which is what the surrounding expression already promises to do.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{KEY}.onnx").write_bytes(b"not a real model")
    (tmp_path / f"{KEY}.onnx.json").write_text(
        json.dumps({"language": None, "audio": {"sample_rate": 22050}, "dataset": None}),
        encoding="utf-8",
    )
    voice = load(tmp_path).voices()[0]
    assert voice.labels["language"] == "en_US"
    assert voice.name == "lessac"


def test_engine_facts_with_no_elevenlabs_field_travel_as_labels(tmp_path):
    """The open map is how a voice keeps what the schema has no room for.

    `speakers` in particular: there is no ElevenLabs field to select a speaker
    with, so a multi-speaker model always speaks as its default, and a caller is
    better off reading that than discovering it by listening.
    """
    make_voice(tmp_path)
    (tmp_path / f"{KEY}.onnx.json").write_text(
        json.dumps({"audio": {"sample_rate": 22050}, "num_speakers": 4}),
        encoding="utf-8",
    )
    labels = load(tmp_path).voices()[0].labels
    assert labels["speakers"] == "4"
    assert labels["engine"] == "piper"


def test_a_missing_voice_is_downloaded_when_downloading_is_allowed(tmp_path, monkeypatch):
    """The branch that actually fetches, which nothing entered before."""
    calls = []

    def fake_download(key, directory):
        calls.append((key, Path(directory)))
        make_voice(Path(directory), key)

    monkeypatch.setitem(
        sys.modules,
        "piper.download_voices",
        type("M", (), {"download_voice": staticmethod(fake_download)}),
    )
    engine = load(tmp_path, allow_download=True)
    assert calls == [(KEY, tmp_path)]
    assert [v.id for v in engine.voices()] == [KEY]


def test_a_download_that_produces_nothing_refuses_to_boot(tmp_path, monkeypatch):
    """[LAW:no-silent-failure] Returning is not the same as having delivered.

    `download_voice` reports success by returning, and a half-written pair is the
    same realistic outcome the pre-check exists for. Without a check after the
    call it surfaced as a bare FileNotFoundError from reading the sidecar, which
    names the missing file but not the download that failed to produce it.
    """
    def writes_nothing(key, directory):
        return None

    monkeypatch.setitem(
        sys.modules,
        "piper.download_voices",
        type("M", (), {"download_voice": staticmethod(writes_nothing)}),
    )
    with pytest.raises(FileNotFoundError, match="did not produce"):
        load(tmp_path, allow_download=True)


@pytest.mark.parametrize("rate", [0, -1, None])
def test_a_voice_with_no_usable_sample_rate_refuses_to_boot(tmp_path, rate):
    """Falsy counts as missing, because a zero is worse than an absent key.

    `sample_rate=0` passed an `is None` check and was stored, then failed as a
    ZeroDivisionError inside the alignment's seconds-per-sample — at request
    time, on the timestamp endpoints, far from the sidecar that caused it. There
    is no safe default either: a guessed rate plays perfectly at the wrong pitch.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{KEY}.onnx").write_bytes(b"not a real model")
    (tmp_path / f"{KEY}.onnx.json").write_text(
        json.dumps({"audio": {"sample_rate": rate}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="sample_rate"):
        load(tmp_path)
