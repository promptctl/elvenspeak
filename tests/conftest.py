"""Fixtures shared by more than one test module.

[LAW:one-source-of-truth] `make_voice` lives here because two files now need a
voice on disk that no real download produced, and a second copy of "what a
`.onnx.json` has to contain" would be free to drift from the first — leaving one
file testing against a sidecar shape the other has stopped believing in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

#: Every variable `elvenspeak.settings.Settings.from_env` reads. One copy,
#: because a second one is free to drift: the next setting added to `from_env`
#: would be remembered in one test's clearing list and forgotten in the other,
#: and the test that forgot goes flaky later with nothing pointing at the cause.
_ENVIRONMENT = (
    "PIPER_VOICES",
    "PIPER_FALLBACK_VOICE",
    "PIPER_MODELS_DIR",
    "PIPER_ALLOW_DOWNLOAD",
    "ELVENSPEAK_API_KEY",
    "ELVENSPEAK_TIMESTAMPS",
    "HOST",
    "PORT",
)


@pytest.fixture
def clean_env(monkeypatch):
    """An environment holding nothing this service reads.

    For the tests that call `from_env` with no argument — the process-facing
    path, which cannot be handed a dict. Without this they inherit whatever the
    shell exports, and an auto-injected `PORT` or a leftover
    `ELVENSPEAK_TIMESTAMPS` fails them for reasons unrelated to their subject.
    """
    for name in _ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def make_voice(
    models_dir: Path, key: str = "en_US-lessac-medium", sample_rate: int = 22050
) -> None:
    """A voice as far as `install` and `load` can tell: both halves present.

    No real Piper model is needed. `_describe` reads only the `.onnx.json`
    sidecar and never opens the weights, so the `.onnx` here is a placeholder
    whose only job is to exist.

    Every field the key states is read back out of it, rather than fixed at the
    values its first caller happened to use. A sidecar saying `en_US` for a key
    saying `en_GB` is a fixture contradicting itself, and it costs nothing until
    the first test to assert on a language label gets a wrong answer from the
    thing it was trusting to be right.

    The unpack is the check: a key that is not `<lang>-<name>-<quality>` fails
    here and says so, rather than being written to disk as a voice.
    """
    language, dataset, quality = key.split("-")
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / f"{key}.onnx").write_bytes(b"not a real model")
    (models_dir / f"{key}.onnx.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "language": {"code": language},
                "audio": {"sample_rate": sample_rate, "quality": quality},
                "num_speakers": 1,
            }
        ),
        encoding="utf-8",
    )
