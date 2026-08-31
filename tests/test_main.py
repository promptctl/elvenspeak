"""The entry point's one job: report a bad environment the same way, either way.

`Settings.from_env_or_exit` exists because only the script path used to catch
`ConfigError`, so `uvicorn main:build --factory` — a documented way to start
this service — answered a bad environment with a raw traceback carrying every
problem joined onto one line. Every entry point now comes through it, and what
that reporting looks like is `tests/test_settings.py`'s subject; what this file
checks is that the factory path still goes through it.

The success path does need a real voice, and that is the point of the design it
covers: `build()` is the composition root, so it opens the engine before handing
it to the app. A bad deployment therefore fails here, with an exit code, rather
than inside the first request.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi import FastAPI

import main

VOICE = "en_US-lessac-medium"
MODELS = Path(
    os.environ.get("PIPER_MODELS_DIR", Path(__file__).parent.parent / "models")
)


def test_the_factory_entry_point_exits_the_same_way(monkeypatch, capsys):
    """The path that used to raise a traceback instead of reporting."""
    monkeypatch.setenv("PORT", "not-a-number")

    with pytest.raises(SystemExit) as raised:
        main.build()

    assert raised.value.code == 2
    assert "PORT" in capsys.readouterr().err


@pytest.mark.skipif(
    not (MODELS / f"{VOICE}.onnx").exists(),
    reason=f"no {VOICE} model in {MODELS}; "
    "fetch it with PIPER_ALLOW_DOWNLOAD=1 uv run main.py",
)
def test_a_good_environment_builds_an_application(monkeypatch):
    """The success path, so the failure tests are not the only thing exercised.

    Against the installed voice with downloading off, because that is what a
    deployment looks like: the entry point opens the engine, so this covers the
    wiring from environment through to a server that could actually speak.
    """
    monkeypatch.setenv("PIPER_VOICES", VOICE)
    monkeypatch.setenv("PIPER_MODELS_DIR", str(MODELS))
    monkeypatch.setenv("PIPER_ALLOW_DOWNLOAD", "0")
    monkeypatch.delenv("PIPER_FALLBACK_VOICE", raising=False)
    monkeypatch.delenv("ELVENSPEAK_API_KEY", raising=False)
    monkeypatch.setenv("PORT", "5001")
    monkeypatch.setenv("ELVENSPEAK_TIMESTAMPS", "0")

    app = main.build()
    assert isinstance(app, FastAPI)
