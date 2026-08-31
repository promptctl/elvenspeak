"""The entry point's one job: report a bad environment the same way, either way.

`_settings()` exists because only the script path used to catch `ConfigError`,
so `uvicorn main:build --factory` — a documented way to start this service —
answered a bad environment with a raw traceback carrying every problem joined
onto one line. Both paths now come through here, and nothing checked that they
do.

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


def test_a_bad_environment_exits_two_naming_every_problem(monkeypatch, capsys):
    """One restart, the whole list — the contract settings.py accumulates for.

    Reporting only the first problem would send an operator round the fix-restart
    loop once per mistake, which is exactly what `ConfigError` carrying a list is
    meant to prevent. So this asserts every problem reaches stderr, not just that
    the exit code is right.
    """
    monkeypatch.setenv("PIPER_VOICES", "en_US-lessac-medium")
    monkeypatch.setenv("PIPER_FALLBACK_VOICE", "not-installed")
    monkeypatch.setenv("PORT", "99999")
    monkeypatch.setenv("ELVENSPEAK_TIMESTAMPS", "maybe")

    with pytest.raises(SystemExit) as raised:
        main._settings()

    assert raised.value.code == 2
    stderr = capsys.readouterr().err
    for expected in ("PIPER_FALLBACK_VOICE", "PORT", "ELVENSPEAK_TIMESTAMPS"):
        assert expected in stderr


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
