"""The entry point's one job: report a bad environment the same way, either way.

`_settings()` exists because only the script path used to catch `ConfigError`,
so `uvicorn main:build --factory` — a documented way to start this service —
answered a bad environment with a raw traceback carrying every problem joined
onto one line. Both paths now come through here, and nothing checked that they
do.

Needs no voice model: `create_app` only builds the application, and voices are
installed by the lifespan, which does not run until startup.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI

import main


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


def test_a_good_environment_builds_an_application(monkeypatch, tmp_path):
    """The success path, so the failure tests are not the only thing exercised."""
    monkeypatch.setenv("PIPER_VOICES", "en_US-lessac-medium")
    monkeypatch.setenv("PIPER_MODELS_DIR", str(tmp_path))
    monkeypatch.delenv("PIPER_FALLBACK_VOICE", raising=False)
    monkeypatch.delenv("ELVENSPEAK_API_KEY", raising=False)
    monkeypatch.setenv("PORT", "5001")
    monkeypatch.setenv("ELVENSPEAK_TIMESTAMPS", "0")

    app = main.build()
    assert isinstance(app, FastAPI)
