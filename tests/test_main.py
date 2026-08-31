"""The entry point's one job: report a bad environment the same way, either way.

`settings.reported_or_exit` exists because only the script path used to catch
`ConfigError`, so `uvicorn main:build --factory` — a documented way to start
this service — answered a bad environment with a raw traceback carrying every
problem joined onto one line. Every entry point now comes through it, and what
that reporting looks like is `tests/test_settings.py`'s subject; what this file
checks is that the factory path still goes through it — including for the
problem that is not discoverable until the engine is open.

The success path does need a real voice, and that is the point of the design it
covers: `build()` is the composition root, so it opens the engine before handing
it to the app. A bad deployment therefore fails here, with an exit code, rather
than inside the first request.
"""

from __future__ import annotations

import pytest
from conftest import INSTALLED_VOICE as VOICE
from conftest import MODELS_DIR as MODELS
from fastapi import FastAPI

import main


def test_the_factory_entry_point_exits_the_same_way(monkeypatch, capsys):
    """The path that used to raise a traceback instead of reporting."""
    monkeypatch.setenv("PORT", "not-a-number")

    with pytest.raises(SystemExit) as raised:
        main.build()

    assert raised.value.code == 2
    assert "PORT" in capsys.readouterr().err


@pytest.mark.usefixtures("piper_installed")
def test_a_fallback_naming_no_offered_voice_exits_the_same_way(monkeypatch, capsys):
    """The one configuration problem that cannot be found while parsing.

    Whether the fallback names a voice the engine offers is only answerable once
    the voices are loaded, so it is checked in `Catalog` and not in
    `Settings.from_env`. For one commit that meant it left `main.build()` as an
    unhandled `ValueError` traceback while every other bad setting got a clean
    line and exit 2 — the exact divergence this module's subject exists to
    prevent, reintroduced from the other end by moving a check rather than by
    omitting a helper.

    Driven through `main.build()` rather than through `Catalog`, because the
    thing that was broken was the path, not the check: asserting on `Catalog`
    alone would have stayed green throughout.
    """
    monkeypatch.setenv("PIPER_VOICES", VOICE)
    monkeypatch.setenv("PIPER_MODELS_DIR", str(MODELS))
    monkeypatch.setenv("PIPER_ALLOW_DOWNLOAD", "0")
    monkeypatch.setenv("ELVENSPEAK_WITHHOLD", "timestamps")
    monkeypatch.setenv("ELVENSPEAK_FALLBACK_VOICE", "en_GB-nonexistent-medium")
    monkeypatch.delenv("ELVENSPEAK_ENGINE", raising=False)

    with pytest.raises(SystemExit) as raised:
        main.build()

    assert raised.value.code == 2
    assert "en_GB-nonexistent-medium" in capsys.readouterr().err


@pytest.mark.usefixtures("piper_installed")
def test_a_good_environment_builds_an_application(monkeypatch):
    """The success path, so the failure tests are not the only thing exercised.

    Against the installed voice with downloading off, because that is what a
    deployment looks like: the entry point opens the engine, so this covers the
    wiring from environment through to a server that could actually speak.
    """
    monkeypatch.setenv("PIPER_VOICES", VOICE)
    monkeypatch.setenv("PIPER_MODELS_DIR", str(MODELS))
    monkeypatch.setenv("PIPER_ALLOW_DOWNLOAD", "0")
    monkeypatch.delenv("ELVENSPEAK_ENGINE", raising=False)
    monkeypatch.delenv("ELVENSPEAK_FALLBACK_VOICE", raising=False)
    monkeypatch.delenv("ELVENSPEAK_API_KEY", raising=False)
    monkeypatch.setenv("PORT", "5001")
    monkeypatch.setenv("ELVENSPEAK_WITHHOLD", "timestamps")

    app = main.build()
    assert isinstance(app, FastAPI)
