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
from elvenspeak import memory as main_memory
from elvenspeak.engines import ENGINES
from elvenspeak.settings import Settings, unsized


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


@pytest.mark.usefixtures("piper_installed")
def test_a_confined_deployment_that_named_no_ceiling_exits_rather_than_serving(
    monkeypatch, capsys
):
    """The refusal reaches an operator through the same door every other one does.

    Driven through `main.build()` rather than through `settings.unsized`, because
    what is being checked is the WIRING: the check lives at the composition root
    precisely so the bake step does not inherit it, and a check placed correctly
    but never called is exactly as useless as no check. `tests/test_settings.py`
    covers what it decides; this covers that anything asks it.

    The confinement is stated rather than read, since a developer machine is
    unconfined and this test would otherwise pass by never reaching the code.
    """
    monkeypatch.setenv("PIPER_VOICES", VOICE)
    monkeypatch.setenv("PIPER_MODELS_DIR", str(MODELS))
    monkeypatch.setenv("PIPER_ALLOW_DOWNLOAD", "0")
    monkeypatch.delenv("ELVENSPEAK_ENGINE", raising=False)
    monkeypatch.delenv("ELVENSPEAK_CONCURRENT_SYNTHESES", raising=False)
    monkeypatch.setattr(main_memory, "limit", lambda *_: 2048 * 1048576)

    with pytest.raises(SystemExit) as raised:
        main.build()

    assert raised.value.code == 2
    assert "ELVENSPEAK_CONCURRENT_SYNTHESES" in capsys.readouterr().err


@pytest.mark.usefixtures("piper_installed")
def test_the_refusal_does_not_reach_the_bake_step(monkeypatch):
    """The reason the check is not in the shared parse.

    `python -m elvenspeak.bake` runs inside the image build and synthesizes
    nothing, so a synthesis ceiling is not its concern. If this refusal sat on
    `Settings.from_env`, a memory-limited builder would fail every image build --
    and while CI now runs each image before pushing it (piper-build-b4h.2), it
    runs it unconfined, so a refusal that fires only under a ceiling is still
    invisible there. The first evidence would be a publish that spent a dated tag
    on an image that cannot boot.

    The confinement is patched over conftest's `_unconfined` fixture BEFORE the
    parse, which is the whole of what makes this a regression test. Left to the
    fixture, a refusal moved inside `from_env` would read UNCONFINED, never fire,
    and this test would stay green through precisely the change it exists to
    catch.

    Both halves are then asserted on one `Settings`: the parse yields a usable
    value under that confinement, and the server-side question refuses that very
    value under that very limit — so the confinement cannot be a number the test
    merely mentions.
    """
    monkeypatch.setenv("PIPER_VOICES", VOICE)
    monkeypatch.setenv("PIPER_MODELS_DIR", str(MODELS))
    monkeypatch.setenv("PIPER_ALLOW_DOWNLOAD", "0")
    monkeypatch.delenv("ELVENSPEAK_ENGINE", raising=False)
    monkeypatch.delenv("ELVENSPEAK_CONCURRENT_SYNTHESES", raising=False)

    confined = 2048 * 1048576
    monkeypatch.setattr(main_memory, "limit", lambda *_: confined)

    settings = Settings.from_env(ENGINES)

    assert unsized(settings, confined) is not None, (
        "the confinement this test states must actually refuse a server, or the "
        "assertion below proves nothing about the parse being exempt from it"
    )
    assert settings.engine is not None
