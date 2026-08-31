"""Configuration parsing, and the promise that it reports everything at once.

`Settings.from_env` takes a registry and a `dict[str, str]` and returns either a
`Settings` or a `ConfigError` — pure, and testable with no filesystem, no model
and no server. That stays true with a real engine in the registry, because
`piper.configure` is a parse and does no I/O either.

What this file no longer covers, and where it went: `PIPER_*` belongs to the
engine that reads it and is `tests/test_piper.py`'s subject, and whether a
fallback names an installed voice is checked against the voices an engine really
loaded, in `tests/test_voices.py`. Neither moved because it was awkward here —
each had a second home that already knew more than this module could.
"""

from __future__ import annotations

import pytest
from conftest import DeclaredPrepared

from elvenspeak import piper
from elvenspeak.engines import ENGINES
from elvenspeak.piper import DEFAULT_VOICE
from elvenspeak.provisioning import ConfigError, Registry
from elvenspeak.settings import Settings, reported_or_exit
from elvenspeak.voices import Substitution


def env(**overrides) -> dict[str, str]:
    """A minimal valid environment, with the test's changes applied."""
    base = {"PIPER_VOICES": DEFAULT_VOICE}
    base.update({k: v for k, v in overrides.items() if v is not None})
    return base


def from_env(**overrides) -> Settings:
    return Settings.from_env(ENGINES, env(**overrides))


def test_defaults_are_usable_with_one_voice_named():
    settings = from_env()
    assert settings.fallback is Substitution.FIRST_OFFERED
    assert settings.port == 5001
    assert settings.api_key is None


def test_the_unnamed_engine_is_the_registry_s_first_entry():
    """[LAW:one-source-of-truth] The default is a position, not a second literal.

    Two entries, because a one-engine registry cannot tell "the first" from "the
    only" — and the property that matters is that no name is spelled anywhere
    outside the registry that could come to name an engine that is not in it.
    """
    registry: Registry = {
        "first": lambda _: DeclaredPrepared(),
        "second": lambda _: pytest.fail("the second entry is not the default"),
    }
    assert isinstance(Settings.from_env(registry, {}).engine, DeclaredPrepared)


def test_a_blank_engine_name_is_refused_rather_than_taken_as_no_preference():
    """`ELVENSPEAK_ENGINE=` is a present key, not an absent one.

    An unset variable interpolated into a compose file is the ordinary way to
    arrive at one, and the operator who got there was trying to select an engine.
    Reading it as "no preference" boots the default instead — which, once a
    second engine exists, is the wrong engine, silently, with nothing reported.

    The same distinction `PIPER_MODELS_DIR` already makes, made here for the same
    reason ([LAW:no-silent-failure]).
    """
    with pytest.raises(ConfigError, match="ELVENSPEAK_ENGINE is empty"):
        from_env(ELVENSPEAK_ENGINE="")
    with pytest.raises(ConfigError, match="ELVENSPEAK_ENGINE is empty"):
        from_env(ELVENSPEAK_ENGINE="   ")


def test_an_empty_registry_is_a_config_error_not_a_stopiteration():
    """The one way out of this module that `reported_or_exit` could not catch.

    `next(iter(engines))` on an empty mapping raises `StopIteration`, which is
    not a `ConfigError` and so becomes an unhandled traceback rather than a
    clean exit 2. Nothing here registers an empty registry, but `Registry` is a
    plain mapping a caller supplies and the type cannot say it is non-empty —
    and the unknown-name message a few lines below already anticipated this
    case, so it was considered on one path and not the other.
    """
    with pytest.raises(ConfigError, match="no engines registered"):
        Settings.from_env({}, {})


def test_an_unknown_engine_is_refused_and_says_what_there_is():
    with pytest.raises(ConfigError) as raised:
        from_env(ELVENSPEAK_ENGINE="kokoro")
    assert "kokoro" in str(raised.value)
    assert "piper" in str(raised.value)


def test_an_engine_s_problems_join_the_server_s_in_one_list():
    """The splice, which is the whole reason engine settings could move away.

    Separating what the server reads from what the engine reads was only safe if
    it did not separate the report: an operator with a bad port and a bad Piper
    flag must be told both on the first run, exactly as when one module read
    both. This is the test that would go red if `from_env` ever raised the
    server's problems before asking the engine for its own.
    """
    with pytest.raises(ConfigError) as raised:
        from_env(PORT="99999", PIPER_ALLOW_DOWNLOAD="tru")
    problems = raised.value.problems
    assert len(problems) == 2
    joined = " ".join(problems)
    assert "PORT" in joined
    assert "PIPER_ALLOW_DOWNLOAD" in joined


def test_an_unknown_engine_reports_only_that():
    """No configure to call means no configuration to complain about.

    Reporting `PIPER_ALLOW_DOWNLOAD` here would mean guessing that the engine the
    operator misnamed was the one whose variables to check, which is a fact this
    module does not have ([LAW:no-silent-failure] — better one true problem than
    a second invented one).
    """
    with pytest.raises(ConfigError) as raised:
        from_env(ELVENSPEAK_ENGINE="kokoro", PIPER_ALLOW_DOWNLOAD="tru")
    assert raised.value.problems == [
        "ELVENSPEAK_ENGINE='kokoro' is not one of: piper"
    ]


def test_an_unset_fallback_defers_the_choice_to_the_voices_that_load():
    """[LAW:types-are-the-program] Unset and switched-off stop sharing a value.

    While the voice list lived here, "the obvious voice" was resolved during
    parsing and only two states survived. The list belongs to the engine now, so
    an unset variable has to travel as its own answer — and if it were spelled
    `None` like the disabled case, an operator who cleared the variable to turn
    substitution off would silently get a voice instead.
    """
    assert from_env().fallback is Substitution.FIRST_OFFERED
    assert from_env(ELVENSPEAK_FALLBACK_VOICE="").fallback is Substitution.OFF
    assert from_env(ELVENSPEAK_FALLBACK_VOICE="  ").fallback is Substitution.OFF


def test_a_named_fallback_is_stripped():
    """A trailing space from a .env file used to report a present voice missing."""
    assert from_env(ELVENSPEAK_FALLBACK_VOICE=" a-b-c ").fallback == "a-b-c"


@pytest.mark.parametrize("port", ["0", "-1", "65536", "99999"])
def test_out_of_range_port_is_refused(port):
    """Parsing is not validating — these are all integers and none is a port."""
    with pytest.raises(ConfigError) as raised:
        from_env(PORT=port)
    assert "PORT" in str(raised.value)


def test_non_numeric_port_is_refused():
    with pytest.raises(ConfigError) as raised:
        from_env(PORT="eighty")
    assert "not a number" in str(raised.value)


def test_a_bad_environment_exits_two_naming_every_problem(monkeypatch, capsys):
    """One restart, the whole list — the contract `ConfigError` accumulates for.

    Reporting only the first problem would send an operator round the
    fix-restart loop once per mistake, which is exactly what a `ConfigError`
    carrying a list is meant to prevent. So this asserts every problem reaches
    stderr, not just that the exit code is right.

    Reads the real environment, unlike everything above it, because
    `reported_or_exit` is what the entry points wrap around a real startup — it
    is the process-facing half of this module, and stubbing the environment out
    of it would leave the half that actually runs untested.
    """
    monkeypatch.setenv("PORT", "99999")
    monkeypatch.setenv("ELVENSPEAK_TIMESTAMPS", "maybe")

    with pytest.raises(SystemExit) as raised, reported_or_exit():
        Settings.from_env(ENGINES)

    assert raised.value.code == 2
    stderr = capsys.readouterr().err
    for expected in ("PORT", "ELVENSPEAK_TIMESTAMPS"):
        assert expected in stderr


def test_a_good_environment_comes_back_as_settings(clean_env):
    """The positive control: the exit is not the only way out of it.

    Against an environment holding none of what a startup reads, so the result is
    the documented defaults rather than whatever the shell running the tests
    happens to export.
    """
    with reported_or_exit():
        settings = Settings.from_env(ENGINES)
    assert settings.fallback is Substitution.FIRST_OFFERED
    # The engine an empty environment produces, compared against the engine an
    # empty environment produces — the point being that `from_env` chose Piper
    # and handed it the same environment, rather than that Piper's own defaults
    # are what they are, which is `test_piper.py`'s business.
    assert settings.engine == piper.configure({})
