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
from elvenspeak.engine import Capability
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
        "first": lambda _env, _withheld: DeclaredPrepared(),
        "second": lambda *_: pytest.fail("the second entry is not the default"),
    }
    settings = Settings.from_env(registry, {})
    assert isinstance(settings.engine, DeclaredPrepared)
    assert settings.engine_name == "first"


def test_the_engine_and_its_name_come_back_from_the_same_lookup():
    """[LAW:one-source-of-truth] The name has to be the key that was looked up.

    It decides which `elvenspeak/aliases/<engine>.toml` is read, and getting it
    wrong is silent by construction: `load_aliases` returns an empty table for a
    name it has no file for, so a server holding the wrong name resolves no
    aliases and says nothing about it — quieter than the bug this whole change
    exists to fix, which at least dropped its entries at INFO.

    Named against a synthetic registry rather than the real one, because the
    property is that the name is whichever key `_prepare` resolved, not which
    engines this repository happens to ship. The second entry is the one chosen,
    so a name that came from defaulting rather than from the lookup fails here.
    """
    registry: Registry = {
        "first": lambda *_: pytest.fail("an explicitly named engine was not built"),
        "second": lambda _env, _withheld: DeclaredPrepared(),
    }
    settings = Settings.from_env(registry, {"ELVENSPEAK_ENGINE": "second"})
    assert isinstance(settings.engine, DeclaredPrepared)
    assert settings.engine_name == "second"


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


#: A name no engine will ever have. It used to be `kokoro`, which stopped being
#: unknown the day a second engine was added — and these tests then passed an
#: engine that really existed and asserted it was refused, which is the shape of
#: a test that goes green by agreeing with the bug.
UNKNOWN_ENGINE = "not-an-engine"


def test_an_unknown_engine_is_refused_and_says_what_there_is():
    with pytest.raises(ConfigError) as raised:
        from_env(ELVENSPEAK_ENGINE=UNKNOWN_ENGINE)
    message = str(raised.value)
    assert UNKNOWN_ENGINE in message
    # Every registered engine, read off the registry rather than listed here: a
    # roster spelled in this file is a second map of `ENGINES` that goes stale
    # exactly when a third engine is added and nobody looks at this test.
    for name in ENGINES:
        assert name in message


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
        from_env(ELVENSPEAK_ENGINE=UNKNOWN_ENGINE, PIPER_ALLOW_DOWNLOAD="tru")
    assert raised.value.problems == [
        f"ELVENSPEAK_ENGINE={UNKNOWN_ENGINE!r} is not one of: "
        f"{', '.join(ENGINES)}"
    ]


def test_nothing_is_withheld_by_default():
    assert from_env().withheld == frozenset()


@pytest.mark.parametrize(
    "value,expected",
    [
        ("timestamps", {Capability.TIMESTAMPS}),
        ("TIMESTAMPS", {Capability.TIMESTAMPS}),
        (" speed , timestamps ", {Capability.SPEED, Capability.TIMESTAMPS}),
        ("", set()),
    ],
)
def test_withheld_capabilities_are_split_stripped_and_read_in_any_case(
    value, expected
):
    """The spelling is the capability's own name, which is what the log prints.

    Present-but-blank is genuinely "withhold nothing" here, unlike
    `ELVENSPEAK_ENGINE=`: an empty list has an obvious meaning and an empty
    engine name does not, so there is nothing for an operator to have meant
    instead.
    """
    assert from_env(ELVENSPEAK_WITHHOLD=value).withheld == frozenset(expected)


def test_a_name_that_is_not_a_capability_is_refused_rather_than_skipped():
    """[LAW:no-silent-failure] A typo that withheld nothing is the original bug.

    An operator who wrote `timestmaps` meant to switch timestamps off, and a
    parse that quietly dropped the word would answer them with the timestamps
    they asked not to have — the same silent disagreement `ELVENSPEAK_TIMESTAMPS`
    produced, reached by a different route.
    """
    with pytest.raises(ConfigError) as raised:
        from_env(ELVENSPEAK_WITHHOLD="timestmaps")
    message = str(raised.value)
    assert "timestmaps" in message
    # The real names, read off the enum rather than listed here, so a capability
    # added to it cannot leave this message describing a vocabulary that moved.
    for capability in Capability:
        assert capability.name.lower() in message


def test_withholding_something_the_engine_never_had_is_not_an_error():
    """Reachable by ordinary deployment, so it must not be a case to remember.

    A Kokoro export with no `duration` output declares no timestamps; a
    deployment that switched them off besides has said the same thing twice.
    Subtracting from a set is what makes that free.
    """
    assert from_env(ELVENSPEAK_WITHHOLD="timestamps,speed").withheld == frozenset(
        Capability
    )


def test_the_setting_reaches_the_engine_so_it_can_decline_to_build_the_machinery():
    """Enforcement is the server's; the message is the economy.

    Piper patches its ONNX graph at load time to expose durations, so a withheld
    `TIMESTAMPS` only saves anything if the engine hears about it before it opens
    a session. That is why this crosses `Configure` rather than being applied to
    the engine's answer alone — which the server also does, in `create_app`.
    """
    assert from_env(ELVENSPEAK_WITHHOLD="timestamps").engine.timings is False
    assert from_env().engine.timings is True


def test_the_retired_name_is_refused_rather_than_ignored():
    """[LAW:no-silent-failure] The defect this setting exists to end, on the way out.

    `ELVENSPEAK_TIMESTAMPS` was read by Piper alone, so a deployment that set it
    and ran another engine got timestamps anyway. Merely no longer reading it
    would reproduce that exactly — the same operator, the same file, the same
    wrong answer — so the name is a startup failure that names its replacement.
    """
    with pytest.raises(ConfigError) as raised:
        from_env(ELVENSPEAK_TIMESTAMPS="0")
    message = str(raised.value)
    assert "ELVENSPEAK_TIMESTAMPS" in message
    assert "ELVENSPEAK_WITHHOLD=timestamps" in message


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
    monkeypatch.setenv("ELVENSPEAK_WITHHOLD", "timestmaps")

    with pytest.raises(SystemExit) as raised, reported_or_exit():
        Settings.from_env(ENGINES)

    assert raised.value.code == 2
    stderr = capsys.readouterr().err
    for expected in ("PORT", "ELVENSPEAK_WITHHOLD"):
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
    assert settings.engine == piper.configure({}, frozenset())
