"""Configuration parsing, and the promise that it reports everything at once.

`Settings.from_env` takes a `dict[str, str]` and returns either a `Settings` or a
`ConfigError` — pure, and testable with no filesystem, no model and no server.
"""

from __future__ import annotations

import pytest

from elvenspeak.settings import DEFAULT_VOICE, ConfigError, Settings


def env(**overrides) -> dict[str, str]:
    """A minimal valid environment, with the test's changes applied."""
    base = {"PIPER_VOICES": "en_US-lessac-medium"}
    base.update({k: v for k, v in overrides.items() if v is not None})
    return base


def test_defaults_are_usable_with_one_voice_named():
    settings = Settings.from_env(env())
    assert settings.voices == ("en_US-lessac-medium",)
    assert settings.fallback == "en_US-lessac-medium"
    assert settings.port == 5001
    assert settings.api_key is None


def test_voices_are_split_and_stripped():
    settings = Settings.from_env(env(PIPER_VOICES="a-b-c , d-e-f,  g-h-i "))
    assert settings.voices == ("a-b-c", "d-e-f", "g-h-i")


def test_fallback_is_stripped_before_the_membership_check():
    """A trailing space from a .env file used to report a present voice missing."""
    settings = Settings.from_env(
        env(PIPER_VOICES="en_US-lessac-medium", PIPER_FALLBACK_VOICE="en_US-lessac-medium ")
    )
    assert settings.fallback == "en_US-lessac-medium"


def test_empty_fallback_disables_substitution():
    assert Settings.from_env(env(PIPER_FALLBACK_VOICE="")).fallback is None


def test_fallback_outside_the_voice_list_is_refused():
    with pytest.raises(ConfigError) as raised:
        Settings.from_env(env(PIPER_FALLBACK_VOICE="en_GB-alba-medium"))
    assert "PIPER_FALLBACK_VOICE" in str(raised.value)


def test_empty_voice_list_is_refused():
    with pytest.raises(ConfigError):
        Settings.from_env({"PIPER_VOICES": "  ,  "})


@pytest.mark.parametrize("port", ["0", "-1", "65536", "99999"])
def test_out_of_range_port_is_refused(port):
    """Parsing is not validating — these are all integers and none is a port."""
    with pytest.raises(ConfigError) as raised:
        Settings.from_env(env(PORT=port))
    assert "PORT" in str(raised.value)


def test_non_numeric_port_is_refused():
    with pytest.raises(ConfigError) as raised:
        Settings.from_env(env(PORT="eighty"))
    assert "not a number" in str(raised.value)


@pytest.mark.parametrize("value,expected", [
    ("1", True), ("true", True), ("YES", True), ("on", True),
    ("0", False), ("false", False), ("No", False), ("off", False),
])
def test_flags_accept_both_spellings(value, expected):
    assert Settings.from_env(env(PIPER_ALLOW_DOWNLOAD=value)).allow_download is expected


@pytest.mark.parametrize("typo", ["tru", "enabled", "y", ""])
def test_a_boolean_typo_is_a_config_error_not_a_silent_false(typo):
    """[LAW:no-silent-failure] `PIPER_ALLOW_DOWNLOAD=tru` used to quietly mean off.

    In the one module whose stated job is catching configuration mistakes at
    startup, a typo in a boolean is exactly as much a mistake as a typo in a port.
    """
    with pytest.raises(ConfigError) as raised:
        Settings.from_env(env(PIPER_ALLOW_DOWNLOAD=typo))
    assert "PIPER_ALLOW_DOWNLOAD" in str(raised.value)


def test_every_problem_is_reported_together():
    """The module's whole reason for existing: one restart, the whole list.

    An operator who fixes one problem per restart is being served worse than one
    who is handed all of them at once, which is why these accumulate rather than
    raising on the first.
    """
    with pytest.raises(ConfigError) as raised:
        Settings.from_env({
            "PIPER_VOICES": "en_US-lessac-medium",
            "PIPER_FALLBACK_VOICE": "not-installed",
            "PORT": "99999",
            "ELVENSPEAK_TIMESTAMPS": "maybe",
        })
    problems = raised.value.problems
    assert len(problems) == 3
    joined = " ".join(problems)
    assert "PIPER_FALLBACK_VOICE" in joined
    assert "PORT" in joined
    assert "ELVENSPEAK_TIMESTAMPS" in joined


def test_an_empty_voice_list_does_not_hide_a_bad_fallback():
    """Both problems, from one pass, when the two coincide.

    The membership check used to be guarded by `voices and`, so an empty
    PIPER_VOICES suppressed it — the operator fixed the empty list, restarted,
    and only then learned the fallback was wrong too. That is precisely the
    one-problem-per-restart loop this module exists to prevent, and it appeared
    only in the case where the operator was already misconfigured twice.
    """
    with pytest.raises(ConfigError) as raised:
        Settings.from_env({
            "PIPER_VOICES": "  ,  ",
            "PIPER_FALLBACK_VOICE": "en_US-lessac-medium",
        })
    problems = raised.value.problems
    assert len(problems) == 2
    joined = " ".join(problems)
    assert "PIPER_VOICES is empty" in joined
    assert "PIPER_FALLBACK_VOICE" in joined


def test_an_empty_models_dir_is_refused_not_taken_as_the_working_directory():
    """The one setting here that was not validated.

    `PIPER_MODELS_DIR=` is a present key, so `get` returns "" rather than the
    default, `Path("")` is `Path(".")`, and `mkdir` on it succeeds. Nothing
    fails — the server just reads and writes 60 MB models into whatever
    directory it was launched from, which is the silent wrong thing rather than
    the clean refusal this module produces for every other misconfiguration.
    """
    with pytest.raises(ConfigError) as raised:
        Settings.from_env({
            "PIPER_VOICES": "en_US-lessac-medium",
            "PIPER_MODELS_DIR": "   ",
        })
    assert any("PIPER_MODELS_DIR" in problem for problem in raised.value.problems)


def test_an_absent_models_dir_still_gets_the_default():
    """Unset is not the same as set-to-empty, and only one of them is a problem."""
    settings = Settings.from_env({"PIPER_VOICES": "en_US-lessac-medium"})
    assert settings.models_dir.name == "models"


def test_a_bad_environment_exits_two_naming_every_problem(monkeypatch, capsys):
    """One restart, the whole list — the contract `ConfigError` accumulates for.

    Reporting only the first problem would send an operator round the
    fix-restart loop once per mistake, which is exactly what a `ConfigError`
    carrying a list is meant to prevent. So this asserts every problem reaches
    stderr, not just that the exit code is right.

    Reads the real environment, unlike everything above it, because
    `from_env_or_exit` is what the entry points call with no argument — it is
    the process-facing half of this module, and stubbing the environment out of
    it would leave the half that actually runs untested.
    """
    monkeypatch.setenv("PIPER_VOICES", "en_US-lessac-medium")
    monkeypatch.setenv("PIPER_FALLBACK_VOICE", "not-installed")
    monkeypatch.setenv("PORT", "99999")
    monkeypatch.setenv("ELVENSPEAK_TIMESTAMPS", "maybe")

    with pytest.raises(SystemExit) as raised:
        Settings.from_env_or_exit()

    assert raised.value.code == 2
    stderr = capsys.readouterr().err
    for expected in ("PIPER_FALLBACK_VOICE", "PORT", "ELVENSPEAK_TIMESTAMPS"):
        assert expected in stderr


def test_a_good_environment_comes_back_as_settings(monkeypatch):
    """The positive control: the exit is not the only way out of it.

    Every variable this module reads is cleared first, so the result is the
    documented defaults rather than whatever the shell running the tests
    happens to export.
    """
    for name in (
        "PIPER_VOICES",
        "PIPER_FALLBACK_VOICE",
        "PIPER_MODELS_DIR",
        "PIPER_ALLOW_DOWNLOAD",
        "ELVENSPEAK_API_KEY",
        "ELVENSPEAK_TIMESTAMPS",
        "HOST",
        "PORT",
    ):
        monkeypatch.delenv(name, raising=False)

    assert Settings.from_env_or_exit().voices == (DEFAULT_VOICE,)
