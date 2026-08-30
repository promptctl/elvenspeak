"""Configuration parsing, and the promise that it reports everything at once.

`Settings.from_env` takes a `dict[str, str]` and returns either a `Settings` or a
`ConfigError` — pure, and testable with no filesystem, no model and no server.
"""

from __future__ import annotations

import pytest

from elvenspeak.settings import ConfigError, Settings


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
