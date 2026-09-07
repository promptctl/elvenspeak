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

import re
from pathlib import Path

import asyncio
import threading
import time

import pytest
from conftest import _ENVIRONMENT, DeclaredPrepared, serves

from elvenspeak import piper
from elvenspeak.engine import Capability
from elvenspeak.engines import ENGINES
from elvenspeak.piper import DEFAULT_VOICE
from elvenspeak.provisioning import ConfigError, Registry
from elvenspeak.memory import Unconfined
from elvenspeak.settings import (
    CONCURRENT_SYNTHESES,
    MEASURED_PIPER,
    Settings,
    reported_or_exit,
    unsized,
)
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
        "first": lambda _env, _withheld, _serves: DeclaredPrepared(),
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
        "second": lambda _env, _withheld, _serves: DeclaredPrepared(),
    }
    settings = Settings.from_env(registry, {"ELVENSPEAK_ENGINE": "second"})
    assert isinstance(settings.engine, DeclaredPrepared)
    assert settings.engine_name == "second"


def test_the_roster_is_the_registry_that_was_handed_in():
    """[LAW:one-source-of-truth] Which engines exist is asked of the registry.

    `known_engines` is what lets a deployment refuse a `model_id` naming an
    engine it is not running instead of answering in the one it is
    ([`elvenspeak.models.Reach.ELSEWHERE`]). Read off the registry the caller
    supplied, because a list spelled anywhere else is a second answer to "which
    engines exist" — and the direction it fails in is the bad one: an engine
    missing from it goes back to being served by the wrong engine, silently.

    Against a synthetic registry for the same reason the test above uses one: the
    property is that the roster is whatever was handed in, not that this
    repository happens to ship two engines today.
    """
    registry: Registry = {
        "first": lambda _env, _withheld, _serves: DeclaredPrepared(),
        "second": lambda *_: pytest.fail("the unnamed engine is the first entry"),
    }
    settings = Settings.from_env(registry, {})

    assert settings.known_engines == {"first", "second"}
    assert settings.engine_name in settings.known_engines


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


@pytest.mark.parametrize("at_once", ["0", "-1"])
def test_a_concurrency_below_one_is_refused(at_once):
    """Zero is not "no limit" here, and refusing it is what keeps that true.

    A gate that disappears at zero would be a second mode of the whole synthesis
    path — bounded and unbounded — reachable by one character in an environment
    file, and the unbounded one is the shape that OOM-killed elvenspeak-piper
    (piper-memory-9rc). A deployment that wants no practical bound says so with a
    number ([LAW:no-mode-explosion]).
    """
    with pytest.raises(ConfigError) as raised:
        from_env(ELVENSPEAK_CONCURRENT_SYNTHESES=at_once)
    assert "at least 1" in str(raised.value)


def test_a_non_numeric_concurrency_is_refused():
    with pytest.raises(ConfigError) as raised:
        from_env(ELVENSPEAK_CONCURRENT_SYNTHESES="lots")
    assert "not a number" in str(raised.value)


def test_the_concurrency_default_is_the_bound_that_was_already_there():
    """[LAW:one-source-of-truth] Naming the ceiling must not move it.

    Every synthesis dispatches through `asyncio.to_thread`, so the default
    executor's width was already the limit on concurrent synthesis — set by the
    host rather than by anyone's decision. The default here has to be that same
    number, or switching this setting on silently changes a deployment's capacity
    while claiming to change nothing.

    MEASURED, NOT RESTATED, and that is the whole design of this test. It used to
    assert against a second copy of CPython's formula, spelled `os.cpu_count()` —
    which is not the formula 3.13's `ThreadPoolExecutor` uses. A copy cannot
    detect its own drift, so the check written to enforce the invariant was the
    one thing structurally unable to see it break.

    Saturating the executor and counting asks the question of the thing itself,
    so the next time CPython changes this a test fails rather than a deployment
    quietly getting a ceiling nobody chose.
    """

    async def widest() -> int:
        peak = 0
        running = 0
        counting = threading.Lock()

        def occupy() -> None:
            nonlocal peak, running
            with counting:
                running += 1
                peak = max(peak, running)
            # Long enough that every worker the executor will create is still
            # inside this sleep when the last of the first batch is dispatched.
            # `_adjust_thread_count` reuses an idle thread rather than spawning
            # another, so a sleep short enough for the first worker to finish
            # early would undercount the width and fail this test with a message
            # about the default — on a loaded runner, which this suite has
            # already met once at load average 104.
            time.sleep(0.5)
            with counting:
                running -= 1

        # Comfortably more than any width `min(32, ...)` can produce, submitted
        # in one loop iteration so the executor is saturated rather than sampled.
        await asyncio.gather(*(asyncio.to_thread(occupy) for _ in range(80)))
        return peak

    assert from_env().concurrent_syntheses == asyncio.run(widest())


@pytest.mark.parametrize("unset", ["", "   "])
def test_an_empty_concurrency_reads_as_unset(unset):
    """The spelling the README documents must not crashloop the service.

    `ELVENSPEAK_CONCURRENT_SYNTHESES=` with nothing after it is how the README
    lists it and how a Nomad `env { NAME = "" }` arrives. Every other optional
    variable in that block already tolerates empty as unset; a number that alone
    refused it would fail a deployment copied from the documentation.
    """
    assert (
        from_env(ELVENSPEAK_CONCURRENT_SYNTHESES=unset).concurrent_syntheses
        == from_env().concurrent_syntheses
    )


def test_a_chosen_concurrency_is_what_arrives():
    assert from_env(ELVENSPEAK_CONCURRENT_SYNTHESES="3").concurrent_syntheses == 3


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
    assert settings.engine == piper.configure({}, frozenset(), serves("piper"))


def _settings_read_from_the_environment() -> dict[str, str]:
    """Every variable name the package reads out of an environment, by module.

    Found rather than listed. `env.get("NAME")` and `flag(env, "NAME", …)` are the
    only two shapes a setting is read through, and a name reached through either
    is a name a startup answers for.

    `provisioning.flag`'s own `env.get(name)` resolves to a parameter and is
    skipped, which is correct: that function reads whichever name its caller
    passed, and those callers are found here individually.
    """
    import ast

    import elvenspeak

    found: dict[str, str] = {}
    for path in sorted(Path(elvenspeak.__file__).parent.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Module-level `NAME = "VALUE"`, so a setting exported as a constant —
        # `router.CONSUL_URL` — is resolved to the variable it names.
        constants = {
            target.id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
            for target in node.targets
            if isinstance(target, ast.Name) and isinstance(node.value.value, str)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            reads_env = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "env"
                and node.args
            )
            reads_flag = (
                isinstance(node.func, ast.Name)
                and node.func.id == "flag"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "env"
            )
            named = (
                node.args[0] if reads_env else node.args[1] if reads_flag else None
            )
            if isinstance(named, ast.Constant) and isinstance(named.value, str):
                found[named.value] = path.name
            elif isinstance(named, ast.Name) and named.id in constants:
                found[constants[named.id]] = path.name
    return found


def test_the_clean_environment_clears_every_setting_a_startup_reads():
    """[FRAMING:representation] A map the machine redraws, not one somebody remembers.

    `conftest._ENVIRONMENT` is what `clean_env` strips before a test parses an
    environment, and its own docstring predicts what happens when it falls behind:
    "the test that forgot goes flaky later with nothing pointing at the cause". It
    then fell behind twice while the router was being added — once for each of that
    engine's two settings — which is the law's point exactly. Two representations
    that can diverge do not merely risk it.

    One-directional on purpose. A name in the list that nothing reads is fine and
    deliberate: the docstring keeps retired names like `ELVENSPEAK_TIMESTAMPS` so a
    shell still exporting one cannot leak into a test either. What must never
    happen is a setting that is read and not cleared.
    """
    read = _settings_read_from_the_environment()
    missing = {name: module for name, module in read.items() if name not in _ENVIRONMENT}

    assert not missing, (
        "these are read at startup but not cleared by `clean_env`: "
        + ", ".join(f"{name} ({module})" for name, module in sorted(missing.items()))
    )


# ------------------------------------------ the ceiling a confined deployment owes

# `unsized` is asked at the composition root and not during the parse, so these
# drive it directly and state the confinement rather than inheriting the machine's.
# A developer laptop is unconfined, so a test that read the real cgroup would pass
# here and prove nothing about the deployment this exists for.

CONFINED = 2048 * 1048576


def test_a_confined_deployment_that_named_no_ceiling_is_refused():
    """[LAW:no-silent-failure] The configuration that OOM-killed piper four times.

    A memory limit plus a ceiling inherited from the host's core count. Nothing
    about it looks wrong until the burst arrives, and then it is a SIGKILL with
    no traceback.
    """
    refusal = unsized(from_env(), CONFINED)
    assert refusal is not None
    assert CONCURRENT_SYNTHESES in refusal
    assert "2048 MiB" in refusal


def test_the_refusal_names_the_width_it_would_have_run_at():
    """A refusal an operator cannot act on is a crashloop with better prose.

    Asserted on the exact phrase rather than on the bare number, because the bare
    number collides: the message carries 1140, 1164, 1335, 1773 and 2048, so on a
    4-core host (the gpu node) the default of 8 is a substring of "2048", and on a
    3-core host 7 is a substring of "1773". Either way the assertion would hold
    with the interpolation deleted from the f-string entirely.
    """
    settings = from_env()
    refusal = unsized(settings, CONFINED)
    assert f"({settings.concurrent_syntheses} here)" in refusal
    # The measured curve, so the number to choose is in the message rather than
    # in a README the operator is not reading at 3am.
    assert "1335" in refusal


def test_the_refusal_converts_the_limit_it_was_actually_handed():
    """A confinement that collides with nothing already in the message.

    2048 is the historical incident figure, hardcoded in the narrative sentence
    every refusal carries -- so a test asserting "2048 MiB" against a 2048 MiB
    confinement passes on that sentence alone, even if the byte-to-MiB conversion
    divided by the wrong constant or read the wrong variable. At 3072 the figure
    can only come from the conversion.
    """
    refusal = unsized(from_env(), 3072 * 1048576)
    assert refusal is not None
    assert "3072 MiB" in refusal


def test_a_confined_deployment_that_chose_a_ceiling_serves():
    assert unsized(from_env(**{CONCURRENT_SYNTHESES: "4"}), CONFINED) is None


def test_choosing_exactly_what_the_default_would_have_been_still_counts_as_choosing():
    """Provenance cannot be recovered from the value, which is why it is carried.

    An operator who works out the right number and finds it equals the core-derived
    default has still chosen it, and refusing them would be the check calling a
    correct deployment broken.
    """
    settings = from_env(**{CONCURRENT_SYNTHESES: str(from_env().concurrent_syntheses)})
    assert settings.concurrency_chosen
    assert unsized(settings, CONFINED) is None


def test_an_unconfined_process_is_untouched():
    """Every development machine, and every `docker run` without --memory."""
    assert unsized(from_env(), Unconfined.UNCONFINED) is None


def test_an_unset_variable_is_read_as_having_chosen_nothing():
    assert not from_env().concurrency_chosen


def test_the_field_defaults_to_unchosen():
    """The conservative reading, pinned at the field rather than at one caller.

    A `Settings` constructed in code did not come from an operator, so it cannot
    have chosen. Defaulting the other way would make this check silently
    inapplicable to every deployment that builds its settings directly — the
    failure would be that nothing is ever refused, which looks exactly like
    nothing being wrong.
    """
    assert Settings.__dataclass_fields__["concurrency_chosen"].default is False


README = Path(__file__).parent.parent / "README.md"


def test_the_readme_quotes_the_measured_curve_this_module_holds():
    """[LAW:one-source-of-truth] One measurement, three readers.

    The curve is quoted by the field comment, by the refusal an operator reads at
    boot, and by the README paragraph they read beforehand. An operator picks a
    ceiling off whichever they happen to see, so the three disagreeing is the
    failure -- and it is silent, because every copy looks authoritative.

    Read off README.md rather than off a rendered copy, for the reason
    `tests/test_packaging.py` reads pyproject.toml: the file is what a reader
    opens.
    """
    readme = README.read_text()
    for _, mib in MEASURED_PIPER:
        assert str(mib) in readme, (
            f"{mib} MiB is in MEASURED_PIPER but appears nowhere in README.md; "
            f"the measurement and the prose describing it have drifted"
        )


def test_the_readme_quotes_no_figure_the_measurement_does_not_have():
    """The direction that catches a number left behind by an edit.

    Scoped to the paragraph documenting this variable, and allowing 2048 -- the
    limit it died against, which is a fact about the node rather than a row of
    the curve.
    """
    readme = README.read_text()
    start = readme.index(CONCURRENT_SYNTHESES)
    paragraph = readme[start : readme.index("PORT=5001", start)]
    allowed = {str(mib) for _, mib in MEASURED_PIPER} | {"2048", "32"}
    cited = set(re.findall(r"\b\d{4}\b", paragraph))
    assert cited <= allowed, (
        f"README cites {sorted(cited - allowed)} where the measurement has "
        f"{sorted(allowed)}; a figure was changed in one place only"
    )
