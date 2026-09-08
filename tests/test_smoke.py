"""The smoke script's parsers, held against output real runtimes actually printed.

`smoke.py` proves an image serves by running it, so almost all of it is only
provable with a container runtime — which a CI runner executing this suite does
not have, and which no unit test should require. What *is* provable here is the
part where the mistakes live: reading an image's declared port and healthcheck
out of two runtimes that spell an image config differently.

Every fixture string below was captured from a real `image inspect`, not
composed to match the parser. That distinction is the whole value of the file:
a fixture invented alongside the regex it feeds can only confirm that the regex
matches itself, which is the failure `tests/test_dockerfile.py` was written
about — a check that shares an assumption with the thing it checks.

The two runtimes are two maps of one territory, so the sharpest test here is not
either parser alone but that they agree: `test_both_runtimes_read_one_image_the
_same_way` describes a single image in both spellings and requires one answer.
"""

from __future__ import annotations

import ast
import http.server
import json
import pathlib
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from unittest import mock

import fleetstub
import pytest
import smoke
from conftest import SERVES
from fleet import consul_app, routed, serving

from elvenspeak import discovery, remote, router
from elvenspeak.engine import Capability

from smoke import (
    APPLE,
    CLI_TIMEOUT,
    DOCKER,
    PODMAN,
    ImageConfig,
    SmokeFailure,
    _apple_image_config,
    _attempt,
    _declared_port,
    _docker_image_config,
    _fleet_address,
    _read_config,
    select_runtime,
)

#: Verbatim from `container image inspect registry.sanctuary.gdn/elvenspeak-piper:2026.09.07.1`
#: on 2026-09-07 — Apple's runtime renders the healthcheck as Go's `%+v` of the
#: struct, and this is the only place it exposes it at all. Kept exactly as
#: printed because its two hard properties are both accidental-looking: the
#: command contains `]` (from `os.environ['PORT']`) and it contains `%s`.
PIPER_HEALTHCHECK_HISTORY = (
    "HEALTHCHECK {Test:[CMD-SHELL python -c \"import urllib.request,os; "
    "urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ['PORT'])\"] "
    "Interval:30s Timeout:5s StartPeriod:2m0s StartInterval:0s Retries:0}"
)

#: The command inside it — what both parsers must hand back, byte for byte.
PIPER_HEALTHCHECK = (
    "python -c \"import urllib.request,os; "
    "urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ['PORT'])\""
)


def apple_inspect(env: list[str], history: list[str]) -> str:
    """An Apple `image inspect` payload, shaped as the real one is."""
    return json.dumps(
        [
            {
                "id": "33284f32e77d",
                "variants": [
                    {
                        "config": {
                            "architecture": "amd64",
                            "config": {"Env": env, "Cmd": ["uv", "run", "--no-dev", "main.py"]},
                            "history": [{"created_by": line} for line in history],
                        }
                    }
                ],
            }
        ]
    )


def docker_inspect(env: list[str], test: list[str] | None) -> str:
    """A docker/podman payload in the shape `Runtime.inspect_image`'s template emits."""
    healthcheck = None if test is None else {"Test": test, "Interval": 30000000000}
    return json.dumps({"Env": env, "Healthcheck": healthcheck})


def test_the_apple_runtime_healthcheck_survives_the_bracket_inside_it():
    """The command ends `os.environ['PORT'])"`, so it closes a bracket of its own.

    A non-greedy match to the first `]` truncates it mid-expression and yields a
    command that is still a plausible-looking string — it would be `exec`d, fail
    for a reason having nothing to do with the image, and report the image
    unhealthy. Anchoring on the `Interval:` that follows the real closing bracket
    is what makes that unavailable, and this is the case that proves it.
    """
    config = _apple_image_config(apple_inspect(["PORT=5001"], [PIPER_HEALTHCHECK_HISTORY]))
    assert config.healthcheck == PIPER_HEALTHCHECK


def test_both_runtimes_read_one_image_the_same_way():
    """[LAW:one-source-of-truth] Two spellings of one image config, one answer.

    The parsers exist only because `docker` and Apple's `container` disagree
    about how to print an image; nothing about the *image* differs. If they ever
    return different answers for the same artifact, then which runtime ran the
    smoke test decides what was proved, and the check stops meaning one thing.
    """
    env = ["PATH=/usr/local/bin", "ELVENSPEAK_ENGINE=piper", "PORT=5001"]
    assert _apple_image_config(apple_inspect(env, [PIPER_HEALTHCHECK_HISTORY])) == (
        _docker_image_config(docker_inspect(env, ["CMD-SHELL", PIPER_HEALTHCHECK]))
    )


def test_the_port_comes_from_the_image_rather_than_from_this_repository():
    """An image built with a different PORT is polled on that port, not on 5001."""
    env = ["PORT=8080"]
    assert _apple_image_config(apple_inspect(env, [PIPER_HEALTHCHECK_HISTORY])).port == 8080
    assert _docker_image_config(docker_inspect(env, ["CMD-SHELL", "true"])).port == 8080


@pytest.mark.parametrize(
    "env",
    [[], ["PATH=/usr/local/bin"], ["PORTAL=5001"]],
    ids=["no environment", "an environment without PORT", "a name PORT is a prefix of"],
)
def test_an_image_that_names_no_port_is_refused(env):
    """[LAW:no-silent-failure] No default: a guessed port polls the wrong endpoint.

    Defaulting to 5001 here would turn "this image does not say where it serves"
    into "this image is unhealthy", which is a different sentence about a
    different problem.
    """
    with pytest.raises(SmokeFailure, match="declares no PORT"):
        _declared_port(env)


def test_a_port_that_is_not_a_number_is_refused():
    with pytest.raises(SmokeFailure, match="not a port number"):
        _declared_port(["PORT=highest"])


def test_an_environment_value_containing_an_equals_sign_survives():
    """Only the first separator splits, so a value with `=` in it is not truncated."""
    assert _declared_port(["PIPER_VOICES=a=b", "PORT=5001"]) == 5001


@pytest.mark.parametrize(
    ("apple_history", "docker_test"),
    [([], None), (["RUN echo hello"], None)],
    ids=["no history at all", "history naming no healthcheck"],
)
def test_an_image_declaring_no_healthcheck_is_refused_by_both(apple_history, docker_test):
    """Half this script's question is the image's own healthcheck; without one there
    is no question to ask, and reporting such an image "proven" would be a lie."""
    with pytest.raises(SmokeFailure, match="declares no HEALTHCHECK"):
        _apple_image_config(apple_inspect(["PORT=5001"], apple_history))
    with pytest.raises(SmokeFailure, match="declares no HEALTHCHECK"):
        _docker_image_config(docker_inspect(["PORT=5001"], docker_test))


def test_an_exec_form_healthcheck_is_refused_rather_than_guessed_at():
    """`["CMD", "prog", "arg"]` is space-joined beyond recovery in Apple's dump.

    Refusing it in both parsers keeps the two runtimes answering alike. Accepting
    it under `docker` only would mean an image that smokes clean in CI and cannot
    be smoked at all on the machine of whoever is asked to reproduce the failure.
    """
    with pytest.raises(SmokeFailure, match="only the shell form"):
        _docker_image_config(docker_inspect(["PORT=5001"], ["CMD", "wget", "-q", "http://x/y"]))
    with pytest.raises(SmokeFailure, match="only the shell form"):
        _apple_image_config(
            apple_inspect(
                ["PORT=5001"],
                ["HEALTHCHECK {Test:[CMD wget -q http://x/y] Interval:30s}"],
            )
        )


def test_the_last_healthcheck_declared_is_the_one_that_binds():
    """A later HEALTHCHECK replaces an earlier one, which is what the runtime does."""
    config = _apple_image_config(
        apple_inspect(
            ["PORT=5001"],
            [
                "HEALTHCHECK {Test:[CMD-SHELL /bin/false] Interval:30s}",
                PIPER_HEALTHCHECK_HISTORY,
            ],
        )
    )
    assert config.healthcheck == PIPER_HEALTHCHECK


def test_a_parsed_config_is_the_whole_answer():
    """[LAW:parse-dont-validate] Nothing downstream re-reads the inspect output."""
    assert _docker_image_config(
        docker_inspect(["PORT=5001"], ["CMD-SHELL", PIPER_HEALTHCHECK])
    ) == ImageConfig(port=5001, healthcheck=PIPER_HEALTHCHECK)


def test_podman_is_docker_with_a_different_binary_name():
    """The one thing that may differ between them, and the ones that may not.

    They are one instance apart on purpose: `podman` accepts docker's argv for
    every call this script makes, verified against both. A future edit that gave
    podman its own `running_names` or `read_config` would be re-describing a
    runtime that has not diverged.
    """
    assert PODMAN.binary == "podman" and DOCKER.binary == "docker"
    assert PODMAN.running_names == DOCKER.running_names
    assert PODMAN.inspect_image == DOCKER.inspect_image
    assert PODMAN.read_config is DOCKER.read_config


def test_asking_for_a_runtime_that_is_not_installed_names_what_is(monkeypatch):
    """[LAW:no-silent-failure] No falling through to whatever else is lying around.

    Silently running podman when docker was asked for would smoke a different
    artifact path than the one the caller meant to test, and say nothing.
    """
    monkeypatch.setattr("smoke.shutil.which", lambda binary: binary == "podman")
    with pytest.raises(SmokeFailure, match="wanted docker, found podman"):
        select_runtime("docker")
    assert select_runtime(None) is PODMAN


def test_no_runtime_at_all_is_refused_naming_every_candidate(monkeypatch):
    monkeypatch.setattr("smoke.shutil.which", lambda binary: None)
    with pytest.raises(SmokeFailure, match="none installed"):
        select_runtime(None)



#: The oldest interpreter `python3 smoke.py <image>` must survive. Held here
#: rather than in a docstring because a version nobody re-checks is a claim, and
#: this one is checkable: 3.8 is what Ubuntu 20.04 answers `python3` with, and a
#: runner that old is exactly the one nobody would think to try before pushing.
OLDEST_PYTHON = (3, 8)


def test_the_smoke_script_parses_on_an_interpreter_nobody_installed():
    """The premise of the whole file, machine-checked instead of asserted.

    `smoke.py` argues in its own docstring that it imports nothing outside the
    standard library so that it runs on a bare CI runner and an unactivated
    laptop — which buys nothing if the *syntax* is newer than the `python3`
    already there. A version cliff is the one portability break that cannot
    report itself: the file dies at parse time, so none of its careful failure
    messages ever get to run, and the operator sees a `SyntaxError` in a file
    they were told needed no setup.

    `feature_version` reproduces that judgement without the interpreter being
    installed, which matters because the machines that would catch it honestly
    are the ones nobody develops on. It is not a tautology: run against the
    commit that introduced this file, it fails on the `match` statement in
    `_shell_healthcheck`.
    """
    source = (pathlib.Path(__file__).parent.parent / "smoke.py").read_text()
    ast.parse(source, "smoke.py", feature_version=OLDEST_PYTHON)


def test_a_wedged_runtime_call_becomes_a_result_cleanup_can_print(monkeypatch):
    """[LAW:dataflow-not-control-flow] A timeout during cleanup is data, not an exception.

    `_attempt` is what the context manager's `finally` runs, and it is the reason
    that block cannot throw. A `logs` or `rm` call against a wedged daemon used to
    raise `TimeoutExpired` out of the `finally` — skipping the removal on the next
    line and replacing whatever failure sent us there. Coming back as a return
    code instead means both call sites report it through the printing they already
    do, with no arm written for the wedged case.
    """

    def wedge(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, CLI_TIMEOUT)

    monkeypatch.setattr("smoke.subprocess.run", wedge)
    done = _attempt(["docker", "rm", "--force", "elvenspeak-smoke-abc"])

    assert done.returncode == 124
    assert "did not return within" in done.stderr


@pytest.mark.parametrize(
    ("runtime", "stdout"),
    [
        (DOCKER, ""),
        (DOCKER, "Error: image not found"),
        (DOCKER, "{}"),
        (DOCKER, '{"Env": ["PORT=5001"]}'),
        (DOCKER, "[]"),
        (DOCKER, '{"Env": [5001], "Healthcheck": null}'),
        (DOCKER, '{"Env": ["PORT=5001"], "Healthcheck": 7}'),
        (APPLE, ""),
        (APPLE, "[]"),
        (APPLE, '[{"variants": []}]'),
        (APPLE, '[{"variants": [{"config": {}}]}]'),
    ],
    ids=[
        "docker-empty",
        "docker-not-json",
        "docker-no-keys",
        "docker-no-healthcheck-key",
        "docker-array-not-object",
        "docker-env-not-strings",
        "docker-healthcheck-not-object",
        "apple-empty",
        "apple-no-image",
        "apple-no-variants",
        "apple-no-history",
    ],
)
def test_a_runtime_that_answers_in_an_unreadable_shape_is_refused_not_traced(
    monkeypatch, runtime, stdout
):
    """[LAW:no-silent-failure] The parsers walk into JSON by index; the boundary owns the fall.

    Indexing straight into a runtime's output is right for a shape that is a
    contract and wrong for one that is not, and every miss used to leave as a bare
    `KeyError`/`IndexError`/`JSONDecodeError`/`AttributeError` — past `main`'s
    `except SmokeFailure` and out as a traceback, which reads like a bug in this
    file rather than a runtime that answered strangely. Both spellings cross at one
    boundary, so both are listed here: the point of covering them together is that
    neither parser carries its own apology for them to drift apart
    ([LAW:single-enforcer]).
    """
    monkeypatch.setattr("smoke._capture", lambda argv: stdout)
    with pytest.raises(SmokeFailure, match="shape this cannot read"):
        _read_config(runtime, "registry.example/elvenspeak-piper:2026.09.07.1")


def test_a_readable_shape_that_is_still_refused_keeps_its_own_sentence(monkeypatch):
    """The generic refusal must not swallow the specific one.

    `_shell_healthcheck` and `_declared_port` raise `SmokeFailure`, which is not a
    `LookupError`, `ValueError`, `TypeError` or `AttributeError` — so their
    carefully worded refusals pass the boundary untouched rather than being
    rewrapped as "a shape this cannot read", which would be true of nothing and
    useless to everyone. Held here because it rests on `SmokeFailure`'s base class.
    """
    monkeypatch.setattr(
        "smoke._capture",
        lambda argv: docker_inspect(["PORT=5001"], ["CMD", "curl", "-f", "/health"]),
    )
    with pytest.raises(SmokeFailure, match="only the shell form"):
        _read_config(DOCKER, "registry.example/elvenspeak-piper:2026.09.07.1")


# ---------------------------------------------------------------------------
# The stub fleet (`fleetstub.py`), and the three facts it states that it cannot
# import.
#
# It runs inside the image under test, on that image's bare `python3` rather than
# on the uv virtualenv the package lives in, so `from elvenspeak import ...` fails
# there — see its own docstring. Every value it therefore has to spell out is held
# equal to its owner below rather than promised in a comment, which is the
# arrangement `tests/test_workflow.py` already uses to hold the gitea engine
# matrix equal to `elvenspeak.engines.ENGINES`.
#
# The point of each of these is a failure that is silent in the shape that matters
# most: a stub that registers under the wrong tag, or fills a variable the image
# no longer reads, does not produce a mismatch anyone can see. It produces an
# empty fleet — a router that boots correctly, has no voices, and answers 503
# until `smoke.py` gives up at 180 seconds, which reads like a hang and says
# nothing about the cause.
# ---------------------------------------------------------------------------


def test_the_stub_registers_under_the_tag_discovery_selects_on():
    """A stub tagged anything else is a stub the router is right to ignore."""
    assert fleetstub.ENGINE_TAG == discovery.ENGINE_TAG


def test_the_variable_the_stub_fills_is_the_one_the_router_reads():
    """`--fleet` is only worth anything if the image is looking where it points."""
    assert smoke.CONSUL_URL_VAR == router.CONSUL_URL


def test_the_paths_the_stub_answers_are_the_paths_discovery_asks():
    """Held against the lookups `discovery` builds, not against its own constants.

    [LAW:behavior-not-structure] `discovery` composes both URLs inline, so there is
    nothing there to compare a string to. What can be compared is the request it
    actually makes: a recording agent is asked for a fleet, and every target it
    was given must be one `fleetstub` answers.
    """
    asked: list[str] = []

    def record(url: str, what: str) -> object:
        asked.append(url.removeprefix("http://consul.example"))
        if url.endswith(fleetstub.CATALOG_PATH):
            return {fleetstub.SERVICE: [fleetstub.ENGINE_TAG]}
        return []

    with mock.patch.object(discovery, "_fetch", record):
        discovery.engines("http://consul.example")

    answer = fleetstub.answering("http://fleet.example:8500", {})
    assert {target.partition("?")[0] for target in asked} == {
        fleetstub.CATALOG_PATH,
        fleetstub.HEALTH_PATH + fleetstub.SERVICE,
    }
    assert all(answer(target) is not None for target in asked)


def test_the_voice_the_stub_publishes_is_one_a_router_can_parse():
    """The shape `elvenspeak.remote` requires, checked by that parser and no other.

    A hand-written payload is a rendering of `elvenspeak.api`'s serialization that
    the compiler cannot check, and `_voice` is strict in three separate ways that
    each raise their own `ConfigError`: a voice naming no capabilities, no models
    or no language is refused rather than defaulted. Asserting the keys are present
    would only confirm the fixture matches itself; running the real parser over it
    is what makes this fail the day `_voice` asks for a fourth thing.
    """
    parsed = remote._voice(fleetstub.VOICE, fleetstub.SERVICE)

    assert parsed.id == fleetstub.VOICE["voice_id"]
    assert parsed.models == frozenset(fleetstub.VOICE["models"])
    assert parsed.language == fleetstub.VOICE["language"]
    assert parsed.capabilities == frozenset(Capability)


def test_every_capability_the_stub_claims_is_one_that_exists():
    """Not that it claims all of them — that it claims nothing invented.

    `_voice` reads an unknown capability as absent, so a typo here would quietly
    narrow what the stub's voice offers instead of failing, and the assertion above
    would then be the thing that broke, naming the wrong cause.
    """
    assert set(fleetstub.VOICE["capabilities"]) <= {
        capability.name.lower() for capability in Capability
    }


@contextmanager
def stub_serving(make_handler) -> Iterator[str]:
    """`fleetstub` on loopback, yielding the base URL it was told to advertise.

    `make_handler` is handed that URL and returns the handler, because the fleet
    has to advertise the address it is reachable at and nothing knows that address
    until the socket is bound. Binding happens in the constructor and the handler
    class is only consulted per request, so it can be settled in between — which
    beats picking a free port first and racing something else to it
    ([LAW:no-ambient-temporal-coupling]).

    The handler itself comes from [`fleetstub.handler`], not from a copy: what is
    being proven is the file that will run inside the container, including its path
    matching and its query parsing, both of which the FastAPI transport in
    `tests/fleet.py` replaces with its own. Loopback stands in for the container
    network — the router asks over real HTTP either way, and the address the
    catalog carries is the one it dials.
    """
    served = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), http.server.BaseHTTPRequestHandler
    )
    host, port = served.server_address[:2]
    base = f"http://{host}:{port}"
    served.RequestHandlerClass = make_handler(base)
    thread = threading.Thread(target=served.serve_forever, daemon=True)
    thread.start()
    try:
        yield base
    finally:
        served.shutdown()
        served.server_close()
        thread.join(timeout=10)
        # [LAW:no-silent-failure] An unchecked join leaks the thread and its bound
        # socket past the test that started them, and the suite stays green.
        assert not thread.is_alive(), "the stub fleet did not stop in time"


def stub_handler(health, unhealthy=None):
    """A [`stub_serving`] factory answering for `health`, as `fleetstub.serve` does."""
    return lambda base: fleetstub.handler(fleetstub.answering(base, health, unhealthy))


@contextmanager
def stub_fleet() -> Iterator[str]:
    """The fleet `fleetstub.serve` serves, on loopback instead of a container IP.

    Assembled the way `serve` assembles it: one tagged service whose sole instance
    is the server answering, advertising the address it is reachable at. Supplying
    that address is the whole difference between here and the container, and it is
    the one thing `fleetstub.own_ip` exists to decide there.
    """

    def fleet(base: str) -> type[http.server.BaseHTTPRequestHandler]:
        host, _, port = base.removeprefix("http://").rpartition(":")
        registered = {fleetstub.SERVICE: [fleetstub.health_entry(host, int(port))]}
        return stub_handler(registered)(base)

    with stub_serving(fleet) as base:
        yield base


def test_a_router_boots_against_the_stub_fleet_and_offers_its_voice():
    """The whole of `piper-build-b4h.4`, one rung below the container.

    [LAW:verifiable-goals] The ticket's "done" is the router image answering 200
    once a backend is discoverable. The half of that which does not need a runtime
    is this: the real `elvenspeak.router`, configured exactly as a deployment
    configures it, discovering the real `fleetstub` over real HTTP and coming back
    with a voice. Nothing is patched — a patched client is a different program, for
    the reason `tests/test_router.py` gives.

    What this catches that no container run would catch quickly: every way the stub
    can be wrong takes 180 seconds to report inside CI and one second here.
    """
    with stub_fleet() as consul:
        engine = router.configure({router.CONSUL_URL: consul}, frozenset(), SERVES).open()

    assert [voice.id for voice in engine.voices()] == [fleetstub.VOICE["voice_id"]]


def test_the_stub_fleet_is_what_makes_the_router_answer_health_200():
    """The endpoint `smoke.py` actually waits on, against the server that serves it.

    Asserted through the whole app rather than on the engine, because the 503 this
    replaces is `elvenspeak.api`'s answer to an engine with no voices and not the
    router's own — so an engine-level assertion would prove the fleet was found
    while saying nothing about the status `--fleet` exists to change.

    Only the 200 half is here. `test_router.test_a_router_that_found_no_engines_
    reports_itself_unfit_at_health` already owns the 503, and it owns it better —
    it is the regression test for the 2026-09-02 deploy that produced the rule.
    Restating it here would be a second enforcer of one invariant
    ([LAW:single-enforcer]), sited at the reader least able to explain it.
    """
    with stub_fleet() as consul, routed(consul) as client:
        assert client.get("/health").status_code == 200


def test_both_transports_of_one_catalog_answer_a_lookup_the_same_way():
    """The drift this file's shared shape exists to prevent, asserted rather than argued.

    `fleetstub` matches Consul's two paths and parses `?passing=true` by hand;
    `tests/fleet.py` mounts the same answers behind FastAPI, which parses the query
    for it. Sharing [`passing_instances`] makes the *filtering* one rule, and leaves
    the two readings of the flag able to disagree — at which point the suite would
    keep proving a filter the container stub does not apply, which is the drift the
    hand-written twin in `test_discovery` already demonstrated once.

    So both are driven with one fleet holding a healthy instance and an unhealthy
    one, through the real `discovery`, and required to come back with one answer.
    [LAW:behavior-not-structure] — nothing here compares the parsers.
    """
    healthy = fleetstub.health_entry("10.0.0.4", 29280)
    failing = fleetstub.health_entry("10.0.0.9", 29289)
    catalog = {fleetstub.SERVICE: [fleetstub.ENGINE_TAG]}
    health = {fleetstub.SERVICE: [healthy]}
    unhealthy = {fleetstub.SERVICE: [failing]}

    with serving(consul_app(catalog, health, unhealthy)) as through_fastapi:
        by_suite = discovery.engines(through_fastapi)

    with stub_serving(stub_handler(health, unhealthy)) as through_stdlib:
        by_container = discovery.engines(through_stdlib)

    assert by_container == by_suite
    assert [backend.base_url for backend in by_container] == ["http://10.0.0.4:29280"]


def test_smoke_reads_the_address_the_stub_actually_publishes():
    """The seam between the two files, which neither can check alone.

    `fleetstub` answers `/address` and `smoke.py` reads it to build
    `ROUTER_CONSUL_URL`. Both sides looked right while the key was spelled
    differently on each, and nothing failed: the stub came up, the read raised a
    `LookupError` at the boundary, and the leg reported "answered with something
    that is not an address" — true, unhelpful, and about the wrong file.
    """
    with stub_fleet() as base:
        assert _fleet_address(base + fleetstub.ADDRESS_PATH).startswith("http://")


@pytest.mark.parametrize(
    "answered",
    ['{"url": "http://10.0.0.4:8500"}', "[]", '"http://10.0.0.4:8500"', "not json at all"],
    ids=["another key", "a list", "a bare string", "not json"],
)
def test_a_fleet_that_answers_with_something_else_is_named_not_traced(answered):
    """[LAW:parse-dont-validate] The boundary owns the fall, as `_read_config`'s does.

    A list or a bare string takes a string subscript and raises `TypeError`, not
    the `LookupError` a missing key gives — which is why catching only the obvious
    two would let the least likely answer out as a traceback, past `main`'s
    handler, reading like a bug in this file rather than a stub that is out of date.
    """
    with stub_serving(lambda base: _literal_handler(answered)) as base:
        with pytest.raises(SmokeFailure, match="not an address"):
            _fleet_address(base + fleetstub.ADDRESS_PATH)


def _literal_handler(body: str):
    """A handler answering `body` verbatim, whatever shape it is."""

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(body.encode("utf-8"))))
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))

        def log_message(self, fmt: str, *args: object) -> None:
            """Silenced: pytest owns this process's stderr, unlike the container's."""

    return Handler
