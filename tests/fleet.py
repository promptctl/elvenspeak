"""A fleet of real elvenspeak servers on loopback, and a Consul that finds them.

The router's whole job is to talk to other elvenspeak servers over HTTP, so a
test that replaced the HTTP with a patched function would be testing a different
program. Everything here is real: real ASGI apps built by `create_app`, served by
the same uvicorn the image runs, discovered through a Consul-shaped catalog over
a real socket. What is fake is only the *engine* behind each server — the
existing `DeclaredEngine` stand-in — and the cluster that would otherwise have to
exist to run any of it.

Loopback, so this needs no network and no cluster. Port 0, so parallel runs and a
developer's already-bound 8500 never collide.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field

import uvicorn
from conftest import DeclaredEngine, DeclaredPrepared, declaring
from fastapi import FastAPI

from elvenspeak import api
from elvenspeak.discovery import ENGINE_TAG
from elvenspeak.engine import Capability, Voice
from elvenspeak.engines import ENGINES
from elvenspeak.settings import Settings
from elvenspeak.voices import Substitution

#: How long a server gets to bind before the test gives up on it. Generous for a
#: loaded machine; finite because a server that never starts must fail the test
#: rather than hang the suite.
_STARTUP_DEADLINE_SECONDS = 10.0


@contextmanager
def serving(app: FastAPI) -> Iterator[str]:
    """Runs `app` on a loopback port for the duration, yielding its base URL.

    [LAW:no-ambient-temporal-coupling] Waits on uvicorn's own `started` flag —
    the state whose whole job is answering "is it up" — rather than on a sleep
    long enough to usually work. The deadline turns a server that never starts
    into a named failure instead of a suite that hangs.
    """
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + _STARTUP_DEADLINE_SECONDS
        while not server.started:
            if time.monotonic() > deadline or not thread.is_alive():
                raise AssertionError("the test server never finished starting")
            time.sleep(0.01)
        bound: socket.socket = server.servers[0].sockets[0]
        yield f"http://127.0.0.1:{bound.getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(timeout=_STARTUP_DEADLINE_SECONDS)
        # [LAW:no-silent-failure] An unchecked join is the leak nobody sees: the
        # thread and its bound socket would outlive the test that started them,
        # accumulating across every case that serves anything, and the suite
        # would go green the whole way.
        assert not thread.is_alive(), "the test server did not stop in time"


def engine_app(
    engine_name: str,
    voices: tuple[Voice, ...],
    capabilities: frozenset[Capability] = frozenset(Capability),
    api_key: str | None = None,
) -> FastAPI:
    """One elvenspeak deployment, assembled exactly as `main.build` assembles one.

    `engine_name` is what this server will call itself in `GET /v1/models`, which
    is the word the router quotes when two of these collide.
    """
    settings = Settings(
        engine=DeclaredPrepared(capabilities),
        engine_name=engine_name,
        # Derived rather than stated beside it, for the reason `Settings.from_env`
        # derives both from one registry lookup: a deployment missing its own
        # engine from its own roster is a state production cannot reach.
        known_engines=frozenset(ENGINES) | {engine_name},
        withheld=frozenset(),
        fallback=Substitution.FIRST_OFFERED,
        api_key=api_key,
        host="127.0.0.1",
        port=0,
    )
    return api.create_app(settings, DeclaredEngine(declaring(capabilities, voices)))


@dataclass
class Registered:
    """One service as the stub Consul will report it."""

    service: str
    base_url: str
    tags: list[str] = field(default_factory=lambda: [ENGINE_TAG])
    #: Whether this instance's health check is passing. A stub that returned
    #: every instance regardless would let `?passing=true` be dropped from the
    #: lookup without a single test noticing — and that filter is what separates
    #: a server whose voices are open from one that is still loading them.
    passing: bool = True


def health_entry(host: str, port: int, service_address: str = "") -> dict:
    """One instance in the shape Consul's health endpoint reports it.

    `Service.Address` defaults to empty exactly as the real agent leaves it for a
    service that did not override its node's address, so the fallback to
    `Node.Address` is the ordinary path here rather than a special case only one
    test remembers to build.
    """
    return {
        "Node": {"Address": host},
        "Service": {"Address": service_address, "Port": port},
    }


def consul_app(
    catalog: dict[str, list[str]],
    health: dict[str, list[dict]],
    unhealthy: dict[str, list[dict]] | None = None,
) -> FastAPI:
    """The two endpoints [`elvenspeak.discovery`] asks, answering what it is told.

    [LAW:one-source-of-truth] The one Consul-shaped fake. There were two — this
    and a hand-written twin in `test_discovery` — and they drifted exactly as two
    copies do: when the `?passing=true` filter turned out to be unverified, the
    fix had to be made in both, and either could have been missed while the other
    kept its file green. The endpoint shapes, the filter and the address fallback
    are stated here, once, and both callers build on it.

    `unhealthy` holds instances that exist but fail their check. Withheld when the
    lookup filters to passing and returned when it does not, which is what makes
    the filter observable at all.
    """
    failing = unhealthy or {}
    app = FastAPI()

    @app.get("/v1/catalog/services")
    def services() -> dict:
        return catalog

    @app.get("/v1/health/service/{name}")
    def instances(name: str, passing: bool = False):
        """Honours `passing` exactly as the real agent does.

        `discovery` appends `?passing=true` so that only servers whose voices are
        already open are routed to. A stub that ignored the parameter would let
        that be deleted from the URL with the whole suite still green.
        """
        listed = list(health.get(name, []))
        return listed if passing else listed + list(failing.get(name, []))

    return app


def registered_consul(registered: list[Registered]) -> FastAPI:
    """A [`consul_app`] describing a fleet given as [`Registered`] services."""
    catalog: dict[str, list[str]] = {
        # Something that is not ours, to prove the tag is what selects rather than
        # the name looking familiar.
        "elvenspeak-lookalike": [],
        "gitea": ["vcs"],
    }
    health: dict[str, list[dict]] = {}
    unhealthy: dict[str, list[dict]] = {}
    for entry in registered:
        catalog[entry.service] = entry.tags
        entries = health if entry.passing else unhealthy
        entries.setdefault(entry.service, []).append(
            health_entry(_host(entry.base_url), _port(entry.base_url))
        )
    return consul_app(catalog, health, unhealthy)


def _host(base_url: str) -> str:
    return base_url.removeprefix("http://").split(":")[0]


def _port(base_url: str) -> int:
    return int(base_url.rsplit(":", 1)[1])


@contextmanager
def cluster(
    *engines: tuple[str, tuple[Voice, ...], frozenset[Capability]],
    replicas: int = 1,
    api_key: str | None = None,
) -> Iterator[str]:
    """A running fleet and the Consul that knows it, yielding the Consul's URL.

    Each entry becomes one elvenspeak server registered under `elvenspeak-<name>`
    and tagged as an engine, which is the registration the real job files perform.
    What the router is handed is therefore the same string a deployment hands it:
    somewhere to ask.

    `api_key` guards every engine in the fleet, which is the ordinary thing a
    deployment does and the case a router has to be able to reach.

    `replicas` scales every deployment, registering that many separate servers
    under the one service name — which is what Nomad does to a scaled job and
    what a router therefore has to tolerate. Real servers rather than one address
    listed twice, because the point is that the router sees several backends and
    must still see one deployment.
    """
    with ExitStack() as stack:
        registered = [
            Registered(
                service=f"elvenspeak-{name}",
                base_url=stack.enter_context(
                    serving(engine_app(name, voices, capabilities, api_key))
                ),
            )
            for name, voices, capabilities in engines
            for _ in range(replicas)
        ]
        yield stack.enter_context(serving(registered_consul(registered)))
