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
from conftest import DeclaredEngine, DeclaredPrepared
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


def engine_app(
    engine_name: str,
    voices: tuple[Voice, ...],
    capabilities: frozenset[Capability] = frozenset(Capability),
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
        api_key=None,
        host="127.0.0.1",
        port=0,
    )
    return api.create_app(settings, DeclaredEngine(capabilities, voices))


@dataclass
class Registered:
    """One service as the stub Consul will report it."""

    service: str
    base_url: str
    tags: list[str] = field(default_factory=lambda: [ENGINE_TAG])


def consul_app(registered: list[Registered]) -> FastAPI:
    """A Consul answering the two endpoints [`elvenspeak.discovery`] asks.

    Shaped after the real agent's replies, including the part that matters most
    for the parsing: `Service.Address` is left empty here exactly as Consul leaves
    it for a service that did not override its node's address, so the fallback to
    `Node.Address` is exercised by the same fixture that exercises everything
    else rather than only by a unit test that asserts it in isolation.
    """
    app = FastAPI()

    @app.get("/v1/catalog/services")
    def catalog() -> dict:
        return {entry.service: entry.tags for entry in registered} | {
            # Something that is not ours, to prove the tag is what selects rather
            # than the name looking familiar.
            "elvenspeak-lookalike": [],
            "gitea": ["vcs"],
        }

    @app.get("/v1/health/service/{name}")
    def health(name: str) -> list:
        return [
            {
                "Node": {"Address": _host(entry.base_url)},
                "Service": {"Address": "", "Port": _port(entry.base_url)},
            }
            for entry in registered
            if entry.service == name
        ]

    return app


def _host(base_url: str) -> str:
    return base_url.removeprefix("http://").split(":")[0]


def _port(base_url: str) -> int:
    return int(base_url.rsplit(":", 1)[1])


@contextmanager
def cluster(
    *engines: tuple[str, tuple[Voice, ...], frozenset[Capability]],
) -> Iterator[str]:
    """A running fleet and the Consul that knows it, yielding the Consul's URL.

    Each entry becomes one elvenspeak server registered under `elvenspeak-<name>`
    and tagged as an engine, which is the registration the real job files perform.
    What the router is handed is therefore the same string a deployment hands it:
    somewhere to ask.
    """
    with ExitStack() as stack:
        registered = [
            Registered(
                service=f"elvenspeak-{name}",
                base_url=stack.enter_context(
                    serving(engine_app(name, voices, capabilities))
                ),
            )
            for name, voices, capabilities in engines
        ]
        yield stack.enter_context(serving(consul_app(registered)))
