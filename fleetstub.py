#!/usr/bin/env python3
"""The smallest fleet a router can be booted against.

The router synthesizes nothing, so it is the one image whose smoke cannot be
"start it and ask". Started alone it discovers no engine, has no voices, and
`elvenspeak.api` answers `/health` 503 by design — `elvenspeak/router.py` says so
in as many words, and calls refusing that state a crash loop for a cluster that
is merely still starting. `smoke.py` waits for a 200, so an empty fleet does not
produce a red saying "no backends": it produces a 180-second timeout that reads
like a hang. Gitea run 38's router leg died at 43 seconds on the smaller version
of the same problem — no `ROUTER_CONSUL_URL` at all — and pointing that variable
at an empty catalog would have traded the 43 seconds for the 180.

So proving the router artifact needs something for it to find, and this is the
least that will do: a Consul-shaped catalog naming one tagged service, and a
server at that address answering the two questions the router asks every backend
while it is still constructing itself — `GET /v1/voices` and `GET /v1/models`
(`elvenspeak/remote.py`). Since `piper-routing-7e2.17` the second is not optional:
a backend that cannot say which models its voices speak makes `remote._voice`
raise while the router is still being constructed.

ONE PROCESS SERVING BOTH ROLES, which looks like a shortcut and is not. The
router reaches a backend at whatever address the catalog handed it and has no
opinion about whether that address is also the agent it asked — so a second
container would prove nothing further and would double what can fail. What is
being proven here is the *router*, not Consul.

WHY IT SHIPS AS SOURCE. `smoke.py` runs this inside a container started from the
image under test, by reading this file and passing it to that image's own
`python3 -c`. No second image is built, none is pulled, and nothing is mounted or
copied — which matters because the runner's filesystem and the daemon's are not
reliably the same one, and because on a fresh registry there is no other
elvenspeak image to borrow. The cost of that decision is this file's one hard
constraint: **standard library only, and no import from `elvenspeak`**. The image
runs the package from a uv virtualenv that a bare `python3` does not see, so an
import of it here fails inside the very container this exists to serve.

WHAT THAT CONSTRAINT COSTS, AND WHO PAYS IT. Three facts below are stated here
and owned elsewhere — the engine tag, the variable the router reads, and the
shape of a published voice. None of them can be imported, so each is held equal
to its owner by a test in `tests/test_smoke.py` rather than promised in a
comment. That is the same arrangement `tests/test_workflow.py` uses to hold the
gitea engine matrix equal to `elvenspeak.engines.ENGINES`.

WHAT IS SHARED RATHER THAN RESTATED. `tests/fleet.py` serves the same catalog to
the suite through FastAPI, and imports [`health_entry`], [`passing_instances`]
and the two path constants from here so that there is one description of Consul's
answer and two transports for it ([LAW:one-source-of-truth],
[LAW:effects-at-boundaries]). Inventing a second description of that API is the
defect the discovery epic exists to fix, and it had already happened once: a
hand-written twin in `test_discovery` drifted from this shape until `?passing=true`
turned out to be unverified in both.
"""

from __future__ import annotations

import argparse
import http.server
import json
import socket
import urllib.parse
from collections.abc import Callable, Mapping, Sequence

#: What a service must be tagged with for [`elvenspeak.discovery`] to treat it as
#: one of ours. Held equal to `discovery.ENGINE_TAG` by `tests/test_smoke.py`: a
#: stub registering under any other tag is a stub the router correctly ignores,
#: which would present as an empty fleet — the exact failure this file exists to
#: prevent, arriving in the exact shape it is hardest to read.
ENGINE_TAG = "elvenspeak-engine"

#: The two endpoints [`elvenspeak.discovery`] asks, and the two the router asks of
#: every backend it finds. Constants because `tests/fleet.py` mounts the first two
#: as FastAPI routes and this file matches all four by hand, and a path spelled in
#: both places is a path that can be corrected in one.
CATALOG_PATH = "/v1/catalog/services"
HEALTH_PATH = "/v1/health/service/"
VOICES_PATH = "/v1/voices"
MODELS_PATH = "/v1/models"

#: Where this process reports the address a *sibling container* reaches it at.
#: Not part of any API being imitated — see [`own_ip`] for why the answer has to
#: come from here rather than from the runtime or from `smoke.py`.
ADDRESS_PATH = "/address"

#: Consul's own port, which this is not — but a stub alone in a container can
#: have the obvious number, and the obvious number is the one an operator reading
#: `ROUTER_CONSUL_URL` in a failed run's logs will not have to look up.
DEFAULT_PORT = 8500

#: What the fleet calls itself. `elvenspeak-` prefixed like every real deployment,
#: `-stub` so that a router log naming it cannot be mistaken for a real engine
#: that wandered into a CI run.
SERVICE = "elvenspeak-stub"

#: The engine id this fleet's voice answers to. A router quotes it in the startup
#: line naming which engine each address turned out to be.
MODEL_ID = "elvenspeak-stub"

#: One voice, carrying exactly the fields `elvenspeak.remote._voice` reads and no
#: others. A real `GET /v1/voices` answers with some thirty ElevenLabs-compatible
#: keys around these; copying all of them would be a second, unchecked rendering
#: of `elvenspeak.api`'s serialization, free to drift into describing a server
#: this project does not ship. Every field here is refused by `_voice` when it is
#: absent or the wrong shape — `capabilities`, `models` and `language` each raise
#: a named `ConfigError` — so this is not a subset chosen for brevity but the
#: whole of what one backend must say to be routable.
#: `tests/test_smoke.py` drives the real parser over it rather than trusting this
#: paragraph.
VOICE = {
    "voice_id": "stub-voice",
    "name": "Stub",
    "description": "the one voice the smoke fleet offers",
    "labels": {"kind": "stub"},
    "capabilities": ["speed", "timestamps"],
    "models": [MODEL_ID],
    "language": "en",
}


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


def passing_instances(
    name: str,
    passing: bool,
    health: Mapping[str, Sequence[dict]],
    unhealthy: Mapping[str, Sequence[dict]],
) -> list[dict]:
    """The instances of `name` Consul reports, honouring `passing` as it does.

    `discovery` appends `?passing=true` so that only servers whose voices are
    already open are routed to. A stub that ignored the parameter would let that
    be deleted from the URL with the whole suite still green — which is what it
    did until the filter turned out to be unverified.
    """
    listed = list(health.get(name, ()))
    return listed if passing else listed + list(unhealthy.get(name, ()))


def own_ip() -> str:
    """The address a sibling container reaches this process at.

    ASKED OF THE KERNEL, because nobody else in the arrangement can answer it.
    `smoke.py` cannot: it knows the container's name and its published host port,
    and the address a *second* container must dial is neither of those. The
    runtime can, and every runtime spells that question differently — Apple's
    prints it in `container ls`, docker's behind an inspect format string — so
    reading it there would put a per-runtime parser in `smoke.py` for a fact this
    process already has. Container-name DNS would avoid the whole question and is
    not available: on Apple's runtime a container on a user-defined network does
    not resolve its siblings' names, only their addresses, which was measured
    rather than assumed.

    A connected UDP socket sends nothing. It asks the routing table which local
    address would be used to reach the given one, which is the same address a
    sibling on that network dials back. `192.0.2.1` is RFC 5737 TEST-NET-1 —
    documentation space, guaranteed never routed to anything real — so this
    cannot accidentally depend on a host being up.

    Loopback would be the wrong answer and is not a reachable one: it is the
    container's own, and a router in a different container dialling it reaches
    itself. A host with no route at all raises here, at boot, in a process whose
    logs `smoke.py` always prints.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("192.0.2.1", 9))
        return str(probe.getsockname()[0])


def answering(
    base_url: str,
    health: Mapping[str, Sequence[dict]],
    unhealthy: Mapping[str, Sequence[dict]] | None = None,
) -> Callable[[str], object]:
    """The whole fleet as one function from a request target to its JSON answer.

    A function rather than a set of handlers because the two roles this process
    plays — the agent that is asked where the engines are, and the engine that is
    asked what it can say — differ only in which path arrived. `None` means no
    such route, which no route answers legitimately, so the caller needs no second
    value to tell "not found" from "found nothing".

    Dispatch on the path is a branch and stays one: the path *is* the domain's
    discriminator here, and a table would only spell the same four cases with a
    prefix match bolted onto its side.
    """
    catalog = {SERVICE: [ENGINE_TAG]}
    failing = unhealthy or {}

    def answer(target: str) -> object:
        path, _, query = target.partition("?")
        if path == CATALOG_PATH:
            return catalog
        if path.startswith(HEALTH_PATH):
            # Unquoted because `discovery` quotes it: a service name is whatever
            # the catalog said, so the lookup encodes it rather than trusting it
            # to be path-safe. Reading it back raw would ask about a service whose
            # name merely looked similar.
            name = urllib.parse.unquote(path[len(HEALTH_PATH) :])
            asked = urllib.parse.parse_qs(query).get("passing") == ["true"]
            return passing_instances(name, asked, health, failing)
        if path == VOICES_PATH:
            return {"voices": [VOICE]}
        if path == MODELS_PATH:
            return [{"model_id": MODEL_ID}]
        if path == ADDRESS_PATH:
            return {"base_url": base_url}
        return None

    return answer


def handler(answer: Callable[[str], object]) -> type[http.server.BaseHTTPRequestHandler]:
    """`answer` as something an `http.server` can be given.

    Separate from [`serve`] so that the suite can drive this exact class rather
    than a copy of it. [`serve`] ends in `serve_forever` and binds a fixed port,
    so a test reaching the handler through it would have to either restate these
    lines or start a real server on a port it does not own — and a restated
    handler is a second rendering of the one thing the container actually runs,
    free to be correct while the shipped one is not ([LAW:one-source-of-truth]).
    """

    class Handler(http.server.BaseHTTPRequestHandler):
        # Declared, so that `Content-Length` below is honoured as a message
        # boundary and a client asking several questions is not made to reconnect
        # between them. The router asks two of every backend.
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            found = answer(self.path)
            body = json.dumps(found if found is not None else {"error": self.path})
            self.send_response(404 if found is None else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body.encode("utf-8"))))
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))

    return Handler


def serve(port: int) -> None:
    """Answer for the fleet on `port`, until the container is removed.

    Never returns. The lifetime belongs to `smoke.py`'s context manager, which
    removes this container after the image under test has been proven or has
    failed — so there is no shutdown path here to get wrong, and none that could
    run while the router still needs an answer.
    """
    # Asked once. The catalog's address and the URL a router is pointed at are the
    # same fact, and two calls could answer differently on a host that gained a
    # route between them — a fleet advertising an address it is not reachable at.
    ip = own_ip()
    address = f"http://{ip}:{port}"
    answer = answering(address, {SERVICE: [health_entry(ip, port)]})

    # The one line that says what a router pointed here will be told. It is
    # printed before the socket is bound rather than after, because `serve_forever`
    # does not come back to print anything, and `smoke.py` prints these logs on
    # every path — so a run whose router could not reach this address still shows
    # which address it was offering.
    print(f"fleetstub: serving {SERVICE} at {address}", flush=True)
    http.server.ThreadingHTTPServer(("", port), handler(answer)).serve_forever()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve the smallest fleet an elvenspeak router can boot against.",
        epilog=(
            "Answers Consul's catalog and health endpoints for one tagged service, "
            "and that service's own /v1/voices and /v1/models. Runs until killed."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="the port to serve on (default: %(default)s)",
    )
    serve(parser.parse_args(argv).port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
