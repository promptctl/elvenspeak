"""Which elvenspeak engines are running, and where to reach them.

The router fronts other elvenspeak servers, and this module is the only thing
that answers "which ones". It asks Consul, which already carries that fact for
this cluster, rather than reading a roster somebody wrote down.

[LAW:one-source-of-truth] A configured list of engines and their addresses would
be a second copy of what the cluster already knows, and it drifts in exactly one
direction: toward a router offering a voice from a server that is no longer
there. The engines register themselves when they deploy; this asks. An engine
appears by being deployed and disappears by stopping, with no edit anywhere.

# Why a tag and not a name pattern

The services are named after their images — `elvenspeak-piper`, `elvenspeak-
kokoro` — and matching that prefix would make this module guess which services
are its business from the shape of a string. A tag is the cluster stating it. The
registration side of this decision is already deployed and carries the same
reasoning: see the `service` block in home-infra's `jobs/elvenspeak-piper.nomad.hcl`,
which tags itself `elvenspeak-engine` and says why.

So a third engine joins the fleet by deploying with that tag, and something that
merely happens to be called `elvenspeak-something` does not.

[LAW:effects-at-boundaries] The one place that talks to Consul. Everything above
takes [`Backend`] values and never learns that a service catalog exists, which is
what lets the router be tested against a fleet that is a tuple of values.
"""

from __future__ import annotations

import http.client
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from .provisioning import ConfigError

#: What a service must be tagged with to be one of ours. Not configurable: it is
#: half of a contract whose other half is written into every engine's job file,
#: and a setting here would let one side be changed alone.
ENGINE_TAG = "elvenspeak-engine"

#: Long enough for a loaded agent, short enough that a wedged one fails the boot
#: rather than hanging it. Discovery happens while the router is starting, so the
#: cost of this timeout is paid before the port is bound.
_TIMEOUT_SECONDS = 5.0

#: Everything another process can do to a socket we are reading, as one fact both
#: HTTP-speaking modules read. `urlopen` wraps only the connect and headers in a
#: `URLError`, so a server that dies or truncates while the body is still being
#: read raises a bare `ConnectionResetError` (an `OSError`) or an
#: `IncompleteRead` (an `http.client.HTTPException`) instead — and a handler
#: catching only `URLError` promises more than it delivers.
#:
#: [LAW:one-source-of-truth] Here rather than in each module because it was
#: written twice and the copies immediately disagreed: `remote` was corrected and
#: this module, making the same promise in its own docstring, was not.
#: [`elvenspeak.remote`] already depends on this one, so a single definition
#: costs no new edge.
TRANSPORT_FAILURES = (OSError, http.client.HTTPException)


@dataclass(frozen=True)
class Backend:
    """One running elvenspeak server, by the address that reaches it.

    `service` is the Consul service name, carried for messages rather than for
    routing: when two engines offer the same voice id the operator has to be told
    *which two*, and an address alone would make them go look it up.

    [LAW:types-are-the-program] No voices and no capabilities here. What a server
    can speak is that server's own answer, asked over HTTP at the moment it
    matters; a field for it on this type would be a snapshot free to disagree
    with the thing it describes.
    """

    service: str
    #: Scheme and authority, no trailing slash — `http://10.0.1.4:29280`. Paths
    #: are appended by whoever calls it, so this is never half of a URL that
    #: somebody has to remember not to double the slash on.
    base_url: str


def _fetch(url: str, what: str) -> Any:
    """The JSON at `url`, or a [`ConfigError`] saying which lookup failed.

    [LAW:no-silent-failure] Every failure here — refused connection, timeout, a
    non-200, a body that is not JSON — raises. The tempting alternative is to
    treat an unreachable Consul as an empty fleet, which is an answer-shaped
    void: "no engines are running" and "I could not find out" would arrive as the
    same value, and the router would boot into a healthy-looking service that can
    speak nothing.
    """
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS) as response:
            return json.load(response)
    except (*TRANSPORT_FAILURES, ValueError) as failure:
        raise ConfigError([f"{what} failed ({url}): {failure}"]) from None


def _tagged_services(consul_url: str) -> tuple[str, ...]:
    """Every service name in the catalog carrying [`ENGINE_TAG`].

    Consul's catalog answers with the tags per service name, which is one request
    for the whole fleet — the alternative is asking about names this module would
    have had to know in advance, which is the roster it exists not to hold.
    """
    catalog = _fetch(f"{consul_url}/v1/catalog/services", "consul service catalog")
    if not isinstance(catalog, dict):
        raise ConfigError(
            [f"consul service catalog was not an object: {type(catalog).__name__}"]
        )
    return tuple(
        sorted(
            name
            for name, tags in catalog.items()
            if isinstance(tags, list) and ENGINE_TAG in tags
        )
    )


def _address(entry: Any, service: str) -> str:
    """Where one healthy instance actually listens.

    Consul leaves `Service.Address` empty for a service that did not override its
    node's address, and the node's is then the right one. That is the documented
    shape of the answer rather than a missing field, so both readings are taken
    here — the one place that has both halves — instead of downstream where only
    the empty string would survive.
    """
    if not isinstance(entry, dict):
        raise ConfigError([f"{service}: health entry was not an object"])
    registered = entry.get("Service")
    node = entry.get("Node")
    if not isinstance(registered, dict) or not isinstance(node, dict):
        raise ConfigError([f"{service}: health entry had no Service and Node objects"])
    host = registered.get("Address") or node.get("Address")
    port = registered.get("Port")
    if not isinstance(host, str) or not host or not isinstance(port, int):
        raise ConfigError(
            [f"{service}: health entry named no address and port ({entry!r})"]
        )
    # An IPv6 literal has to be bracketed or its own colons are read as the port
    # separator: `http://fd00::4:29280` has no unambiguous authority, and every
    # request built from it afterwards is against a different address than the
    # one Consul named. A hostname or IPv4 address contains no colon and is
    # unaffected, so this is the whole rule rather than a special case.
    authority = f"[{host}]" if ":" in host else host
    return f"http://{authority}:{port}"


def engines(consul_url: str) -> tuple[Backend, ...]:
    """Every healthy elvenspeak engine the cluster is currently running.

    Healthy, not registered: Consul's health endpoint filtered to passing is what
    distinguishes a server that can speak from one that has merely been declared.
    Each engine's own check reports passing only once its voices are open, so an
    instance offered here is one whose models are loaded — see the `check` block
    in `jobs/elvenspeak-piper.nomad.hcl`, which exists to make that true.

    An empty result is a real answer and not a failure: a cluster running no
    engines is a cluster the router can discover nothing in, and it says so by
    having no voices, which fails its own healthcheck. Every way of *not finding
    out* raises instead — see [`_fetch`].

    Order is stable, by service name then address, because the first voice a
    router offers is the fallback voice of every deployment that named none
    ([`elvenspeak.engine.Engine.voices`]), and a fleet that reordered between
    boots would move it.
    """
    found: list[Backend] = []
    for service in _tagged_services(consul_url):
        # Encoded for the same reason a voice id is in [`elvenspeak.remote`]: the
        # name comes out of Consul's catalog, not from anything this module
        # constrains, and one holding `?` or `/` would corrupt the query string
        # or redirect the path — silently asking about a different service.
        named = urllib.parse.quote(service, safe="")
        instances = _fetch(
            f"{consul_url}/v1/health/service/{named}?passing=true",
            f"consul health lookup for {service}",
        )
        if not isinstance(instances, list):
            raise ConfigError([f"{service}: health lookup was not a list"])
        found.extend(
            Backend(service=service, base_url=_address(entry, service))
            for entry in instances
        )
    return tuple(sorted(found, key=lambda backend: (backend.service, backend.base_url)))
