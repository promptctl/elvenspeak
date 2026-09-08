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

AND IT SPEAKS, because `speaks.py` asks the router every property it asks of every
other image, and a router synthesizes nothing of its own — each of those answers
is a backend's, relayed. Nothing here opens a model, and nothing needs to: the
properties being asked are arithmetic on the *length* of an utterance — more text
is more audio, twice the speed is half of it, a character alignment that runs
forward and stops where the audio does — so what a backend has to be able to do is
return the right number of samples and say truthfully where in them each character
fell. The samples themselves are zeros. `speaks.py` names audio content as the one
property it cannot check from bytes, so filling them with a tone would be a
decoration no reader reads — the rule [`VOICE`] already follows in carrying exactly
the fields `remote._voice` reads and no others.

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
import base64
import http.server
import json
import socket
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

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

#: Where a synthesis arrives, in the two spellings [`elvenspeak.remote`] builds.
#: Named here because this file matches them by hand and `remote._spoken_at`
#: composes them inline, so there is no constant over there to be held equal to —
#: what holds these right is `tests/test_smoke.py` driving the real client at the
#: real handler, which is the same arrangement the Consul paths get.
SPEAK_PREFIX = "/v1/text-to-speech/"
STREAM_SUFFIX = "/stream"
TIMED_SUFFIX = "/with-timestamps"

#: Two bytes to a sample, because every `pcm_*` format is signed 16-bit. The same
#: arithmetic `speaks.py` does from the far side of the wire, and the reason a
#: byte count there is a sample count.
BYTES_PER_SAMPLE = 2

#: How long one character takes to say at speed 1.0. Not a measurement of
#: anything — no model runs here — but the constant that makes the properties
#: `speaks.py` checks true of this stub. Small enough that the longest utterance
#: conformance asks for is under three seconds of samples to build, to base64 and
#: to carry, and large enough that a rounding difference cannot swallow the gap
#: between two texts a dozen characters apart.
SECONDS_PER_CHARACTER = 0.04

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


class Malformed(Exception):
    """A synthesis request this stub could not read, carrying its own reason.

    [LAW:no-silent-failure] Raised rather than answered with a zero-length
    utterance, which is the shape of the mistake this whole file exists to keep
    out of CI: a stub whose failure is indistinguishable from a working answer
    costs a leg its full timeout and reports the wrong subject when it expires.
    """


@dataclass(frozen=True)
class Answer:
    """One HTTP response: its status, what the bytes are, and the bytes.

    [LAW:types-are-the-program] The catalogue answers JSON and the stream answers
    raw samples, so a route's media type is a value the route decides rather than
    a fact the handler already knows. Before this file spoke, every answer was
    JSON and [`handler`] could encode them all itself; a second handler method for
    the audio would have been that vanished assumption preserved by copying it.
    """

    status: int
    content_type: str
    body: bytes


@dataclass(frozen=True)
class Synthesis:
    """One synthesis this stub was asked for: what to say, how fast, at what rate.

    [LAW:parse-dont-validate] Built only by [`asked_for`], which refuses anything
    it cannot read, so nothing below divides by a text length that might be zero
    or multiplies by a speed that might be a string. The absent `voice_settings`
    an ElevenLabs-compatible caller is entitled to omit is resolved to 1.0 there
    too — at the crossing, once — rather than defaulted again at each place that
    would otherwise have to wonder.
    """

    text: str
    speed: float
    rate: int

    @property
    def samples(self) -> int:
        """How many samples this utterance runs to.

        The one arithmetic in this file that `speaks.py` is really asking about,
        stated once so that the streamed audio and the alignment measured over it
        cannot describe two different utterances ([LAW:one-source-of-truth]).
        """
        return round(self.rate * len(self.text) * SECONDS_PER_CHARACTER / self.speed)

    @property
    def pcm(self) -> bytes:
        """The utterance as signed 16-bit samples — silent ones. See the header."""
        return bytes(BYTES_PER_SAMPLE * self.samples)


def asked_for(query: str, body: bytes) -> Synthesis:
    """The synthesis `query` and `body` describe, or [`Malformed`].

    The rate comes from `output_format` because that is where a caller states it
    and there is nowhere else to learn it: `elvenspeak.remote` asks every backend
    for `pcm_48000` and `speaks.py` asks a server directly for `pcm_22050`, and a
    stub that answered one fixed rate would be lying to whichever of them it was
    not built for — undetectably, since every property either one checks is a
    ratio and survives the wrong rate intact.
    """
    formats = urllib.parse.parse_qs(query).get("output_format", [])
    if len(formats) != 1 or not formats[0].startswith("pcm_"):
        raise Malformed(f"output_format must be one pcm_<rate>, got {formats!r}")
    try:
        rate = int(formats[0].removeprefix("pcm_"))
    except ValueError as unreadable:
        raise Malformed(f"{formats[0]!r} does not name a sample rate") from unreadable

    try:
        asked = json.loads(body)
    except ValueError as unreadable:
        raise Malformed(f"the request body is not JSON: {unreadable}") from unreadable
    if not isinstance(asked, dict):
        raise Malformed(f"the request body is not an object: {asked!r:.200}")
    text = asked.get("text")
    if not isinstance(text, str) or not text:
        raise Malformed(f"no text to speak: {text!r:.200}")

    settings = asked.get("voice_settings")
    speed = settings.get("speed", 1.0) if isinstance(settings, dict) else 1.0
    if not isinstance(speed, (int, float)) or speed <= 0:
        raise Malformed(f"speed must be a positive number, got {speed!r:.200}")
    return Synthesis(text=text, speed=float(speed), rate=rate)


def _json(payload: object, status: int = 200) -> Answer:
    """`payload` as the JSON body every route but the stream answers with."""
    return Answer(status, "application/json", json.dumps(payload).encode("utf-8"))


def _measured(spoken: Synthesis) -> dict:
    """`spoken` in the shape `elvenspeak.remote.speak_timed` reads it.

    Three keys, which is what that parser reads and no more — the rule [`VOICE`]
    follows for the same reason. `character_start_times_seconds` is published by
    the real surface and is not here: nothing consumes it, and a field nothing
    consumes is a second unchecked rendering of `elvenspeak.alignment` free to
    describe a timeline this file is not producing.

    [LAW:one-source-of-truth] The boundaries are divided out of the *sample count*
    rather than out of the ideal duration it was rounded from, so the last one
    lands exactly on the last sample. `remote._timings` clamps timestamps that
    overshoot the audio, and a stub relying on that clamp would be a stub whose
    alignment is quietly wrong everywhere the clamp happens to hide it.
    """
    samples = spoken.samples
    ends = [
        (index + 1) * samples / len(spoken.text) / spoken.rate
        for index in range(len(spoken.text))
    ]
    return {
        "audio_base64": base64.b64encode(spoken.pcm).decode("ascii"),
        "alignment": {
            "characters": list(spoken.text),
            "character_end_times_seconds": ends,
        },
        # What `speak_timed` reads to decide `TimedSpeech.measured`. This stub did
        # place every character deliberately, so claiming the word-exact fidelity
        # is the true answer rather than the flattering one.
        "alignment_fidelity": "word-exact",
    }


def answering(
    base_url: str,
    health: Mapping[str, Sequence[dict]],
    unhealthy: Mapping[str, Sequence[dict]] | None = None,
) -> Callable[[str], object]:
    """The whole fleet as one function from a request to the answer it gets.

    A function rather than a set of handlers because the two roles this process
    plays — the agent that is asked where the engines are, and the engine that is
    asked to speak — differ only in which request arrived. `None` means no such
    route, which no route answers legitimately, so the caller needs no second
    value to tell "not found" from "found nothing".

    [LAW:dataflow-not-control-flow] `body` is a value every route is handed and
    the catalogue routes ignore, which is what lets one reply path serve both HTTP
    methods. Taking the body only on the routes that read it would put the request
    method back into the dispatch, and [`handler`] would need a second copy of
    itself to supply it.

    Dispatch on the path is a branch and stays one: the path *is* the domain's
    discriminator here, and a table would only spell the same seven cases with two
    prefix matches bolted onto its side.
    """
    catalog = {SERVICE: [ENGINE_TAG]}
    failing = unhealthy or {}
    spoken_at = f"{SPEAK_PREFIX}{urllib.parse.quote(VOICE['voice_id'], safe='')}"

    def answer(target: str, body: bytes) -> Answer | None:
        path, _, query = target.partition("?")
        if path == CATALOG_PATH:
            return _json(catalog)
        if path.startswith(HEALTH_PATH):
            # Unquoted because `discovery` quotes it: a service name is whatever
            # the catalog said, so the lookup encodes it rather than trusting it
            # to be path-safe. Reading it back raw would ask about a service whose
            # name merely looked similar.
            name = urllib.parse.unquote(path[len(HEALTH_PATH) :])
            asked = urllib.parse.parse_qs(query).get("passing") == ["true"]
            return _json(passing_instances(name, asked, health, failing))
        if path == VOICES_PATH:
            return _json({"voices": [VOICE]})
        if path == MODELS_PATH:
            return _json([{"model_id": MODEL_ID}])
        if path == ADDRESS_PATH:
            return _json({"base_url": base_url})
        # Matched against the one voice this fleet offers, quoted the way
        # `remote._spoken_at` quotes it, so that asking for a voice nobody
        # published is the 404 it is rather than an utterance invented for it.
        if path == f"{spoken_at}{STREAM_SUFFIX}":
            return Answer(200, "audio/pcm", asked_for(query, body).pcm)
        if path == f"{spoken_at}{TIMED_SUFFIX}":
            return _json(_measured(asked_for(query, body)))
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
        # between them. The router asks two of every backend before it speaks to
        # any of them, and then speaks to one over the same connection.
        protocol_version = "HTTP/1.1"

        def _reply(self) -> None:
            # A GET carries no `Content-Length` and so reads no body, which is the
            # absence itself rather than a special case: an empty request body is
            # what the catalogue routes are handed and ignore.
            length = int(self.headers.get("Content-Length") or 0)
            try:
                found = answer(self.path, self.rfile.read(length))
            except Malformed as unreadable:
                # [LAW:single-enforcer] Both refusals this process can make are
                # phrased here, at the one place a reply is written, so a stub that
                # cannot read a request and a stub that has no such route answer in
                # one shape and are read by one client path.
                found = _json({"error": str(unreadable)}, 400)
            reply = found if found is not None else _json({"error": self.path}, 404)
            self.send_response(reply.status)
            self.send_header("Content-Type", reply.content_type)
            self.send_header("Content-Length", str(len(reply.body)))
            self.end_headers()
            self.wfile.write(reply.body)

        # [LAW:dataflow-not-control-flow] One operation, named twice because
        # `BaseHTTPRequestHandler` dispatches on the method. The method is not a
        # discriminator of anything this fleet does — the path is — so making it
        # one here would be inventing a fork the domain does not have.
        do_GET = _reply
        do_POST = _reply

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
            "that service's own /v1/voices and /v1/models, and requests to speak in "
            "the one voice it publishes. Runs until killed."
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
