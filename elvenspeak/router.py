"""An engine whose voices are spoken by other elvenspeak servers.

Registered in [`elvenspeak.engines`] beside Piper and Kokoro, because it is not a
new kind of component: [`elvenspeak.engine`] promised that every member of its
seam is "answerable by a remote HTTP API and by a local ONNX model alike", and a
deployment running this one keeps the whole surface — the catalog, the fallback,
the encoding path, the 501 gate, the ignored header — with a fleet behind it
instead of a model file.

`ELVENSPEAK_ENGINE=router` selects it exactly as any other engine is selected.
That is the point of being an entry rather than a service: nothing above had to
learn that routing exists.

# The router owns no data

Not a list of engines, not their addresses, not their voices. Those are facts the
engine images already own, and a second copy drifts in one direction only —
toward a router advertising a voice that no longer exists. It asks
([`elvenspeak.discovery`] for who is running, [`elvenspeak.remote`] for what they
can say) and derives the rest.

[LAW:one-source-of-truth] Its authority is **refusal, not override**. An override
table would make it a second answer about voices the engines own; a refusal
asserts nothing at all — it only declines to proceed where the one source is
ambiguous.

# Why a collision stops the boot

Two backends offering the same voice id is the one thing discovery cannot resolve,
because both answers are equally well-founded and the caller sent one id.
First-registered-wins was considered and rejected: openconv sends a bare voice id,
so the operator would learn about the ambiguity from audio that sounds like the
wrong person rather than from a log.

This is not the same check as the one `tests/test_aliases.py` runs in CI, and
neither replaces the other. CI sees one commit's tables and catches the mistake
before an image exists. Boot sees a running fleet, where a rolling deploy
legitimately mixes image versions and a claim added in a later version can collide
with an older image still serving — which CI structurally cannot see.

A *shared ElevenLabs alias* is deliberately not a collision. Every engine's table
claims the same foreign ids by design, and two engines each offering their own
local substitute for one globally-unique ElevenLabs id is two compatibility
mappings rather than two answers to one question. Only local voice ids — the ids a
caller sends to reach one specific voice — collide.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

from . import discovery, engine
from .provisioning import ConfigError
from .remote import Remote

_LOGGER = logging.getLogger("elvenspeak.router")

#: Where to ask which engines are running. Required and not defaulted: a router
#: is nothing without it, and the obvious guess — the local agent — is right only
#: for a deployment sharing the host's network. Guessing it would turn a bridge
#: network into a router that discovers nothing and reports itself unhealthy,
#: which is a true statement about a false cause.
CONSUL_URL = "ROUTER_CONSUL_URL"

#: The key the engines behind this router are guarded with, if they are. Its own
#: setting and not [`Settings.api_key`], which guards the router's *own* callers:
#: the edge a client authenticates at and the credential the router presents
#: inward are two facts, and one variable meaning both would force a deployment
#: to use the same secret on both sides of itself.
#:
#: Optional, because a fleet reachable only from inside the cluster is the case
#: this was built for. Unset means no header is sent, which is exactly what an
#: unguarded backend expects.
BACKEND_API_KEY = "ROUTER_BACKEND_API_KEY"


@dataclass(frozen=True)
class RouterEngine:
    """The fleet, as the one engine the surface above it thinks it is talking to.

    Satisfies [`elvenspeak.engine.Engine`] in full. Every field was decided at
    [`_Prepared.open`], which is where discovery happened, so nothing here does
    I/O to answer a question about itself — [`voices`] is a read, exactly as it is
    for an engine holding ONNX sessions. What each of those voices can do is on
    the voice ([`elvenspeak.engine.Voice.capabilities`]); this engine holds no
    capability answer of its own and so has none to be wrong with.
    """

    #: Every voice the fleet offers, backends in discovery order and each
    #: backend's voices in its own. The first is load-bearing: it answers unknown
    #: ids in a deployment that named no fallback, so this order is stable rather
    #: than tidy.
    _voices: tuple[engine.Voice, ...]
    #: Which backend speaks each voice. Total over `_voices` by construction —
    #: they are built from one pass in [`_Prepared.open`] — which is why nothing
    #: below checks for a miss.
    _speakers: Mapping[str, Remote]

    def voices(self) -> tuple[engine.Voice, ...]:
        """Every voice the fleet offers, each carrying what its own backend honours.

        There is no fleet-wide capability answer here and that is the whole of
        `piper-routing-7e2.4`: it used to be the *intersection*, which switched
        timestamps off for every voice as soon as one backend could not measure.
        A union would have lied the other way. Each voice arrives from
        [`elvenspeak.remote`] carrying its own backend's answer, so the router
        holds no opinion to be wrong with.
        """
        return self._voices

    def speak(
        self, voice: engine.Voice, text: str, prosody: engine.Prosody
    ) -> engine.Speech:
        return self._speakers[voice.id].speak(voice, text, prosody)

    def speak_timed(
        self, voice: engine.Voice, text: str, prosody: engine.Prosody
    ) -> engine.TimedSpeech:
        return self._speakers[voice.id].speak_timed(voice, text, prosody)


@dataclass(frozen=True)
class _Found:
    """One backend and everything it said about itself, from one look.

    [LAW:types-are-the-program] A record rather than the tuple this started as:
    three parallel positions that every reader has to remember the order of is a
    shape that admits being unpacked wrong, and the names cost nothing.
    """

    remote: Remote
    voices: tuple[engine.Voice, ...]
    #: The engine this backend runs, for the startup log and nothing else. What it
    #: will honour travels on each voice above.
    engine_name: str


def _fleet(consul_url: str, api_key: str | None) -> tuple[_Found, ...]:
    """Every discovered backend, with what it offers and what it calls itself.

    Two requests per backend, with nothing synchronising them: a rolling deploy
    that replaces one in between can pair a stale name with a fresh voice list.
    Tolerable only because the name reaches the startup log and nothing else —
    the voices decide routing and each arrives with its own capabilities from the
    one request, so what is load-bearing is consistent with itself. Anything that
    made the name matter would have to close that gap first.
    """
    return tuple(
        _Found(
            remote=remote,
            voices=remote.voices(),
            engine_name=remote.engine_name(),
        )
        for remote in (
            Remote(backend, api_key) for backend in discovery.engines(consul_url)
        )
    )


def _collisions(offered: Mapping[str, set[str]]) -> list[str]:
    """One message per voice id that more than one *deployment* claims.

    [LAW:types-are-the-program] Deployment, not backend and not engine, and the
    distinction is the whole correctness of this check. A service scaled to two
    allocations is two backends offering identical voices, which is not an
    ambiguity — either replica is the same answer — so counting backends would
    refuse the boot of any fleet that scaled an engine past one instance, which
    [`elvenspeak.discovery`] explicitly supports. Counting *engine names* fails
    the other way: two separate deployments can both run piper with different
    voices, and collapsing them by engine name would hide a real ambiguity.

    A Consul service name is exactly the identity that replicas share and that
    distinct deployments do not, so it is what the set holds — and it is also
    what an operator edits, which is why the message quotes it.
    """
    return [
        f"voice {voice_id!r} is offered by more than one deployment: "
        f"{', '.join(sorted(services))}"
        for voice_id, services in sorted(offered.items())
        if len(services) > 1
    ]


@dataclass(frozen=True)
class _Prepared:
    """The router as this deployment configured it, before it has looked.

    Satisfies [`elvenspeak.provisioning.Prepared`]. Holds only what
    [`configure`] proved, so both methods take no arguments and no unchecked
    string can reach either of them.
    """

    consul_url: str
    #: Presented to every backend as `xi-api-key`. `None` for a fleet that is not
    #: guarded, which sends no header at all.
    backend_api_key: str | None

    def acquire(self) -> tuple[engine.Voice, ...]:
        """Nothing: a router installs no assets.

        The empty tuple is the honest answer here and not a failure — see
        [`elvenspeak.provisioning.Prepared.acquire`], which names this exact case
        ("an engine with nothing to install — a remote API — returns no voices").
        An engine that *does* have assets refuses an empty list while parsing, so
        the two never arrive at the same value.

        This is what makes `python -m elvenspeak.bake` succeed for a router image
        rather than needing a branch that knows routers exist.
        """
        return ()

    def open(self) -> RouterEngine:
        """Looks once, refuses an ambiguous fleet, and returns the engine.

        [LAW:effects-at-boundaries] The moment discovery happens. Everything the
        surface asks afterwards is answered from this snapshot, which is what
        [`elvenspeak.engine.Engine.voices`] requires — a voice offered is one that
        can be spoken *now*, and a router that discovered per request would offer
        voices it had not confirmed.

        A fleet with nothing in it is not refused. A router that found no engine
        has no voices, so [`elvenspeak.api`] answers `/health` 503 and nothing is
        routed to it. Refusing here instead would turn a cluster that is merely
        still starting into a crash loop.

        Discovery happens once, here, so recovery is somebody else's job: an
        empty router does not look again, and the thing that fixes it is being
        restarted. The deployment owns that — home-infra's router jobspec carries
        a `check_restart` on this check, which is what turns "looked too early"
        into "looks again in thirty seconds".

        This paragraph used to name the container's `HEALTHCHECK` as the enforcer.
        It was the only correct rule in the system and the only one nothing ran:
        Nomad's docker driver ignores an image's `HEALTHCHECK`, and the cluster
        read a `/health` that answered 200 unconditionally. On 2026-09-02 a router
        in exactly this state registered as passing and served silence for half an
        hour. Both halves were wrong at once — the check could not fail, and the
        task's `restart` budget answers a process exiting rather than a check
        going red, so it could not have fired either.
        """
        found = _fleet(self.consul_url, self.backend_api_key)

        offered: dict[str, set[str]] = {}
        for backend in found:
            for voice in backend.voices:
                offered.setdefault(voice.id, set()).add(backend.remote.backend.service)

        collisions = _collisions(offered)
        if collisions:
            raise ConfigError(collisions)

        for backend in found:
            _LOGGER.info(
                "routing %d voice(s) to %s at %s",
                len(backend.voices),
                backend.engine_name,
                backend.remote.backend.base_url,
            )

        # One pass, because the voice list and the routing table are two views of
        # one decision: which backend answers for each id. Replicas of a
        # deployment offer identical voices and are interchangeable — the check
        # above is what makes that true — so the first in discovery order wins and
        # the rest add nothing. Taking them separately would let a scaled fleet
        # list the same voice twice, which the conformance suite forbids.
        # Spreading requests across replicas is load balancing: a different job.
        speakers: dict[str, Remote] = {}
        offered_voices: list[engine.Voice] = []
        for backend in found:
            for voice in backend.voices:
                if voice.id not in speakers:
                    speakers[voice.id] = backend.remote
                    offered_voices.append(voice)

        return RouterEngine(
            _voices=tuple(offered_voices),
            _speakers=speakers,
        )


def configure(
    env: "Mapping[str, str]",
    withheld: frozenset[engine.Capability],
    serves: frozenset[str],
) -> _Prepared:
    """Reads the router's own environment, or says everything wrong with it at once.

    [LAW:parse-dont-validate] The checkpoint for this engine. Nothing below holds
    a string out of `env`, and what crosses is a [`_Prepared`] that could not have
    been built before these checks ran.

    `withheld` is accepted and unused, which is the honest answer and not an
    oversight. It exists so an engine can skip building machinery for a capability
    the deployment switched off — Piper's alignment graph is the case it was added
    for — and a router builds no machinery at all: the backends already made
    whatever they made before this process started. The subtraction still happens
    once, in [`elvenspeak.api.create_app`], so a withheld capability is never
    offered no matter what the fleet declared.

    `serves` is accepted and unused for a sharper version of the same reason, and
    it is the argument this engine is the shape of. It is what a deployment
    running *this* engine answers to by name — `{"router"}`, since a router ships
    no declaration file and never will. Stamping it onto the fleet's voices is
    precisely the bug `piper-routing-7e2.17` was filed for: it would republish
    every backend as `router` and refuse `piper` as an engine this deployment is
    not running, while running it. The voices arrive from
    [`elvenspeak.remote`] already carrying their own backend's answer, which is
    the one that can be right ([LAW:one-source-of-truth]), and the union over them
    is what [`elvenspeak.api`] advertises.
    """
    problems: list[str] = []

    consul_url = (env.get(CONSUL_URL) or "").strip().rstrip("/")
    if not consul_url:
        problems.append(
            f"{CONSUL_URL} is required: the router discovers the engines it "
            f"fronts, and has nowhere to ask without it"
        )
    elif not consul_url.startswith(("http://", "https://")):
        problems.append(f"{CONSUL_URL}={consul_url!r} is not an http(s) URL")

    # Empty and unset are one situation — no key to present — so the empty string
    # is not a second spelling of "guarded with nothing".
    backend_api_key = (env.get(BACKEND_API_KEY) or "").strip() or None

    if problems:
        raise ConfigError(problems)
    return _Prepared(consul_url=consul_url, backend_api_key=backend_api_key)
