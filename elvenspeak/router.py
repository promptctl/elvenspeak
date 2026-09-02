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
from .remote import Description, Remote

_LOGGER = logging.getLogger("elvenspeak.router")

#: Where to ask which engines are running. Required and not defaulted: a router
#: is nothing without it, and the obvious guess — the local agent — is right only
#: for a deployment sharing the host's network. Guessing it would turn a bridge
#: network into a router that discovers nothing and reports itself unhealthy,
#: which is a true statement about a false cause.
CONSUL_URL = "ROUTER_CONSUL_URL"


@dataclass(frozen=True)
class RouterEngine:
    """The fleet, as the one engine the surface above it thinks it is talking to.

    Satisfies [`elvenspeak.engine.Engine`] in full. Every field was decided at
    [`_Prepared.open`], which is where discovery happened, so nothing here does
    I/O to answer a question about itself — [`voices`] and [`capabilities`] are
    reads, exactly as they are for an engine holding ONNX sessions.
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
    _capabilities: frozenset[engine.Capability]

    def voices(self) -> tuple[engine.Voice, ...]:
        return self._voices

    def capabilities(self) -> frozenset[engine.Capability]:
        """What every backend in the fleet will honour — the intersection.

        Intersection and not union, because this one answer is given for every
        voice ([`elvenspeak.engine.Engine.capabilities`] is constant for the
        engine's life), and a union would advertise timestamps that the backend
        owning some particular voice cannot produce. Absence is the safe default
        there for exactly this reason: a router that undersells is pessimistic,
        one that oversells lies in the audio.

        The cost is real — one backend without timestamps switches them off for
        the whole fleet — and it is the honest price of a fleet-wide answer.
        `piper-routing-7e2.4` makes capability a per-voice question, and this
        becomes a union of what each voice's own backend declares.
        """
        return self._capabilities

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
    description: Description


def _fleet(consul_url: str) -> tuple[_Found, ...]:
    """Every discovered backend, with what it offers and what it calls itself.

    Each backend is asked each question exactly once, and the answers are kept
    together: a name taken from one request and capabilities from another could
    describe two different moments of a rolling deploy, and the pair would be
    reported as one engine that never existed.
    """
    return tuple(
        _Found(remote=remote, voices=remote.voices(), description=remote.describe())
        for remote in (Remote(backend) for backend in discovery.engines(consul_url))
    )


def _collisions(
    offered: Mapping[str, list[str]],
) -> list[str]:
    """One message per voice id that more than one backend claims.

    Named by engine rather than by address: an operator reading this has to
    decide which deployment to change, and `piper` and `kokoro` are the words
    that decision is made in.
    """
    return [
        f"voice {voice_id!r} is offered by more than one engine: "
        f"{', '.join(sorted(claimants))}"
        for voice_id, claimants in sorted(offered.items())
        if len(claimants) > 1
    ]


@dataclass(frozen=True)
class _Prepared:
    """The router as this deployment configured it, before it has looked.

    Satisfies [`elvenspeak.provisioning.Prepared`]. Holds only what
    [`configure`] proved, so both methods take no arguments and no unchecked
    string can reach either of them.
    """

    consul_url: str

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
        has no voices, fails its own healthcheck, and is therefore never routed
        to — which is the correct outcome and is already what the container's
        `HEALTHCHECK` enforces. Refusing here instead would turn a cluster that is
        merely still starting into a crash loop.
        """
        found = _fleet(self.consul_url)

        offered: dict[str, list[str]] = {}
        for backend in found:
            for voice in backend.voices:
                offered.setdefault(voice.id, []).append(backend.description.engine_name)

        collisions = _collisions(offered)
        if collisions:
            raise ConfigError(collisions)

        for backend in found:
            _LOGGER.info(
                "routing %d voice(s) to %s at %s",
                len(backend.voices),
                backend.description.engine_name,
                backend.remote.backend.base_url,
            )

        return RouterEngine(
            _voices=tuple(voice for backend in found for voice in backend.voices),
            _speakers={
                voice.id: backend.remote
                for backend in found
                for voice in backend.voices
            },
            # An empty fleet has no intersection to take, and `frozenset` is the
            # right answer rather than a special case: a router with no backends
            # honours nothing, which is exactly what it can do.
            _capabilities=frozenset.intersection(
                *(backend.description.capabilities for backend in found)
            )
            if found
            else frozenset(),
        )


def configure(
    env: "Mapping[str, str]", withheld: frozenset[engine.Capability]
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

    if problems:
        raise ConfigError(problems)
    return _Prepared(consul_url=consul_url)
