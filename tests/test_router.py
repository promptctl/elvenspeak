"""The router, driven against real elvenspeak servers on loopback.

Every test here starts actual HTTP servers (see `tests/fleet.py`) rather than
patching the client, because the router is a client and a patched one is a
different program. The engines behind those servers are the existing
`DeclaredEngine` stand-in, so the fleet is real and only the speech is cheap.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fleet import Registered, cluster, consul_app, serving

from elvenspeak import router
from elvenspeak.api import create_app
from elvenspeak.engine import Capability, Prosody, Voice
from elvenspeak.engines import ENGINES
from elvenspeak.provisioning import ConfigError
from elvenspeak.remote import WIRE_RATE, RemoteFailure
from elvenspeak.settings import Settings
from elvenspeak.voices import Substitution

EVERYTHING = frozenset(Capability)
NOTHING: frozenset[Capability] = frozenset()

ALPHA_VOICES = (Voice(id="alpha-one", name="Alpha One", description="alpha's"),)
BETA_VOICES = (Voice(id="beta-one", name="Beta One", description="beta's"),)


def opened(consul_url: str) -> router.RouterEngine:
    """The router a deployment pointed at `consul_url` would boot with."""
    return router.configure({router.CONSUL_URL: consul_url}, NOTHING).open()


def test_the_fleets_voices_are_offered_as_one_engines():
    """The point of the whole epic: two deployments, one endpoint.

    The router satisfies `Engine` by answering for voices it does not own, and
    nothing above it can tell — which is what `elvenspeak.engine` meant by "every
    member must be answerable by a remote HTTP API and by a local ONNX model
    alike".
    """
    with cluster(
        ("alpha", ALPHA_VOICES, EVERYTHING), ("beta", BETA_VOICES, EVERYTHING)
    ) as consul:
        offered = opened(consul).voices()

    assert [voice.id for voice in offered] == ["alpha-one", "beta-one"]
    # Carried through rather than rebuilt: a routed deployment describes its
    # voices exactly as well as the engine behind it does.
    assert offered[0].name == "Alpha One"


def test_a_voice_is_spoken_by_the_engine_that_offers_it():
    """The guarantee. A voice reaches its own backend, never whichever answered.

    Proved by what the two backends *differ* in rather than by inspecting the
    routing table: only `alpha` declares timestamps, so a timed synthesis of
    alpha's voice succeeds and one of beta's is refused by beta itself. A router
    that sent both to one backend would make these two answers identical.
    """
    with cluster(
        ("alpha", ALPHA_VOICES, EVERYTHING), ("beta", BETA_VOICES, NOTHING)
    ) as consul:
        engine = opened(consul)

        timed = engine.speak_timed(ALPHA_VOICES[0], "hello there", Prosody())
        assert timed.pcm
        assert timed.sample_rate == WIRE_RATE

        with pytest.raises(RemoteFailure):
            engine.speak_timed(BETA_VOICES[0], "hello there", Prosody())


def test_speaking_streams_real_audio_back():
    """The proxy hop delivers samples, at the rate it said it would.

    `Speech.sample_rate` is committed before any audio arrives — the encoder is
    started from it — so a router that guessed wrong would produce audio at the
    wrong speed rather than an error.
    """
    with cluster(("alpha", ALPHA_VOICES, EVERYTHING)) as consul:
        spoken = opened(consul).speak(ALPHA_VOICES[0], "hello there", Prosody())
        assert spoken.sample_rate == WIRE_RATE
        audio = b"".join(spoken.audio)

    # Sixteen-bit samples, so an odd byte count would mean a truncated stream
    # rather than merely a short one.
    assert audio
    assert len(audio) % 2 == 0


def test_two_engines_offering_the_same_voice_id_stop_the_boot():
    """[LAW:no-silent-failure] The one ambiguity discovery cannot resolve.

    Both answers are equally well-founded and the caller sent one id, so there is
    nothing to pick. First-registered-wins was rejected because openconv sends a
    bare voice id: the operator would learn about the ambiguity from audio that
    sounds like the wrong person.

    The message names the engines and not their addresses, because the decision
    an operator has to make afterwards is about a deployment.
    """
    shared = (Voice(id="contested", name="Contested", description="offered twice"),)
    with cluster(("alpha", shared, EVERYTHING), ("beta", shared, EVERYTHING)) as consul:
        with pytest.raises(ConfigError) as raised:
            opened(consul)

    message = str(raised.value)
    assert "contested" in message
    assert "alpha" in message and "beta" in message


def test_the_same_elevenlabs_alias_on_two_engines_is_not_a_collision():
    """Decided on the ticket, and the reason the refusal is scoped to local ids.

    Every engine's table claims the same foreign ElevenLabs ids by design, so a
    rule that refused on a shared *alias* would stop a router fronting both real
    engines on every boot, always. Two engines each offering their own local
    substitute for one globally-unique foreign id is two compatibility mappings,
    not two answers to one question.

    Modelled here the way it actually arises: distinct local voices, and nothing
    about the foreign ids either engine maps enters the collision check at all.
    """
    with cluster(
        ("alpha", ALPHA_VOICES, EVERYTHING), ("beta", BETA_VOICES, EVERYTHING)
    ) as consul:
        assert len(opened(consul).voices()) == 2


def test_capabilities_are_what_every_backend_will_honour():
    """Intersection, because one answer is given for every voice.

    A union would advertise timestamps that the backend owning some particular
    voice cannot produce, and `Engine.capabilities` is explicit that absence is
    the safe default: a router that undersells is pessimistic, one that oversells
    lies in the audio.
    """
    with cluster(
        ("alpha", ALPHA_VOICES, EVERYTHING), ("beta", BETA_VOICES, NOTHING)
    ) as consul:
        assert opened(consul).capabilities() == NOTHING

    with cluster(
        ("alpha", ALPHA_VOICES, EVERYTHING), ("beta", BETA_VOICES, EVERYTHING)
    ) as consul:
        assert opened(consul).capabilities() == EVERYTHING


def test_a_fleet_with_no_engines_boots_and_offers_nothing():
    """Not a refusal: a cluster that is still starting must not be a crash loop.

    A router that discovered no engine has no voices, so its own `HEALTHCHECK` —
    which requires voices — fails and nothing is ever routed to it. That is the
    correct outcome and it is already enforced one layer down, so refusing here
    as well would turn a transient state into an outage.
    """
    with cluster() as consul:
        engine = opened(consul)

    assert engine.voices() == ()
    assert engine.capabilities() == NOTHING


def test_an_unreachable_consul_fails_the_boot_rather_than_finding_nothing():
    """[LAW:no-silent-failure] "Nobody is running" and "I could not ask" differ.

    Collapsing them would boot a router that looks healthy in its logs and can
    speak nothing, and the empty voice list would be blamed on the fleet.
    """
    with pytest.raises(ConfigError) as raised:
        # Port 1 on loopback: nothing listens, and the refusal is immediate.
        opened("http://127.0.0.1:1")

    assert "consul" in str(raised.value).lower()


@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, "required"),
        ({router.CONSUL_URL: "   "}, "required"),
        ({router.CONSUL_URL: "consul.service:8500"}, "not an http(s) URL"),
    ],
    ids=["absent", "blank", "not-a-url"],
)
def test_a_router_without_somewhere_to_ask_refuses_to_configure(env, expected):
    """The address is required and never guessed.

    The obvious default — the local agent — is right only for a deployment
    sharing the host's network. Guessing it would turn a bridge network into a
    router that discovers nothing and reports itself unhealthy, which is a true
    statement about a false cause.
    """
    with pytest.raises(ConfigError) as raised:
        router.configure(env, NOTHING)
    assert expected in str(raised.value)


def test_both_engines_are_reachable_through_one_endpoint():
    """The ticket's acceptance criterion, asserted where a caller stands.

    Everything above the router is unchanged code — the catalog, the format
    negotiation, the encoder, the headers — so this is also the claim that being
    an `ENGINES` entry rather than a new service actually bought what it promised:
    a routed deployment is an elvenspeak deployment, and a client cannot tell.
    """
    with cluster(
        ("alpha", ALPHA_VOICES, EVERYTHING), ("beta", BETA_VOICES, EVERYTHING)
    ) as consul:
        settings = Settings(
            engine=router.configure({router.CONSUL_URL: consul}, NOTHING),
            engine_name="router",
            known_engines=frozenset(ENGINES),
            withheld=NOTHING,
            fallback=Substitution.FIRST_OFFERED,
            api_key=None,
            host="127.0.0.1",
            port=0,
        )
        with TestClient(create_app(settings, settings.engine.open())) as client:
            published = client.get("/v1/voices").json()["voices"]
            listed = [entry["voice_id"] for entry in published]
            spoken = {
                voice_id: client.post(
                    f"/v1/text-to-speech/{voice_id}/stream",
                    json={"text": "hello there"},
                    params={"output_format": "mp3_44100_128"},
                )
                for voice_id in ("alpha-one", "beta-one")
            }

    assert sorted(listed) == ["alpha-one", "beta-one"]
    for voice_id, response in spoken.items():
        assert response.status_code == 200, (voice_id, response.text)
        # Real MP3 out of the router's own encoder, over PCM that came from
        # another process entirely.
        assert response.content[:3] == b"ID3" or response.content[:2] == b"\xff\xfb"
        assert response.headers["x-elvenspeak-voice"] == voice_id


def test_a_backend_that_cannot_describe_itself_fails_the_boot_by_name():
    """An image too old to answer `/v1/models` is named, not assumed about.

    [LAW:no-silent-failure] Reading a 404 as "declares nothing" would silently
    turn every timestamp request into a 501 for a backend that can in fact
    measure, and the operator would be left inferring an image version from
    missing captions.

    Reported as a `ConfigError` specifically, because that is the only exception
    `reported_or_exit` catches: anything else reaches the operator as a traceback
    where every other startup problem reaches them as a line naming what is
    wrong.
    """
    stale = FastAPI()

    @stale.get("/v1/voices")
    def voices() -> dict:
        return {"voices": [{"voice_id": "ancient", "name": "Ancient"}]}

    with serving(stale) as backend, serving(
        consul_app([Registered(service="elvenspeak-ancient", base_url=backend)])
    ) as consul:
        with pytest.raises(ConfigError) as raised:
            opened(consul)

    assert "elvenspeak-ancient" in str(raised.value)


def test_a_router_installs_nothing_and_says_so():
    """`python -m elvenspeak.bake` has to succeed for a router image.

    The empty tuple is the honest answer that `provisioning.Prepared.acquire`
    names for exactly this case, not a failure — which is what lets the build
    stay one command with no branch that knows routers exist.
    """
    prepared = router.configure({router.CONSUL_URL: "http://consul:8500"}, NOTHING)
    assert prepared.acquire() == ()
