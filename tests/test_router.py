"""The router, driven against real elvenspeak servers on loopback.

Every test here starts actual HTTP servers (see `tests/fleet.py`) rather than
patching the client, because the router is a client and a patched one is a
different program. The engines behind those servers are the existing
`DeclaredEngine` stand-in, so the fleet is real and only the speech is cheap.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace

import pytest
from fastapi import FastAPI, Response
from fastapi.testclient import TestClient
from fleet import Registered, cluster, engine_app, registered_consul, serving

from conftest import SERVES

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

#: Each stand-in backend answers to its own engine name and to one foreign id, so
#: a test can tell "reached alpha" from "reached whichever backend answered" — the
#: distinction `piper-routing-7e2.17` is about. Alpha and beta share none of them,
#: which is what makes a router that dropped the engine axis visible here.
ALPHA_MODELS = frozenset({"alpha", "eleven_alpha_v1"})
BETA_MODELS = frozenset({"beta", "eleven_beta_v1"})

ALPHA_VOICES = (
    Voice(
        id="alpha-one",
        name="Alpha One",
        description="alpha's",
        models=ALPHA_MODELS,
        language="en",
    ),
)
BETA_VOICES = (
    Voice(
        id="beta-one",
        name="Beta One",
        description="beta's",
        models=BETA_MODELS,
        language="en",
    ),
)


def opened(consul_url: str) -> router.RouterEngine:
    """The router a deployment pointed at `consul_url` would boot with."""
    return router.configure({router.CONSUL_URL: consul_url}, NOTHING, SERVES).open()


def routed(consul_url: str) -> TestClient:
    """A client on the whole server a routed deployment boots, not just its engine.

    [LAW:one-source-of-truth] Written out once. Three tests need the same
    `Settings`, and the copies were already drifting toward being edited
    separately — a router's deployment settings are one fact about this project,
    not one per test.
    """
    settings = Settings(
        engine=router.configure({router.CONSUL_URL: consul_url}, NOTHING, SERVES),
        engine_name="router",
        known_engines=frozenset(ENGINES),
        withheld=NOTHING,
        fallback=Substitution.FIRST_OFFERED,
        api_key=None,
        host="127.0.0.1",
        port=0,
    )
    return TestClient(create_app(settings, settings.engine.open()))


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

    The message names the deployments, because that is what an operator edits —
    and two deployments can run the same engine, so the engine's name alone would
    not tell them which job file to open.

    Scoped to *local* voice ids. A shared ElevenLabs alias is deliberately not a
    collision: every engine's table claims the same foreign ids by design, so
    refusing on those would stop a router fronting both real engines on every
    boot, always. Two engines each offering their own substitute for one
    globally-unique foreign id is two compatibility mappings, not two answers to
    one question. Nothing alias-shaped can reach this check today — the router
    does not read backends' alias tables at all (`piper-routing-7e2.15`) — so the
    rule is recorded here rather than as a test that could only assert the
    absence of a code path.
    """
    # One engine name for both, which is what two deployments running the same
    # engine really report — and why the message below names services rather than
    # engines.
    shared = (
        Voice(
            id="contested",
            name="Contested",
            description="offered twice",
            models=frozenset({"shared-engine"}),
            language="en",
        ),
    )
    with cluster(("alpha", shared, EVERYTHING), ("beta", shared, EVERYTHING)) as consul:
        with pytest.raises(ConfigError) as raised:
            opened(consul)

    message = str(raised.value)
    assert "contested" in message
    assert "elvenspeak-alpha" in message and "elvenspeak-beta" in message


def test_a_deployment_scaled_to_two_replicas_is_not_a_collision():
    """Replicas offer the same ids and are the same answer, not two answers.

    `discovery` returns every healthy instance of a service on purpose, so a
    fleet that scaled an engine past one allocation would hand the collision
    check two claimants for every voice. Counting *backends* refuses that boot;
    counting deployments — which is what a Consul service name identifies —
    correctly sees one.

    The voice is also offered once rather than twice: the conformance suite
    forbids two voices sharing an id, and a scaled fleet must not be the thing
    that breaks it.
    """
    with cluster(("piper", ALPHA_VOICES, EVERYTHING), replicas=2) as consul:
        engine = opened(consul)
        assert [voice.id for voice in engine.voices()] == ["alpha-one"]

        # Drained inside the fleet's lifetime, because `speak` is lazy: it opens
        # nothing until the generator is pulled, so a call whose audio is never
        # read reaches no backend and proves no routing. Asserted after the
        # servers had stopped, this passed while proving nothing at all.
        spoken = engine.speak(ALPHA_VOICES[0], "hello", Prosody())
        assert spoken.sample_rate == WIRE_RATE
        assert b"".join(spoken.audio)


def test_a_replica_still_loading_its_voices_is_not_routed_to():
    """The whole path, against a deployment mid-rollout.

    A replica that is registered but failing its check has not finished opening
    its voices, so routing to it would 503 every clause of a conversation. The
    filter that excludes it is one query parameter in `discovery`, and until this
    test the router's own `open()` was never driven against a fleet where some
    instances were unhealthy — `test_discovery` proved the parameter is honoured,
    but nothing proved the router ends up with only the servers that can speak.

    The unhealthy instance is a real server here, so a router that ignored the
    filter would boot successfully and quietly include it, which is exactly the
    failure this must not pass through.
    """
    with ExitStack() as running:
        healthy = running.enter_context(
            serving(engine_app("piper", ALPHA_VOICES, EVERYTHING))
        )
        loading = running.enter_context(
            serving(engine_app("piper", BETA_VOICES, EVERYTHING))
        )
        consul = running.enter_context(
            serving(
                registered_consul(
                    [
                        Registered(service="elvenspeak-piper", base_url=healthy),
                        Registered(
                            service="elvenspeak-piper",
                            base_url=loading,
                            passing=False,
                        ),
                    ]
                )
            )
        )
        engine = opened(consul)
        assert [voice.id for voice in engine.voices()] == ["alpha-one"]
        assert b"".join(engine.speak(ALPHA_VOICES[0], "hello", Prosody()).audio)


def test_each_voice_carries_what_its_own_backend_will_honour():
    """The point of `piper-routing-7e2.4`, and what a router is for.

    This was the intersection of the fleet's, because one answer was given for
    every voice — so a single backend that could not measure switched timestamps
    off for every voice in the deployment. A union would have lied the other way,
    promising captions for voices that cannot produce them. Neither is true, and
    per-voice is: alpha's voice measures because alpha does, and beta's does not
    because beta does not, in one process at the same time.
    """
    with cluster(
        ("alpha", ALPHA_VOICES, EVERYTHING), ("beta", BETA_VOICES, NOTHING)
    ) as consul:
        offered = {voice.id: voice.capabilities for voice in opened(consul).voices()}

    assert offered["alpha-one"] == EVERYTHING
    assert offered["beta-one"] == NOTHING


def test_each_voice_carries_the_language_its_own_backend_speaks():
    """The language axis across a fleet, which every fixture in this file hides.

    Every voice here says `en`, so a router that read one backend's language for
    all of its voices, or collapsed the fleet onto a single value, passes the rest
    of the file. Built the way the capability test above is built — the two
    backends made deliberately to disagree — because agreement is what a bug of
    this shape looks like from the outside.

    The fleet is the shape a Spanish deployment actually takes: one backend baking
    English voices, another baking Spanish ones, behind one endpoint. If the
    languages merged into one value, `Catalog.speaking` narrows onto the wrong
    half of the fleet and a Spanish request is answered in English — the failure
    this epic exists to end, arriving through the router instead of through a
    missing field.
    """
    hablante = replace(BETA_VOICES[0], id="beta-uno", language="es")

    with cluster(
        ("alpha", ALPHA_VOICES, EVERYTHING), ("beta", (hablante,), EVERYTHING)
    ) as consul:
        spoken = {voice.id: voice.language for voice in opened(consul).voices()}

    assert spoken == {"alpha-one": "en", "beta-uno": "es"}


def test_a_language_steers_a_request_to_the_backend_that_speaks_it():
    """The same fleet, through the surface a caller actually reaches.

    The listing above proves the languages survive the merge; this proves they
    steer. A router that carried the field into `GET /v1/voices` and then resolved
    without it would pass that test and answer this request in English — which is
    exactly the split the whole epic is about, since nothing audible says which
    happened.

    The caller names alpha's English voice and asks for Spanish, so the language
    has to outrank the id *and* cross a backend boundary to be honoured. The two
    headers together are the proof: one says a Spanish voice spoke, the other says
    nothing was dropped to make that happen.
    """
    hablante = replace(BETA_VOICES[0], id="beta-uno", language="es")

    with cluster(
        ("alpha", ALPHA_VOICES, EVERYTHING), ("beta", (hablante,), EVERYTHING)
    ) as consul:
        response = routed(consul).post(
            "/v1/text-to-speech/alpha-one/stream",
            json={"text": "hola", "language_code": "es"},
            params={"output_format": "pcm_22050"},
        )

    assert response.status_code == 200, response.text
    assert response.headers["x-elvenspeak-voice"] == "beta-uno"
    assert response.headers["x-elvenspeak-voice-requested"] == "alpha-one"
    assert "language_code" not in response.headers.get("x-elvenspeak-ignored", "")


def test_a_fleet_with_no_engines_boots_and_offers_nothing():
    """Not a refusal: a cluster that is still starting must not be a crash loop.

    A router that discovered no engine has no voices and says so at `/health`,
    which is what keeps it out of rotation until it finds them — see the test
    below, which is the other half of this one and the reason refusing here would
    turn a transient state into an outage.

    This docstring used to say that outcome was "already enforced one layer
    down", meaning the image's `HEALTHCHECK`. It was, and that layer is not the
    one a Nomad deployment reads: on 2026-09-02 this exact state registered in
    Consul as passing, because the cluster's check asked `/health` for a status
    code and got 200. The claim was true about the wrong enforcer, and no test
    covered the one that mattered.
    """
    with cluster() as consul:
        engine = opened(consul)

    assert engine.voices() == ()


def test_a_router_that_found_no_engines_reports_itself_unfit_at_health():
    """The half of the boot-ordering story that a cluster actually reads.

    [LAW:no-silent-failure] `{"voices": []}` behind a 200 is an answer-shaped
    void: "I can speak nothing" arrives in the same shape as "I am fine", and the
    load balancer believes the shape. This is the regression test for the
    2026-09-02 deploy, where all three allocations started in the same second,
    the router looked before either engine had registered, and Consul marked it
    passing — so it stayed in rotation, mute, and its thirty-attempt restart
    budget never spent an attempt because nothing ever failed.

    Asserted through the whole app rather than on the engine, because the defect
    was never in discovery: `voices() == ()` was correct and the test above
    already covered it. What was missing was the server saying so where a checker
    looks.
    """
    with cluster() as consul, routed(consul) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"voices": []}


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
        router.configure(env, NOTHING, SERVES)
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
        with routed(consul) as client:
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
        # Its voices are well-formed and declare their capabilities and their model
        # ids; what it lacks is `/v1/models`. Otherwise the boot fails at the voice
        # check first and this stops testing the thing it names.
        return {
            "voices": [
                {
                    "voice_id": "ancient",
                    "name": "Ancient",
                    "capabilities": ["speed"],
                    "models": ["ancient"],
                    "language": "en",
                }
            ]
        }

    with serving(stale) as backend, serving(
        registered_consul([Registered(service="elvenspeak-ancient", base_url=backend)])
    ) as consul:
        with pytest.raises(ConfigError) as raised:
            opened(consul)

    message = str(raised.value)
    assert "elvenspeak-ancient" in message
    # Pinned to the endpoint this test is named for. The voices check runs first,
    # so a fixture whose voices were also malformed would fail here for the wrong
    # reason and leave the `/v1/models` path uncovered — which is what happened
    # when per-voice capabilities became mandatory.
    assert "what engine it runs" in message


def test_a_backend_that_dies_mid_answer_fails_the_boot_by_name():
    """[LAW:no-silent-failure] The failure is at the transfer, not the connect.

    A backend that accepts the connection, sends its headers and then dies raises
    `ConnectionResetError` or `IncompleteRead` — neither of which is a
    `URLError`, so catching only that pair left `_asked`'s documented guarantee
    ("every way this can go wrong becomes a `ConfigError`") false, and a fleet in
    the middle of a rolling restart could crash the boot with a traceback.

    Modelled by a server that promises more body than it sends, which is exactly
    what a process killed mid-response leaves on the wire.
    """
    truncating = FastAPI()

    @truncating.get("/v1/voices")
    def voices():
        return Response(
            content=b'{"voices": [',
            media_type="application/json",
            headers={"content-length": "4096"},
        )

    with serving(truncating) as backend, serving(
        registered_consul([Registered(service="elvenspeak-broken", base_url=backend)])
    ) as consul:
        with pytest.raises(ConfigError) as raised:
            opened(consul)

    assert "elvenspeak-broken" in str(raised.value)


def test_a_router_can_front_a_guarded_fleet_and_says_so_when_it_cannot():
    """`ELVENSPEAK_API_KEY` on a backend is the ordinary thing, not an exception.

    Every route a router calls sits behind that check, so without a credential to
    present, a router in front of a locked-down fleet 401s on its own boot. The
    key it sends inward is its own setting rather than `ELVENSPEAK_API_KEY`, which
    guards the router's *callers* — one variable meaning both would force a
    deployment to use one secret on both sides of itself.

    Both directions asserted, because the interesting half is the failure: a
    router given the wrong key has to say which backend refused it rather than
    boot into a fleet it cannot speak to.
    """
    with cluster(("alpha", ALPHA_VOICES, EVERYTHING), api_key="s3cret") as consul:
        carrying = router.configure(
            {router.CONSUL_URL: consul, router.BACKEND_API_KEY: "s3cret"},
            NOTHING,
            SERVES,
        ).open()
        assert [voice.id for voice in carrying.voices()] == ["alpha-one"]
        # Drained, so the guarded `/stream` request is actually made. `voices()`
        # above proves the key reaches the boot path; only pulling the generator
        # proves it reaches the synthesis path, which sends its own header.
        assert b"".join(carrying.speak(ALPHA_VOICES[0], "hello", Prosody()).audio)

        with pytest.raises(ConfigError) as raised:
            router.configure({router.CONSUL_URL: consul}, NOTHING, SERVES).open()

    assert "elvenspeak-alpha" in str(raised.value)


def test_a_router_installs_nothing_and_says_so():
    """`python -m elvenspeak.bake` has to succeed for a router image.

    The empty tuple is the honest answer that `provisioning.Prepared.acquire`
    names for exactly this case, not a failure — which is what lets the build
    stay one command with no branch that knows routers exist.
    """
    prepared = router.configure({router.CONSUL_URL: "http://consul:8500"}, NOTHING, SERVES)
    assert prepared.acquire() == ()


@pytest.mark.parametrize(
    "alignment",
    [
        '{"characters": ["a"], "character_end_times_seconds": [Infinity]}',
        '{"characters": ["a"], "character_end_times_seconds": [1e305]}',
        '{"characters": ["a"], "character_end_times_seconds": ["soon"]}',
    ],
    ids=["infinite", "overflows-the-multiply", "not-a-number"],
)
def test_a_timestamp_that_is_not_a_usable_number_fails_the_request(alignment):
    """[LAW:no-silent-failure] Every unusable payload arrives as one named failure.

    `round()` answers `OverflowError` for a non-finite value, which is neither a
    `ValueError` nor a `TypeError` — so an infinity reached the caller as a raw
    traceback out of a live synthesis rather than as the `RemoteFailure` this path
    exists to raise. Both routes to one are covered: a literal `Infinity`, which
    `json` parses without complaint as a non-standard extension, and an ordinary
    huge value that overflows silently when multiplied by the sample rate.

    The body is written as bytes rather than returned as a dict, and that is
    load-bearing: FastAPI refuses to serialise a float infinity and answers 500,
    so a stub built the ordinary way would never put the token on the wire and
    this test would pass by exercising the unreachable-backend path instead —
    green, and about something else entirely.
    """
    backend = FastAPI()

    @backend.get("/v1/models")
    def models():
        return [{"model_id": "odd", "capabilities": ["speed", "timestamps"]}]

    @backend.get("/v1/voices")
    def voices():
        return {
            "voices": [
                {
                    "voice_id": "alpha-one",
                    "name": "Alpha One",
                    "capabilities": ["speed", "timestamps"],
                    "models": ["odd"],
                    "language": "en",
                }
            ]
        }

    @backend.post("/v1/text-to-speech/{voice_id}/with-timestamps")
    def timed(voice_id: str):
        return Response(
            content=(
                '{"audio_base64": "AAAA", "alignment": '
                + alignment
                + ', "alignment_fidelity": "word-exact"}'
            ),
            media_type="application/json",
        )

    with serving(backend) as served, serving(
        registered_consul([Registered(service="elvenspeak-odd", base_url=served)])
    ) as consul:
        engine = opened(consul)
        with pytest.raises(RemoteFailure):
            engine.speak_timed(ALPHA_VOICES[0], "hello", Prosody())


def test_one_deployment_tells_the_truth_about_each_of_its_voices():
    """The ticket's acceptance criterion, asserted where a caller stands.

    One process, two voices, two different answers — in both places the server
    answers about capability. Before this, a deployment had one set for
    everything it served and had to choose which of its voices to lie about:
    refuse `/with-timestamps` for alpha's voice, which works, or promise it for
    beta's, which cannot.

    Both halves are checked because they are two readings of one fact and a fix
    that corrected only the gate would leave the header telling callers the
    opposite of what the endpoint does.
    """
    with cluster(
        ("alpha", ALPHA_VOICES, EVERYTHING), ("beta", BETA_VOICES, NOTHING)
    ) as consul:
        with routed(consul) as client:
            timed = {
                voice_id: client.post(
                    f"/v1/text-to-speech/{voice_id}/with-timestamps",
                    json={"text": "hello there"},
                )
                for voice_id in ("alpha-one", "beta-one")
            }
            spoken = {
                voice_id: client.post(
                    f"/v1/text-to-speech/{voice_id}/stream",
                    json={"text": "hello", "voice_settings": {"speed": 2.0}},
                )
                for voice_id in ("alpha-one", "beta-one")
            }
            listed = {
                entry["voice_id"]: entry["capabilities"]
                for entry in client.get("/v1/voices").json()["voices"]
            }

    # The 501 gate: alpha measures, beta does not, in the same process.
    assert timed["alpha-one"].status_code == 200, timed["alpha-one"].text
    assert timed["beta-one"].status_code == 501, timed["beta-one"].text

    # The ignored header, which has to agree with the gate rather than with some
    # deployment-wide average.
    assert "voice_settings.speed" not in spoken["alpha-one"].headers.get(
        "x-elvenspeak-ignored", ""
    )
    assert "voice_settings.speed" in spoken["beta-one"].headers["x-elvenspeak-ignored"]

    # And a caller can read it before calling, per voice, rather than discovering
    # it from a 501 partway through a conversation.
    assert listed["alpha-one"] == ["speed", "timestamps"]
    assert listed["beta-one"] == []


def test_a_backend_whose_voices_declare_no_capabilities_fails_the_boot():
    """The path a rolling deploy actually takes past an older image.

    A backend from before per-voice capabilities publishes voices with no
    `capabilities` field, and a router cannot tell what it will honour — reading
    the absence as "declares nothing" would refuse every timestamp request for a
    backend that can in fact measure, and reading it as "declares everything"
    would promise captions that never arrive. Neither is knowable from here, so
    the boot stops and names the backend.

    Its own test rather than a case of the stale-backend one, which publishes
    capabilities precisely so it reaches the `/v1/models` check instead.
    """
    ancient = FastAPI()

    @ancient.get("/v1/models")
    def models():
        return [{"model_id": "ancient", "capabilities": ["speed"]}]

    @ancient.get("/v1/voices")
    def voices() -> dict:
        return {"voices": [{"voice_id": "old-one", "name": "Old One"}]}

    with serving(ancient) as backend, serving(
        registered_consul([Registered(service="elvenspeak-ancient", base_url=backend)])
    ) as consul:
        with pytest.raises(ConfigError) as raised:
            opened(consul)

    message = str(raised.value)
    assert "elvenspeak-ancient" in message
    assert "old-one" in message
    assert "named no capabilities" in message


@pytest.mark.parametrize(
    "published, why",
    [
        ({}, "omits the field"),
        ({"models": []}, "publishes an empty list"),
        ({"models": ["piper", None]}, "publishes a non-string entry"),
        ({"models": [""]}, "publishes an empty id"),
    ],
    ids=["absent", "empty", "null-entry", "empty-id"],
)
def test_a_backend_that_names_no_model_ids_fails_the_boot(published, why):
    """The rolling deploy this change's own hazard is about.

    A backend from before `piper-routing-7e2.17` publishes voices carrying
    `capabilities` and no `models` — new enough to survive the check above, old
    enough to say nothing about which engine answers for it. The router cannot
    derive that: the ids belong to the backend, which is the whole reason the
    field is on the voice, so the boot stops rather than serving a fleet whose
    engine axis silently covers fewer backends than it appears to.

    The empty list is here because it is the case that reads as an answer. It
    survives an `isinstance` check and an `all()` over nothing, and the voice it
    produced answered to no model id at all — so `Directory.over` still found that
    engine's name in the union via the backend's *other* voices, and a caller who
    named the engine actually about to speak was told it was not.
    """
    stale = FastAPI()

    @stale.get("/v1/models")
    def models():
        return [{"model_id": "stale", "capabilities": ["speed"]}]

    @stale.get("/v1/voices")
    def voices() -> dict:
        return {
            "voices": [
                {
                    "voice_id": "stale-one",
                    "name": "Stale One",
                    "capabilities": ["speed"],
                }
                | published
            ]
        }

    with serving(stale) as backend, serving(
        registered_consul([Registered(service="elvenspeak-stale", base_url=backend)])
    ) as consul:
        with pytest.raises(ConfigError) as raised:
            opened(consul)

    message = str(raised.value)
    assert "elvenspeak-stale" in message, why
    assert "stale-one" in message, why
    assert "named no models" in message, why


@pytest.mark.parametrize(
    "published, why",
    [
        ({}, "omits the field"),
        ({"language": ""}, "publishes an empty code"),
        ({"language": None}, "publishes a null"),
    ],
    ids=["absent", "empty", "null"],
)
def test_a_backend_that_names_no_language_fails_the_boot(published, why):
    """The rolling deploy `piper-language-j1c.2` opens, held to the `models` bar.

    A backend from before per-voice languages publishes voices carrying
    `capabilities` and `models` — new enough to pass both checks above — and says
    nothing about what its voices speak. There is no honest default: read as `""`,
    the voice matches no caller's `language_code`, so the router narrows every
    Spanish request onto an empty table, falls back to the whole fleet, and
    reports `language_code` ignored for a backend that may well have baked the
    Spanish voice this epic exists to reach. Silent, audible only as English
    phonemes over Spanish words, and the same class of wrong answer the `models`
    check refuses — so the boot stops here too, naming the backend.
    """
    mute = FastAPI()

    @mute.get("/v1/models")
    def models():
        return [{"model_id": "mute", "capabilities": ["speed"]}]

    @mute.get("/v1/voices")
    def voices() -> dict:
        return {
            "voices": [
                {
                    "voice_id": "mute-one",
                    "name": "Mute One",
                    "capabilities": ["speed"],
                    "models": ["mute"],
                }
                | published
            ]
        }

    with serving(mute) as backend, serving(
        registered_consul([Registered(service="elvenspeak-mute", base_url=backend)])
    ) as consul:
        with pytest.raises(ConfigError) as raised:
            opened(consul)

    message = str(raised.value)
    assert "elvenspeak-mute" in message, why
    assert "mute-one" in message, why
    assert "named no language" in message, why


# ------------------------------------------------- the engine axis, end to end
#
# `piper-routing-7e2.17`, whose symptoms were all measured against the running
# router at 2026.09.02.4 rather than inferred: `/v1/models` answered `["router"]`,
# every `eleven_*` id its own backends honour came back in `x-elvenspeak-ignored`,
# and `model_id=piper` was refused as "an engine this deployment is not running"
# by the router that was running it. Voice routing was sound throughout, which is
# why these are their own tests rather than an amendment to the ones above.


def test_the_router_advertises_the_engines_it_fronts_rather_than_itself():
    """The listing is the union over the voices, so it names the fleet.

    A caller cannot choose an engine it cannot discover, and the router used to
    publish only the name of the image it was built from — a name that declares
    nothing and speaks nothing. Both backends' own names and both of their foreign
    ids appear; `router` does not, because no voice answers to it.
    """
    with cluster(
        ("alpha", ALPHA_VOICES, EVERYTHING), ("beta", BETA_VOICES, EVERYTHING)
    ) as consul:
        with routed(consul) as client:
            listing = client.get("/v1/models").json()

    listed = [entry["model_id"] for entry in listing]
    assert listed == ["alpha", "beta", "eleven_alpha_v1", "eleven_beta_v1"]
    assert "router" not in listed


def test_a_voice_reports_the_engine_that_will_speak_it():
    """`GET /v1/voices` carries the answer the listing above is derived from.

    The round trip that makes the whole thing possible: a backend publishes what
    it answers to per voice, the router reads it in the same response that brought
    the voices, and republishes it. Asserted here because a router that derived
    the listing correctly while dropping the field would leave the next router in
    a chain unable to do the same ([LAW:composability]).
    """
    with cluster(
        ("alpha", ALPHA_VOICES, EVERYTHING), ("beta", BETA_VOICES, EVERYTHING)
    ) as consul:
        with routed(consul) as client:
            published = client.get("/v1/voices").json()["voices"]

    answers = {entry["voice_id"]: sorted(entry["models"]) for entry in published}
    assert answers == {
        "alpha-one": ["alpha", "eleven_alpha_v1"],
        "beta-one": ["beta", "eleven_beta_v1"],
    }


@pytest.mark.parametrize("model_id", ["alpha", "eleven_alpha_v1"])
def test_naming_the_engine_that_speaks_the_voice_is_honoured(model_id):
    """The epic's stated outcome, at the endpoint openconv actually calls.

    Both spellings, because a caller may name the engine or one of the ids it
    declares and neither is a second-class way to ask. The absence of the header
    is the whole assertion: the service reports every parameter it could not act
    on, so `model_id` missing from it is the service saying the field steered
    this request.
    """
    with cluster(
        ("alpha", ALPHA_VOICES, EVERYTHING), ("beta", BETA_VOICES, EVERYTHING)
    ) as consul:
        with routed(consul) as client:
            response = client.post(
                "/v1/text-to-speech/alpha-one/stream",
                json={"text": "hello there", "model_id": model_id},
            )

    assert response.status_code == 200, response.text
    assert "model_id" not in response.headers.get("x-elvenspeak-ignored", "")


def test_naming_an_engine_that_is_not_speaking_this_voice_is_refused():
    """Refusal, not override — the epic's one rule, on the engine axis.

    `alpha` is running and reachable, so this is not the "engine this deployment
    is not running" case: it is the caller asking for two things that do not go
    together. Answering in beta would be fluent, 200, and in an engine the caller
    explicitly did not ask for. The served list names the real backends, which is
    what makes the 422 actionable rather than a dead end.
    """
    with cluster(
        ("alpha", ALPHA_VOICES, EVERYTHING), ("beta", BETA_VOICES, EVERYTHING)
    ) as consul:
        with routed(consul) as client:
            response = client.post(
                "/v1/text-to-speech/beta-one/stream",
                json={"text": "hello there", "model_id": "alpha"},
            )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "beta-one" in detail["message"]
    assert detail["served"] == ["alpha", "beta", "eleven_alpha_v1", "eleven_beta_v1"]


def test_an_id_no_backend_answers_to_is_still_ignored_rather_than_refused():
    """A router does not turn a stock ElevenLabs id into a 422.

    The reason has not changed with the number of engines behind the endpoint: a
    client sending `eleven_turbo_v2` is not naming an engine, so it gets audio and
    the field named back — the behaviour the router already had, kept.
    """
    with cluster(("alpha", ALPHA_VOICES, EVERYTHING)) as consul:
        with routed(consul) as client:
            response = client.post(
                "/v1/text-to-speech/alpha-one/stream",
                json={"text": "hello there", "model_id": "eleven_turbo_v2"},
            )

    assert response.status_code == 200, response.text
    assert "model_id" in response.headers["x-elvenspeak-ignored"]


def test_a_model_id_the_language_steered_away_from_is_refused_not_reported():
    """The engine axis holds even when a language moved the request off it.

    The caller names alpha's voice, alpha's own `model_id`, and Spanish — a
    combination that was valid before languages steered anything, and that the
    fleet can no longer honour whole: the Spanish voice lives on beta. One of the
    two has to give, and it is not the engine. A voice can substitute because a
    substitute voice is still an answer to "say this"; a caller who asked for
    alpha and gets fluent beta was answered by something else entirely, and a
    header naming `model_id` as ignored is not consent.

    So this is a 422, and the test exists because narrowing made an old
    combination newly refusable — nothing else here sends `model_id` with a
    language at all. The message has to name what the caller sent: it used to
    name only `beta-uno`, a voice they never asked for and could not connect to
    anything in their request.
    """
    hablante = replace(BETA_VOICES[0], id="beta-uno", language="es")

    with cluster(
        ("alpha", ALPHA_VOICES, EVERYTHING), ("beta", (hablante,), EVERYTHING)
    ) as consul:
        response = routed(consul).post(
            "/v1/text-to-speech/alpha-one/stream",
            json={"text": "hola", "language_code": "es", "model_id": "alpha"},
            params={"output_format": "pcm_22050"},
        )

    assert response.status_code == 422, response.text
    message = response.json()["detail"]["message"]
    assert "alpha" in message
    assert "alpha-one" in message
    assert "es" in message
