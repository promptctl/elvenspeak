"""What the server says about itself, asked of engines that are not Piper.

Every other test here drives Piper, which declares everything there is to
declare — so the whole point of the negotiation is invisible to them: a server
that ignored the declaration entirely and hardcoded Piper's answers would pass
the rest of this suite green. That is not hypothetical, it is what shipped, and
this file exists to be the thing it fails.

The engine driven here is `conftest.DeclaredEngine`, which is not a mock of
Piper: it declares an arbitrary capability set and makes an unconvincing noise,
which is exactly the shape of the second engine this seam is for. The property
under test is that `api.py` gets every answer it gives about capabilities from
the engine's declaration, so an engine written somewhere else and never
anticipated here is described accurately — and, below, asked accurately too.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import DECLARED_VOICES, DeclaredEngine, DeclaredPrepared
from fastapi.testclient import TestClient

from elvenspeak import api, models
from elvenspeak.engine import Capability, Prosody, Speech, Voice
from elvenspeak.engines import ENGINES
from elvenspeak.settings import Settings

VOICE = DECLARED_VOICES[0]

def _settings(withheld: frozenset[Capability] = frozenset()) -> Settings:
    """A deployment's settings, withholding whatever it was asked to."""
    return Settings(
        # The engine the *deployment* would have built, which is not the engine
        # each test below hands to `create_app`. It is here to be ignored: what a
        # server says it can do must come from the engine it was given, and
        # nothing in this file would answer differently if this field were
        # removed.
        engine=DeclaredPrepared(),
        # Named, but named nothing this package ships a table for — these tests
        # are about capabilities, and an engine with no declarations gets an
        # empty alias table rather than a failure.
        engine_name="declared",
        # Named alongside every engine this build has, which is what makes a
        # `model_id` naming one of *those* a refusal rather than a shrug.
        known_engines=frozenset(ENGINES) | {"declared"},
        withheld=withheld,
        fallback=VOICE.id,
        api_key=None,
        host="127.0.0.1",
        port=0,
    )


class RecordingEngine(DeclaredEngine):
    """A [`DeclaredEngine`] that keeps what it was asked to speak with.

    For the tests that are about what crosses the seam rather than what comes
    back over the wire. The header says which parameters were dropped; only the
    engine's own side can say whether they really were.
    """

    def __init__(self, capabilities: frozenset[Capability]) -> None:
        super().__init__(capabilities)
        self.asked: list[Prosody] = []

    def speak(self, voice: Voice, text: str, prosody: Prosody) -> Speech:
        self.asked.append(prosody)
        return super().speak(voice, text, prosody)


def served_by(engine, *withheld: Capability) -> TestClient:
    """The real API surface, over the engine given, withholding what it says."""
    return TestClient(api.create_app(_settings(frozenset(withheld)), engine))


def served(*capabilities: Capability) -> TestClient:
    """The real API surface over an engine declaring exactly `capabilities`."""
    return served_by(DeclaredEngine(frozenset(capabilities)))


def ignored(client: TestClient, **body) -> str:
    response = client.post(
        f"/v1/text-to-speech/{VOICE.id}/stream",
        json={"text": "hello", **body},
        params={"output_format": "pcm_22050"},
    )
    assert response.status_code == 200, response.text
    return response.headers.get("x-elvenspeak-ignored", "")


#: One request per capability-gated parameter, asking for that parameter and for
#: nothing else. Held beside the assertion that it covers `_NEEDS_CAPABILITY`
#: exactly, so a parameter added to that table without an example fails here
#: rather than shipping with neither direction tested.
_ASKS_FOR: dict[str, dict] = {
    "voice_settings.speed": {"voice_settings": {"speed": 1.5}},
}


def test_every_capability_gated_parameter_is_exercised_below():
    """The coverage this file's parametrized tests can't assert about themselves."""
    assert set(_ASKS_FOR) == set(api._NEEDS_CAPABILITY)


@pytest.mark.parametrize("parameter", sorted(_ASKS_FOR))
def test_a_parameter_is_reported_when_the_engine_cannot_honour_it(parameter):
    """The regression this ticket exists for, from the honest side.

    `speed` was honoured unconditionally because the list of unhonourable
    parameters was hand-written against Piper, which varies its rate. An engine
    with a fixed rate got a header saying the speed was applied and audio saying
    it was not — rule 2 broken silently, in the one mechanism whose entire job is
    to stop that.
    """
    with served() as client:
        assert parameter in ignored(client, **_ASKS_FOR[parameter])


@pytest.mark.parametrize("parameter", sorted(_ASKS_FOR))
def test_the_same_parameter_goes_unreported_when_the_engine_can(parameter):
    """The other direction, which is where a header this cautious would fail.

    Naming an honoured parameter is as wrong as omitting an ignored one: a
    caller that trusts the header would work around a limitation the server does
    not have.
    """
    with served(api._NEEDS_CAPABILITY[parameter]) as client:
        assert parameter not in ignored(client, **_ASKS_FOR[parameter])


#: The capability `voice_settings.speed` needs, read out of the table rather than
#: named again, so the two tests below cannot outlive a change to it.
_SPEED = api._NEEDS_CAPABILITY["voice_settings.speed"]


def test_a_speed_the_engine_cannot_vary_never_reaches_it():
    """The reported subtraction, applied rather than only announced.

    Reporting alone leaves the header true by assumption: it says the speed was
    dropped, and it is right only for as long as every engine ignores a value it
    never claimed it could use. Withholding the value makes the header true by
    construction — there is no engine, however written, that can honour a speed
    it was not given.
    """
    engine = RecordingEngine(frozenset())
    with served_by(engine) as client:
        ignored(client, voice_settings={"speed": 1.5})

    assert [asked.speed for asked in engine.asked] == [1.0]


def test_a_speed_the_engine_can_vary_reaches_it_unchanged():
    """The other direction, where a server this cautious would break rule 1.

    Withholding from an engine that declared the capability would be the header
    telling the truth about a limitation the server invented for itself.
    """
    engine = RecordingEngine(frozenset({_SPEED}))
    with served_by(engine) as client:
        ignored(client, voice_settings={"speed": 1.5})

    assert [asked.speed for asked in engine.asked] == [1.5]


def test_an_unasked_for_parameter_is_never_reported():
    """The header's presence has to mean something happened."""
    with served() as client:
        assert ignored(client) == ""


@pytest.mark.parametrize("path", ["/with-timestamps", "/stream/with-timestamps"])
def test_the_timestamp_endpoints_refuse_an_engine_that_cannot_measure(path):
    with served(Capability.SPEED) as client:
        response = client.post(
            f"/v1/text-to-speech/{VOICE.id}{path}", json={"text": "hello there"}
        )
        assert response.status_code == 501


@pytest.mark.parametrize("path", ["/with-timestamps", "/stream/with-timestamps"])
def test_the_timestamp_endpoints_answer_an_engine_that_can(path):
    with served(Capability.TIMESTAMPS) as client:
        response = client.post(
            f"/v1/text-to-speech/{VOICE.id}{path}", json={"text": "hello there"}
        )
        assert response.status_code == 200, response.text


def test_the_refusal_explains_itself_without_naming_an_engine_or_its_settings():
    """[LAW:one-source-of-truth] The 501 used to end `set ELVENSPEAK_TIMESTAMPS=1`.

    That is a Piper environment variable, sent by the module that is arranged
    never to know Piper exists, to an operator whose engine may lack timings for
    a reason no variable can change. Pinned as a prohibition rather than as an
    exact string, because what matters is the class of wrong answer, not the
    wording of the right one.
    """
    with served() as client:
        detail = client.post(
            f"/v1/text-to-speech/{VOICE.id}/with-timestamps", json={"text": "hello"}
        ).json()["detail"]

    assert Capability.TIMESTAMPS.value in detail
    for leak in ("PIPER", "ELVENSPEAK_", "piper", "onnx", "restart"):
        assert leak not in detail


@pytest.mark.parametrize("path", ["/with-timestamps", "/stream/with-timestamps"])
def test_a_withheld_capability_is_refused_however_loudly_the_engine_declares_it(path):
    """The defect this ticket exists for, from the side that used to ship broken.

    Switching timestamps off was `ELVENSPEAK_TIMESTAMPS`, parsed by Piper — so a
    deployment that set it and ran a different engine was answered with the
    timestamps it had asked the service not to give, silently, having used the
    documented name. Enforced against the engine's declaration rather than
    delegated to it, no engine can disagree with the deployment by omission.
    """
    with served_by(
        DeclaredEngine(frozenset(Capability)), Capability.TIMESTAMPS
    ) as client:
        response = client.post(
            f"/v1/text-to-speech/{VOICE.id}{path}", json={"text": "hello there"}
        )
        assert response.status_code == 501


def test_a_withheld_parameter_is_reported_ignored_and_never_reaches_the_engine():
    """Withholding reaches every answer derived from the negotiation, not just the 501.

    The 501 gate and the ignored header are two readings of one capability set,
    which is only worth anything if the subtraction happens where that set is
    settled. Done at either endpoint instead, the other would keep announcing —
    and passing along — a parameter the deployment had switched off.
    """
    engine = RecordingEngine(frozenset({_SPEED}))
    with served_by(engine, _SPEED) as client:
        assert "voice_settings.speed" in ignored(
            client, voice_settings={"speed": 1.5}
        )

    assert [asked.speed for asked in engine.asked] == [1.0]


def test_withholding_what_the_engine_never_had_is_not_an_error():
    """Reachable by ordinary deployment, not a hypothetical.

    A Kokoro export without a `duration` output declares no timestamps, and a
    deployment that switched them off besides has said nothing contradictory —
    it has said the same thing twice. Subtraction from a set is what makes that
    free rather than a case anyone had to remember to allow.
    """
    with served_by(DeclaredEngine(frozenset()), Capability.TIMESTAMPS) as client:
        response = client.post(
            f"/v1/text-to-speech/{VOICE.id}/stream",
            json={"text": "hello there"},
            params={"output_format": "pcm_22050"},
        )
        assert response.status_code == 200, response.text


def listed(client: TestClient) -> set[str]:
    """The capabilities `GET /v1/models` advertises for this deployment.

    The engine's own entry is found by name rather than by being the only one.
    It was the only one until `piper-routing-7e2.2`, and unpacking the response
    with `entry, =` said so — a deployment whose engine declares foreign model
    ids lists those beside it, and every assertion below would have failed on the
    shape instead of on the property it is about.
    """
    response = client.get("/v1/models")
    assert response.status_code == 200, response.text
    entries = {entry["model_id"]: entry for entry in response.json()}
    assert _settings().engine_name in entries, entries
    return set(entries[_settings().engine_name]["capabilities"])


def _timestamps_declined(client: TestClient) -> bool:
    return (
        client.post(
            f"/v1/text-to-speech/{VOICE.id}/with-timestamps", json={"text": "hello"}
        ).status_code
        == 501
    )


def _speed_declined(client: TestClient) -> bool:
    return "voice_settings.speed" in ignored(client, voice_settings={"speed": 1.5})


#: How to find out, by asking the service rather than by reading its listing,
#: whether it will really honour each capability. One probe per member, held
#: beside the assertion that the table covers the enum exactly — the same shape
#: as [`_ASKS_FOR`] above and for the same reason: a capability added without a
#: probe fails here rather than shipping advertised but never checked.
#:
#: The probes deliberately read different mechanisms. TIMESTAMPS is enforced by
#: the 501 gate and SPEED by the ignored header, and the point of the tests below
#: is that one capability set decides both — so a table that asked only one of
#: them would leave the agreement it claims to test half unexamined.
_DECLINES = {
    Capability.TIMESTAMPS: _timestamps_declined,
    Capability.SPEED: _speed_declined,
}


def test_every_capability_can_be_asked_for_below():
    """The coverage the parametrized tests cannot assert about themselves."""
    assert set(_DECLINES) == set(Capability)


@pytest.mark.parametrize("capability", sorted(Capability, key=lambda item: item.name))
def test_a_capability_the_service_withholds_leaves_the_listing(capability):
    """Withholding reaches the advertisement, not only the refusal.

    The engine declares everything, so the only reason the capability is gone is
    the deployment's own subtraction. Asserted together with a probe that the
    service really does decline it, because either half alone is satisfiable by a
    bug: a listing rendered from its own roster would keep advertising what the
    gate refuses, and a gate reading a second set would refuse what the listing
    still offers. The endpoint exists to stop a caller discovering a limit from a
    501 mid-conversation, which it does only while those two agree.
    """
    with served_by(DeclaredEngine(frozenset(Capability)), capability) as client:
        assert capability.name.lower() not in listed(client)
        assert _DECLINES[capability](client)


@pytest.mark.parametrize("capability", sorted(Capability, key=lambda item: item.name))
def test_a_capability_the_service_honours_is_listed(capability):
    """The other direction, which is where a listing this cautious would fail.

    Omitting a capability the service will honour is as wrong as advertising one
    it will not: a caller that trusted the listing would work around a limit this
    deployment does not have.
    """
    with served_by(DeclaredEngine(frozenset(Capability))) as client:
        assert capability.name.lower() in listed(client)
        assert not _DECLINES[capability](client)


def test_the_listing_names_an_engine_this_module_has_never_heard_of():
    """[LAW:one-source-of-truth] The model id is the name the deployment settled on.

    `api.py` is arranged never to import the engine registry, so the id it
    publishes can only be the one `Settings` carries — the registry key kept at
    the point the choice was actually made. Driven with a name no table in this
    package mentions, because a listing that inferred the engine from anything it
    could recognise would answer correctly for `piper` and wrongly for the
    remote engine the router is going to hand it.

    Exactly one entry, still: an engine that declares no foreign model ids
    answers for its own name and nothing else, which is also what a deployment
    of an engine written elsewhere looks like.
    """
    settings = replace(_settings(), engine_name="stentor")
    with TestClient(api.create_app(settings, DeclaredEngine(frozenset()))) as client:
        listing = client.get("/v1/models").json()

    assert [entry["model_id"] for entry in listing] == ["stentor"]


def test_every_reach_says_what_it_honours():
    """The coverage a lookup with a default would have hidden.

    `_HONOURED_BY_REACH` decides whether `model_id` comes back named in
    `x-elvenspeak-ignored`, and a fourth [`Reach`] added without an entry here
    would inherit "not honoured" silently — the header claiming the field was
    dropped for a request the router had just used it to steer.
    """
    assert set(api._HONOURED_BY_REACH) == set(models.Reach)
