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

import pytest
from conftest import DECLARED_VOICES, DeclaredEngine, DeclaredPrepared
from fastapi.testclient import TestClient

from elvenspeak import api
from elvenspeak.engine import Capability, Prosody, Speech, Voice
from elvenspeak.settings import Settings

VOICE = DECLARED_VOICES[0]

_SETTINGS = Settings(
    # The engine the *deployment* would have built, which is not the engine each
    # test below hands to `create_app`. It is here to be ignored: what a server
    # says it can do must come from the engine it was given, and nothing in this
    # file would answer differently if this field were removed.
    engine=DeclaredPrepared(),
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


def served_by(engine) -> TestClient:
    """The real API surface, over the engine given."""
    return TestClient(api.create_app(_SETTINGS, engine))


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
