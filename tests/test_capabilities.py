"""What the server says about itself, asked of engines that are not Piper.

Every other test here drives Piper, which declares everything there is to
declare — so the whole point of the negotiation is invisible to them: a server
that ignored the declaration entirely and hardcoded Piper's answers would pass
the rest of this suite green. That is not hypothetical, it is what shipped, and
this file exists to be the thing it fails.

The engine below is not a mock of Piper. It declares an arbitrary capability set
and makes an unconvincing noise, which is exactly the shape of the second engine
this seam is for: the property under test is that `api.py` gets every answer it
gives about capabilities from the engine's declaration, so an engine written
somewhere else and never anticipated here is described accurately.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from elvenspeak import api
from elvenspeak.engine import (
    Capability,
    Prosody,
    Speech,
    TimedSpeech,
    Timing,
    Voice,
)
from elvenspeak.settings import Settings

VOICE = Voice(id="fake-voice", name="Fake", description="a test engine's voice")

#: Enough samples that an encode has something to do, and short enough that the
#: whole suite pays no attention to it. The content is silence: no test here
#: listens, they read headers and status codes.
_SAMPLES = 2000
_PCM = b"\x00\x00" * _SAMPLES


class DeclaredEngine:
    """An engine that can do exactly what it was told to declare.

    [LAW:one-type-per-behavior] One type taking a capability set, rather than a
    `SpeedlessEngine` and a `TimelessEngine` beside it. What differs between the
    engines these tests need is a value, so it is passed as one — which is the
    same argument the interface itself now makes, tested by being relied upon.
    """

    def __init__(self, capabilities: frozenset[Capability]) -> None:
        self._capabilities = capabilities

    def voices(self) -> tuple[Voice, ...]:
        return (VOICE,)

    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    def speak(self, voice: Voice, text: str, prosody: Prosody) -> Speech:
        return Speech(sample_rate=22050, audio=iter([_PCM]))

    def speak_timed(self, voice: Voice, text: str, prosody: Prosody) -> TimedSpeech:
        return TimedSpeech(
            pcm=_PCM,
            sample_rate=22050,
            timings=(Timing(samples=_SAMPLES, separates_words=False),),
        )


def served(*capabilities: Capability) -> TestClient:
    """The real API surface over an engine declaring exactly `capabilities`."""
    settings = Settings(
        voices=(VOICE.id,),
        fallback=VOICE.id,
        models_dir=Path("/nonexistent"),
        allow_download=False,
        api_key=None,
        # Deliberately the opposite of what the engine below declares. The
        # setting is what an operator asked of Piper; it must reach no decision
        # this file makes, and a server that still consulted it would answer
        # every question here backwards.
        timestamps=True,
        host="127.0.0.1",
        port=0,
    )
    return TestClient(
        api.create_app(settings, DeclaredEngine(frozenset(capabilities)))
    )


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
