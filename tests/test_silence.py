"""What a caller is told when the engine behind this surface says nothing.

The failure this file exists for is piper-routing-7e2.12: Kokoro synthesized zero
samples for ordinary one-word lines, and the two shapes that came out of it were
both useless to a caller. Joined, the empty result was encoded and returned as a
200 carrying a container header and no sound — success at every layer that could
have noticed. Streamed through `kokoro_onnx`, the empty array reached
`pauses._quiet_frames`, whose `loudness.max()` raises on nothing, and escaped as
a 500 with no body.

Neither is a thing a caller can act on, and neither is caught by a green pipeline
— which is why the bug reached production and was found by a person listening.

The engine here is mute by construction rather than a broken Kokoro, because the
rule under test is not Kokoro's. Any engine can go silent, including a remote one
behind [`elvenspeak.router`] whose silence arrives over HTTP looking exactly like
a short successful body, and the boundary owes every one of them the same answer.
"""

from __future__ import annotations

import pytest
from conftest import DECLARED_VOICES, DeclaredEngine, DeclaredPrepared, declaring
from fastapi.testclient import TestClient

from elvenspeak import api, kokoro
from elvenspeak.engine import Capability, Prosody, Speech, TimedSpeech, Voice
from elvenspeak.engines import ENGINES
from elvenspeak.settings import Settings
from elvenspeak.voices import Substitution

VOICE = DECLARED_VOICES[0]
EVERYTHING = frozenset({Capability.SPEED, Capability.TIMESTAMPS})


def _settings() -> Settings:
    return Settings(
        engine=DeclaredPrepared(),
        engine_name="declared",
        known_engines=frozenset(ENGINES) | {"declared"},
        withheld=frozenset(),
        fallback=VOICE.id,
        api_key=None,
        host="127.0.0.1",
        port=0,
    )


class MuteEngine(DeclaredEngine):
    """An engine that declares voices and then produces no samples for them.

    Declaring normally is the point: a caller reaches it exactly as it reaches a
    working engine, so nothing before the synthesis has a chance to refuse the
    request first and hide what this file is measuring.

    `chunks` is a value rather than two subclasses, because "yields nothing at
    all" and "yields empty byte strings" are one engine's behaviour parameterised
    ([LAW:one-type-per-behavior]) — and they are the same fact to a caller, which
    is the claim the tests below make good on.
    """

    def __init__(self, chunks: tuple[bytes, ...] = ()) -> None:
        super().__init__(declaring(EVERYTHING))
        self._chunks = chunks

    def speak(self, voice: Voice, text: str, prosody: Prosody) -> Speech:
        return Speech(sample_rate=22050, audio=iter(self._chunks))

    def speak_timed(self, voice: Voice, text: str, prosody: Prosody) -> TimedSpeech:
        return TimedSpeech(pcm=b"".join(self._chunks), sample_rate=22050)


def served_by(engine) -> TestClient:
    return TestClient(api.create_app(_settings(), engine))


# ------------------------------------------------------------------ the checkpoint


def test_audible_keeps_every_chunk_including_the_one_it_pulled():
    """The proof costs no audio.

    [`api._audible`] establishes that the engine spoke by pulling a chunk, so the
    thing most likely to break is that the pulled chunk never reaches the
    encoder — a first frame silently eaten, which no status code would reveal and
    which sounds like a click rather than a failure.
    """
    chunks = [b"one", b"two", b"three"]
    assert list(api._audible(VOICE, "hi", iter(chunks))) == chunks


@pytest.mark.parametrize(
    "chunks",
    [
        pytest.param((), id="yields-nothing"),
        pytest.param((b"", b""), id="yields-only-empty-chunks"),
    ],
)
def test_an_engine_that_produces_no_samples_is_refused(chunks):
    """Both spellings of silence are the same fact and get the same answer."""
    with pytest.raises(api.Silence):
        list(api._audible(VOICE, "hi", iter(chunks)))


def test_silence_names_the_voice_so_the_log_says_which_one_went_mute():
    """Behind a router the voices come from different engines, so "an engine
    produced nothing" without the voice does not say which backend to look at."""
    with pytest.raises(api.Silence, match=repr(VOICE.id)):
        list(api._audible(VOICE, "hi", iter(())))


# -------------------------------------------------------------- what a caller gets


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("", id="convert"),
        pytest.param("/stream", id="stream"),
        pytest.param("/with-timestamps", id="with-timestamps"),
    ],
)
@pytest.mark.parametrize(
    "chunks",
    [
        pytest.param((), id="yields-nothing"),
        pytest.param((b"", b""), id="yields-only-empty-chunks"),
    ],
)
def test_a_mute_engine_answers_502_and_says_so(path, chunks):
    """The regression proper: no 200, and a body that names what went wrong.

    `/stream` is in here deliberately. Its status line is sent when the response
    object is constructed, so a checkpoint that ran inside the body generator
    would be too late and this case would still 200 — which is why `api.py` pulls
    the first chunk before building the `StreamingResponse` rather than letting
    the encoder do it.
    """
    response = served_by(MuteEngine(chunks)).post(
        f"/v1/text-to-speech/{VOICE.id}{path}", json={"text": "Say something."}
    )
    assert response.status_code == 502
    assert "no audio" in response.json()["detail"]
    assert VOICE.id in response.json()["detail"]


def test_an_engine_that_speaks_is_not_refused():
    """The other half of the regression, and the reason it is not paranoia.

    A checkpoint that answered 502 too eagerly would take the whole service down
    while looking, in the logs, exactly like the bug it was added to fix.
    """
    response = served_by(DeclaredEngine(declaring(EVERYTHING))).post(
        f"/v1/text-to-speech/{VOICE.id}", json={"text": "Say something."}
    )
    assert response.status_code == 200
    assert response.content


# ---------------------------------------------------- the engine's own translation


class _Crashing:
    """Stands in for `Kokoro`, raising what numpy raises on an empty result."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def create(self, *args, **kwargs):
        raise self._error


def test_kokoro_reports_no_samples_rather_than_letting_numpy_escape():
    """The library's crash becomes the seam's way of saying it made no sound.

    Not the API's exception: the engine reports in the engine's vocabulary and
    the boundary decides the answer, so this stays true for an engine that is
    never served over HTTP at all.
    """
    crash = ValueError(
        "zero-size array to reduction operation maximum which has no identity"
    )
    assert (
        kokoro._synthesized(_Crashing(crash), VOICE, "Hi", Prosody(speed=1.0)) == b""
    )


def test_kokoro_does_not_swallow_an_unrelated_value_error():
    """[LAW:no-silent-failure] The narrowness is the point.

    A blanket `except ValueError` here would turn a real defect in this engine
    into a tidy report that the voice was merely quiet — the same class of lie
    the checkpoint exists to stop, just told one layer lower.
    """
    with pytest.raises(ValueError, match="voice pack is corrupt"):
        kokoro._synthesized(
            _Crashing(ValueError("voice pack is corrupt")),
            VOICE,
            "Hi",
            Prosody(speed=1.0),
        )
