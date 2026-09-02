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

VOICE = DECLARED_VOICES[0]
EVERYTHING = frozenset(Capability)

#: A voice built to kokoro's own id grammar, for the tests that call into
#: kokoro directly. `_language` reads the first letter through
#: `_VOICE_LANGUAGES` while the `create` arguments are being built, so a generic
#: fixture id reaches that table too — and `DECLARED_VOICES[0]` only survives it
#: by beginning with an `f` that happens to mean French. Renaming a conftest
#: voice would then break this file for a reason with no visible connection to
#: it, so the kokoro tests bring an id kokoro can actually read.
KOKORO_VOICE = Voice(id="af_test", name="test", description="test")


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


def test_stream_with_timestamps_aborts_rather_than_completing_quietly():
    """The one endpoint that cannot answer 502, pinned to what it really does.

    Its status line is committed when the `StreamingResponse` is constructed,
    before the body is advanced at all, so no checkpoint inside `stream()` can
    change it — which is why `/stream` pulls its first chunk *before* building
    the response and this one structurally cannot.

    What matters is that the failure is not the thing this feature exists to
    stop: a clean 200 carrying a short but well-formed JSON stream. It aborts
    instead. Asserted from a measurement — the request raises rather than
    returning — because the alternative is a comment claiming four endpoints are
    covered when three are, which is the exact failure this service has already
    paid for twice.
    """
    client = served_by(MuteEngine())
    with pytest.raises(RuntimeError, match="response already started"):
        client.post(
            f"/v1/text-to-speech/{VOICE.id}/stream/with-timestamps",
            json={"text": "First sentence. Second sentence."},
        )


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


EMPTY_REDUCTION = ValueError(
    "zero-size array to reduction operation maximum which has no identity"
)


class _Crashing:
    """Stands in for `Kokoro`, raising what numpy raises on an empty result.

    Both entry points, because both insert the pauses that crash and both have
    to reach the same translation.
    """

    def __init__(self, error: Exception) -> None:
        self._error = error

    def create(self, *args, **kwargs):
        raise self._error

    def create_timed(self, *args, **kwargs):
        raise self._error


def _engine_over(model) -> kokoro.KokoroEngine:
    """A Kokoro engine holding a model that only ever raises."""
    return kokoro.KokoroEngine(model, {KOKORO_VOICE.id: KOKORO_VOICE}, 24000)


def test_kokoro_reports_no_samples_rather_than_letting_numpy_escape():
    """The library's crash becomes the seam's way of saying it made no sound.

    Not the API's exception: the engine reports in the engine's vocabulary and
    the boundary decides the answer, so this stays true for an engine that is
    never served over HTTP at all.
    """
    assert (
        kokoro._synthesized(
            _Crashing(EMPTY_REDUCTION), KOKORO_VOICE, "Hi", Prosody(speed=1.0)
        )
        == b""
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
            KOKORO_VOICE,
            "Hi",
            Prosody(speed=1.0),
        )


def test_kokoro_does_not_swallow_another_reduction_that_is_not_the_pause_one():
    """numpy words every identity-less reduction the same way but for its name.

    Matching only the common prefix would catch a `min` or `argmax` raised
    anywhere else in the library and report it to a caller as "the engine
    produced no audio" — this fix telling the lie it exists to stop, one layer
    down.
    """
    other = ValueError(
        "zero-size array to reduction operation minimum which has no identity"
    )
    with pytest.raises(ValueError, match="minimum"):
        kokoro._synthesized(_Crashing(other), KOKORO_VOICE, "Hi", Prosody(speed=1.0))


# `speak_timed` reaches the same translation through the library's other entry
# point. Driven end to end rather than through `_created` directly, because the
# extraction that gave both callers one catch is exactly the kind of change that
# can silently stop wiring one of them up.


def test_speak_timed_reports_no_samples_rather_than_letting_numpy_escape():
    spoken = _engine_over(_Crashing(EMPTY_REDUCTION)).speak_timed(
        KOKORO_VOICE, "Hi", Prosody(speed=1.0)
    )
    assert spoken.pcm == b""
    assert spoken.timings == ()


def test_speak_timed_does_not_swallow_an_unrelated_value_error():
    with pytest.raises(ValueError, match="voice pack is corrupt"):
        _engine_over(_Crashing(ValueError("voice pack is corrupt"))).speak_timed(
            KOKORO_VOICE, "Hi", Prosody(speed=1.0)
        )
