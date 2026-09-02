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

from contextlib import ExitStack

import fleet
import pytest
from conftest import DECLARED_VOICES, DeclaredEngine, DeclaredPrepared, declaring
from fastapi.testclient import TestClient

from elvenspeak import api, kokoro, remote
from elvenspeak import engine as engine_mod
from elvenspeak.discovery import Backend
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

#: kokoro's other way of producing nothing, and the one that reproduces on
#: demand: text the phonemizer empties never reaches synthesis at all. Copied
#: from what the running 2026.09.02.3 image actually raised for `"'"`, because
#: the wording is the whole of what `_created` has to recognise.
NO_PHONEMES = ValueError("Nothing to synthesize, \"'\" produced no phonemes")

#: Both doors, as one parameter set. They are one fact to a caller -- kokoro made
#: no audio -- so every test that pins the translation runs over both, and a
#: third door is an entry here rather than a copied test
#: ([LAW:one-type-per-behavior]).
SILENT = [
    pytest.param(EMPTY_REDUCTION, id="nothing-survived-synthesis"),
    pytest.param(NO_PHONEMES, id="nothing-reached-synthesis"),
]


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


@pytest.mark.parametrize("error", SILENT)
def test_kokoro_reports_no_samples_rather_than_letting_the_library_escape(error):
    """The library's crash becomes the seam's way of saying it made no sound.

    Not the API's exception: the engine reports in the engine's vocabulary and
    the boundary decides the answer, so this stays true for an engine that is
    never served over HTTP at all.
    """
    assert (
        kokoro._synthesized(_Crashing(error), KOKORO_VOICE, "Hi", Prosody(speed=1.0))
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


@pytest.mark.parametrize("error", SILENT)
def test_speak_timed_reports_no_samples_rather_than_letting_the_library_escape(error):
    spoken = _engine_over(_Crashing(error)).speak_timed(
        KOKORO_VOICE, "Hi", Prosody(speed=1.0)
    )
    assert spoken.pcm == b""
    assert spoken.timings == ()


def test_speak_timed_does_not_swallow_an_unrelated_value_error():
    with pytest.raises(ValueError, match="voice pack is corrupt"):
        _engine_over(_Crashing(ValueError("voice pack is corrupt"))).speak_timed(
            KOKORO_VOICE, "Hi", Prosody(speed=1.0)
        )


# ------------------------------------------------- silence that arrives over HTTP


# A routed request is the case that matters in production -- openconv talks to the
# router, not to an engine -- and it is the case that was broken while every test
# above was green. `remote._request` handed urllib's `HTTPError` to the arm that
# catches transport failures, which it reaches because `HTTPError` IS an
# `OSError`, so a backend that answered 502 and a backend that was never reached
# arrived as the same `RemoteFailure` and a caller got a bare 500.
#
# MEASURED, not imagined: against elvenspeak-piper 2026.09.02.3 in the cluster,
# `{"text": "'"}` answered 502 with the voice named, and the identical request
# through elvenspeak-router 2026.09.02.3 answered `Internal Server Error`.
#
# These start real servers for the same reason `tests/test_router.py` does: the
# fault lived in what urllib raises for a non-2xx status, which no stub of the
# transport would have reproduced -- a fake raising `RemoteFailure` directly
# would have passed against the broken code.


def _mute_backend_url(stack, chunks: tuple[bytes, ...] = ()) -> str:
    """A served elvenspeak deployment whose engine declares voices and says nothing."""
    return stack.enter_context(
        fleet.serving(api.create_app(_settings(), MuteEngine(chunks)))
    )


def _remote_to(url: str) -> remote.Remote:
    return remote.Remote(Backend(service="mute", base_url=url))


def test_a_routed_stream_reports_the_backend_s_silence_as_silence():
    """[LAW:parse-dont-validate] 502 is the wire spelling of `Silence`, parsed back.

    The router raises exactly what a local engine raises, so the one handler in
    [`elvenspeak.api`] answers a routed silence without knowing it was routed.
    """
    with ExitStack() as stack:
        spoken = _remote_to(_mute_backend_url(stack)).speak(
            VOICE, "hi", Prosody(speed=1.0)
        )
        with pytest.raises(engine_mod.Silence, match=repr(VOICE.id)):
            list(spoken.audio)


def test_a_routed_with_timestamps_reports_the_backend_s_silence_as_silence():
    """The other speaking path, which reaches the checkpoint through `_read_json`.

    Driven end to end rather than through `_heard`, because one path being left
    unwired is precisely the failure the shared helper exists to prevent and the
    one a direct test of the helper could not see.
    """
    with ExitStack() as stack:
        with pytest.raises(engine_mod.Silence, match=repr(VOICE.id)):
            _remote_to(_mute_backend_url(stack)).speak_timed(
                VOICE, "hi", Prosody(speed=1.0)
            )


def test_a_routed_request_answers_502_end_to_end():
    """The whole path a caller actually takes: router surface, HTTP, mute backend.

    Asserting the status and not just the exception, because the exception was
    always going to be raised somewhere -- what production got was a 500, and the
    number is the entire finding.
    """
    with ExitStack() as stack:
        url = _mute_backend_url(stack)
        client = TestClient(api.create_app(_settings(), _remote_to(url)))
        answer = client.post(f"/v1/text-to-speech/{VOICE.id}", json={"text": "hi"})
    assert answer.status_code == 502
    assert VOICE.id in answer.json()["detail"]


def test_a_backend_that_refuses_for_another_reason_is_not_called_silent():
    """[LAW:no-silent-failure] The narrowness, again, in the other direction.

    A backend refusing an unauthenticated caller has not gone mute, and reporting
    it as silence would invent a diagnosis. Only 502 -- which `elvenspeak.api`
    writes in exactly one place, the `Silence` handler -- crosses back.

    A guarded backend rather than an unknown voice, because an unknown voice is
    not a refusal at all: the fleet's fallback resolves it to a real one and the
    request succeeds. That is the backend behaving correctly, and it took a red
    test to notice this case had to be built out of something else.
    """
    with ExitStack() as stack:
        url = stack.enter_context(
            fleet.serving(
                fleet.engine_app("declared", DECLARED_VOICES, api_key="let-me-in")
            )
        )
        # No key on this side, which is the whole of what makes it a 401.
        with pytest.raises(remote.RemoteFailure) as refused:
            _remote_to(url).speak_timed(VOICE, "hi", Prosody(speed=1.0))
    assert not isinstance(refused.value, engine_mod.Silence)
    assert refused.value.status == 401


def test_a_backend_that_cannot_be_reached_is_not_called_silent():
    """Unreachable and silent are the two facts the old code collapsed into one.

    Kept as a test rather than trusted to the type, because the collapse was
    invisible: both arrived as `RemoteFailure`, both looked handled, and the only
    symptom was a status code nobody was asserting.
    """
    with ExitStack() as stack:
        url = _mute_backend_url(stack)
    # The server is stopped on leaving the block, so the port now refuses.
    with pytest.raises(remote.RemoteFailure) as failed:
        _remote_to(url).speak_timed(VOICE, "hi", Prosody(speed=1.0))
    assert not isinstance(failed.value, engine_mod.Silence)
