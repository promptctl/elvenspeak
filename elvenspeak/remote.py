"""One remote elvenspeak server, as a source of speech.

[`elvenspeak.engine`] wrote its seam for exactly this — "every member must be
answerable by a remote HTTP API and by a local ONNX model alike" — and this is
the module that collects on it. Everything here is the client half of the surface
[`elvenspeak.api`] serves, so the two files describe one protocol from opposite
ends.

Not itself an [`elvenspeak.engine.Engine`]: it speaks for one server, and an
Engine speaks for a deployment. [`elvenspeak.router`] is the Engine, and it holds
several of these.

# Why the wire format is PCM at the highest rate offered

An engine hands [`elvenspeak.engine.Speech`] over as signed 16-bit PCM, so the
proxy hop has to arrive as PCM too, and asking for a format means choosing a
sample rate — a remote's native rate is not in any response header, and there is
nothing to infer it from.

`pcm_48000` is chosen because it is at or above every engine's native rate:
Piper's 22050 and Kokoro's 24000 both resample *upward* into it, so the middle
hop discards no frequency content that the caller's own format might still have
wanted. A lower choice would be the router quietly deciding that nobody needs the
top of Kokoro's band. The router's own edge then encodes down to whatever the
caller actually asked for, which is a resample that was always going to happen.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from . import engine
from .discovery import TRANSPORT_FAILURES, Backend
from .provisioning import ConfigError

#: The format every proxied synthesis is requested in. See the module docstring
#: for why this rate and not the caller's.
WIRE_FORMAT = "pcm_48000"
WIRE_RATE = 48000

#: Bytes pulled from the socket at a time. Only affects how often the streaming
#: endpoint can hand a chunk onward, never what arrives.
_CHUNK = 64 * 1024

#: Generous, because a backend is synthesising a whole utterance before the
#: timestamp endpoints answer, and a long paragraph on a busy engine legitimately
#: takes a while. Finite, because a wedged backend must fail the one request it
#: wedged rather than hold a router thread forever.
_SPEAKING_TIMEOUT_SECONDS = 300.0

#: What a backend gets to answer a question about *itself*, which is a read it
#: does no work for. Short for the same reason [`elvenspeak.discovery`]'s is: this
#: one is spent during boot, before the port is bound, so a backend that is
#: reachable but wedged has to fail the boot rather than hang it — and the router
#: asks two of these per backend, so a long limit here is paid twice over.
_ASKING_TIMEOUT_SECONDS = 5.0


class RemoteFailure(RuntimeError):
    """A backend refused, failed, or could not be reached during a request.

    [LAW:no-silent-failure] Raised rather than substituted. The router fronts
    several engines and the tempting recovery — try another one — would answer in
    a voice the caller did not ask for, which is the wrong-engine answer this
    whole epic exists to refuse. A voice belongs to one backend; if that backend
    cannot speak, the caller is told.
    """


class RemoteRefusal(RemoteFailure):
    """A backend answered, and its answer was a refusal.

    A subclass so that every `except RemoteFailure` already written keeps
    catching it: boot, discovery and description want "this backend did not give
    me what I asked for" and none of them care which way.

    It exists because the two are not one fact where synthesis is concerned. A
    backend that cannot be reached and a backend that answered "I produced no
    audio" are different things to the caller, and the second is one this service
    has a word for. Collapsing them is the answer-shaped void
    [`elvenspeak.engine.Silence`] exists to refuse, re-formed one layer up: a
    router is a caller of engines, so it inherits the obligation to say what it
    was told rather than what it guessed.
    """

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class RemoteSilence(RemoteRefusal):
    """A backend's refusal that the backend itself stamped as [`engine.Silence`].

    [LAW:types-are-the-program] The discriminator lives in the type rather than in
    a status number every reader has to interpret, so [`_heard`] has nothing to
    decide: the one place that holds the response ([`_request`]) is the one place
    that classifies it, and everything above reads the class.

    Narrower than "the backend answered 502" on purpose. 502 is what any proxy,
    ingress or sidecar in the path says when it cannot reach what is behind it,
    and translating one of those into silence would report a mute engine to an
    operator whose engine is fine and whose network is not. Only a response
    carrying [`engine.Silence.WIRE_HEADER`] — which nothing but
    [`elvenspeak.api`]'s own handler writes — is that fact.
    """


def _read_json(
    url: str, body: dict | None, timeout: float, api_key: str | None = None
) -> Any:
    """One whole JSON exchange with a backend, or a [`RemoteFailure`].

    [LAW:single-enforcer] The one place a backend's answer is read and parsed, so
    "reading a backend can fail, and it fails as `RemoteFailure`" is stated once.
    Reading outside the request's own error handling was how a connection that
    dropped mid-body escaped as a raw `OSError` — the failure is at the transfer,
    not the connect, so the conversion has to span both.
    """
    try:
        with _request(url, body, "application/json", timeout, api_key) as response:
            return json.load(response)
    except (*TRANSPORT_FAILURES, ValueError) as failure:
        raise RemoteFailure(f"{url}: {failure}") from None


def _request(
    url: str,
    body: dict | None,
    accept: str,
    timeout: float,
    api_key: str | None = None,
) -> Any:
    """Opens `url`, returning the live response for the caller to drain.

    Left open on purpose: the streaming path needs the socket while it reads, so
    closing is the caller's job and every caller here does it with `with`.

    `timeout` is passed rather than read from a constant because the two things
    this module does want two limits — a backend describing itself should answer
    at once, and a backend synthesising a paragraph should not be cut off. One
    number for both would have to be the longer, which is how the boot path came
    to be allowed to hang for ten minutes per backend.
    """
    payload = None if body is None else json.dumps(body).encode("utf-8")
    headers = (
        {"accept": accept}
        | ({} if payload is None else {"content-type": "application/json"})
        # The backends' key, not the router's own. A deployment that guards its
        # engines is doing the ordinary thing the README describes, and without
        # this every call a router makes comes back 401 — its boot included.
        | ({} if api_key is None else {"xi-api-key": api_key})
    )
    request = urllib.request.Request(
        url,
        data=payload,
        headers=headers,
        method="GET" if payload is None else "POST",
    )
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as refusal:
        # Before the transport arm, and it has to be: `HTTPError` IS an
        # `OSError`, so the broader `except` below would swallow this one whole
        # and every answer a backend gave would come back indistinguishable from
        # never having reached it. That is what it did until 7e2.12 was verified
        # against the cluster, and it is why a routed request met a backend's 502
        # as a bare 500.
        #
        # [LAW:parse-dont-validate] Which class is decided here and only here,
        # because this is the last frame that still holds the response and can
        # see its headers at all. Above this the answer is a type, and no caller
        # re-reads a status to work out what it meant.
        refused = (
            RemoteSilence
            if engine.Silence.WIRE_HEADER in refusal.headers
            else RemoteRefusal
        )
        raise refused(refusal.code, f"{url}: {refusal}") from None
    except TRANSPORT_FAILURES as failure:
        raise RemoteFailure(f"{url}: {failure}") from None


def _heard(call: Callable[[], Any], voice: engine.Voice, text: str) -> Any:
    """`call`'s result, with a backend's silence retold as this service's own.

    [LAW:parse-dont-validate] The checkpoint where a routed synthesis crosses
    back from HTTP into the domain. Above it nothing re-asks what a status meant:
    a remote engine raises [`elvenspeak.engine.Silence`] exactly as a local one
    does, so [`elvenspeak.api`]'s one handler answers both without knowing which
    it has.

    Both speaking paths call it and neither restates the rule, for the reason
    `_audible_pcm` routes through `_audible`: two spellings of one rule drift,
    and the direction they drift in is one path quietly keeping the bare 500.

    Any other refusal stays a [`RemoteRefusal`] and travels on untouched. A
    backend answering 404 or 401 is not silent, and neither is a proxy answering
    502 about a backend it could not reach; reporting either as silence would
    invent a diagnosis — the same lie in the other direction.
    """
    try:
        return call()
    except RemoteSilence:
        # Rebuilt here rather than relayed out of the backend's body: the message
        # a caller reads is `Silence`'s to write, so a routed answer and a direct
        # one read alike and there is one sentence to change if it ever changes.
        raise engine.Silence(voice, text) from None


def _spoken_at(base_url: str, voice: engine.Voice, suffix: str) -> str:
    """Where to ask `base_url` to speak in `voice`.

    [LAW:parse-dont-validate] The one place a voice id becomes part of a URL. Ids
    are operator-supplied and constrained only to being non-empty, so one holding
    `#` or `?` would truncate the path or collide with the query string that is
    appended right after it. `safe=""` encodes `/` too: a voice id may not invent
    a path segment.
    """
    quoted = urllib.parse.quote(voice.id, safe="")
    return (
        f"{base_url}/v1/text-to-speech/{quoted}{suffix}"
        f"?output_format={WIRE_FORMAT}"
    )


def _voice(published: Any, service: str) -> engine.Voice:
    """One entry of a remote `GET /v1/voices` as the voice it describes.

    [LAW:parse-dont-validate] The wire's shape is checked here, once, so nothing
    downstream holds a dict that might be a voice. `labels` is carried through
    because it is where an engine puts the facts that have no field of their own,
    and dropping them at the proxy would make a routed deployment describe its
    voices more poorly than the engine behind it does.
    """
    if not isinstance(published, dict):
        raise ConfigError([f"{service}: a voice entry was not an object"])
    voice_id = published.get("voice_id")
    if not isinstance(voice_id, str) or not voice_id:
        raise ConfigError([f"{service}: a voice entry named no voice_id"])
    declared = published.get("capabilities")
    if not isinstance(declared, list):
        raise ConfigError(
            [
                f"{service}: voice {voice_id!r} named no capabilities. "
                f"This backend predates per-voice capabilities and a router "
                f"cannot tell what it will honour."
            ]
        )
    # Checked to the same standard as `capabilities`, and refused rather than
    # defaulted for the same reason: an empty set here is indistinguishable from
    # "this backend answers to no model id", which would make the router quietly
    # drop the engine axis for that backend instead of saying the fleet is mixed.
    # [LAW:parse-dont-validate] Entry types are checked, not coerced — `str(entry)`
    # over a JSON `null` yields the model id `"None"`, an id no engine has and
    # every caller could send.
    serves = published.get("models")
    if not isinstance(serves, list) or not all(
        isinstance(entry, str) and entry for entry in serves
    ):
        raise ConfigError(
            [
                f"{service}: voice {voice_id!r} named no models. "
                f"This backend predates per-voice model ids and a router "
                f"cannot tell which engine answers for it."
            ]
        )
    labels = published.get("labels")
    return engine.Voice(
        id=voice_id,
        name=str(published.get("name") or voice_id),
        description=str(published.get("description") or ""),
        labels=tuple(
            (str(key), str(value))
            for key, value in sorted(labels.items() if isinstance(labels, dict) else ())
        ),
        # [LAW:one-source-of-truth] Read per voice, from the voice, rather than
        # from the deployment-wide set `GET /v1/models` also publishes. Those two
        # agree for a single-engine backend and cannot for a routed one, and this
        # is the answer a request is actually decided by.
        capabilities=frozenset(
            item for item in engine.Capability if item.name.lower() in declared
        ),
        # [LAW:one-source-of-truth] Read per voice, from the voice, exactly as
        # `capabilities` is and for the reason spelled out above it: the
        # deployment-wide set `GET /v1/models` publishes is the union across a
        # fleet, which is the right answer to "what can be reached at all" and the
        # wrong one to decide whether *this* voice's engine was the one named.
        models=frozenset(serves),
    )


@dataclass(frozen=True)
class Remote:
    """The elvenspeak server at one [`Backend`], asked over HTTP.

    Holds no voices and no capabilities. Both are the server's own answers, and a
    copy kept here would be a second one, free to describe a fleet that has moved
    on ([LAW:one-source-of-truth]). The router snapshots them where a snapshot is
    the honest thing — at boot, into the catalog the surface is built over.
    """

    backend: Backend
    #: The key every backend in the fleet is guarded with, or `None` for a fleet
    #: that is not. One value for the fleet rather than one per backend: the
    #: router is told a credential for the engines it fronts, and a per-backend
    #: table would be a roster — the thing this whole engine exists not to hold.
    api_key: str | None = None

    def _asked(self, path: str, what: str) -> Any:
        """A question asked while the router is booting, and its answer parsed.

        [LAW:no-silent-failure] Every way this can go wrong becomes a
        [`ConfigError`] naming the backend, because that is the only exception
        [`elvenspeak.settings.reported_or_exit`] catches: anything else reaches an
        operator as a traceback where every other startup problem reaches them as
        a line saying which service is wrong. A backend that is unreachable, that
        answers with something other than JSON, or that is simply an image too old
        to have the endpoint, are all one situation from here — this fleet cannot
        be routed to — and they are reported as one.

        The synthesis paths deliberately do *not* come through here.
        [`RemoteFailure`] is right for them: by then the process is serving, and
        one backend failing one request is not a configuration problem.
        """
        try:
            url = f"{self.backend.base_url}{path}"
            return _read_json(url, None, _ASKING_TIMEOUT_SECONDS, self.api_key)
        except RemoteFailure as failure:
            raise ConfigError([f"{self.backend.service}: {what} ({failure})"]) from None

    def voices(self) -> tuple[engine.Voice, ...]:
        """Every voice this server currently offers, in the order it offers them.

        Order preserved rather than sorted: the server put its best voice first
        for the same reason this one will, and re-sorting here would silently
        pick a different fallback for the routed deployment than the engine
        itself would have picked.
        """
        published = self._asked("/v1/voices", "could not be asked for its voices")
        if not isinstance(published, dict) or not isinstance(
            published.get("voices"), list
        ):
            raise ConfigError(
                [f"{self.backend.service}: /v1/voices did not answer with a voice list"]
            )
        return tuple(
            _voice(entry, self.backend.service) for entry in published["voices"]
        )

    def engine_name(self) -> str:
        """The engine this server runs, as `GET /v1/models` lists it first.

        Carried for the router's startup log, so an operator can see which engine
        each address turned out to be. A collision names deployments rather than
        engines — two deployments can run the same one.

        What this server will *honour* is deliberately not read here. That is a
        per-voice fact and travels on each voice in `GET /v1/voices`; the set this
        endpoint publishes is the union across them, which is the right answer to
        "what can this deployment do at all" and the wrong one to decide a request
        by ([LAW:one-source-of-truth]).

        [LAW:no-silent-failure] A server too old to answer this endpoint is a
        refusal to boot naming it, not an assumption. Every image built since
        `GET /v1/models` landed answers it; one that does not is behind, and
        saying so at boot is cheaper than saying it in the audio.
        """
        published = self._asked("/v1/models", "could not be asked what engine it runs")
        if not isinstance(published, list) or not published:
            raise ConfigError(
                [f"{self.backend.service}: /v1/models did not answer with any model"]
            )
        listed = published[0]
        if not isinstance(listed, dict):
            raise ConfigError(
                [f"{self.backend.service}: /v1/models listed something that is not a model"]
            )
        return str(listed.get("model_id") or self.backend.service)

    def speak(
        self, voice: engine.Voice, text: str, prosody: engine.Prosody
    ) -> engine.Speech:
        """Streams `text` back from the backend that owns `voice`.

        The response is drained as it arrives rather than collected, which is what
        makes a routed `/stream` stream: the router's own encoder starts on the
        first chunk, so the caller hears the backend's first words while the
        backend is still making its last.
        """
        url = _spoken_at(self.backend.base_url, voice, "/stream")
        body = _body(text, prosody)

        def chunks() -> Iterator[bytes]:
            # Opened by the first `next()`, not by `speak`. `Speech` promises "a
            # generator handed over unstarted costs nothing if the caller goes
            # away", and a connection opened before the generator exists is a
            # socket held open by a `Speech` nobody ever drains — reclaimed by the
            # garbage collector rather than by leaving this block.
            # The refusal arrives on the open, before a single byte, which is the
            # only moment it can still become a status line: `_audible` is what
            # pulls this generator's first chunk, and it does so before the
            # `StreamingResponse` exists. A backend answering 502 here therefore
            # reaches a caller as a 502, not as a stream that dies mid-body.
            with _heard(
                lambda: _request(
                    url, body, "audio/pcm", _SPEAKING_TIMEOUT_SECONDS, self.api_key
                ),
                voice,
                text,
            ) as live:
                try:
                    while True:
                        block = live.read(_CHUNK)
                        if not block:
                            return
                        yield block
                except TRANSPORT_FAILURES as failure:
                    # A backend that dies mid-utterance fails this request, in
                    # this module's own terms. The audio already handed onward
                    # stands; what must not happen is a raw socket error
                    # surfacing from inside somebody's streaming response.
                    raise RemoteFailure(f"{url}: {failure}") from None

        # Known before any audio and before any connection: the rate is this
        # module's own choice of wire format, not something a backend reports.
        return engine.Speech(sample_rate=WIRE_RATE, audio=chunks())

    def speak_timed(
        self, voice: engine.Voice, text: str, prosody: engine.Prosody
    ) -> engine.TimedSpeech:
        """Synthesises `text` whole, carrying the backend's own measurements back.

        The timings are the backend's, converted rather than recomputed: it opened
        the model that produced the audio and is the only party that measured
        anything. Deriving fresh timings here from the samples would be inventing
        an alignment and presenting it as one.
        """
        url = _spoken_at(self.backend.base_url, voice, "/with-timestamps")
        published = _heard(
            lambda: _read_json(
                url, _body(text, prosody), _SPEAKING_TIMEOUT_SECONDS, self.api_key
            ),
            voice,
            text,
        )
        if not isinstance(published, dict) or "audio_base64" not in published:
            raise RemoteFailure(f"{url}: no timed audio in the response")
        try:
            pcm = base64.b64decode(published["audio_base64"])
            timings = _timings(published.get("alignment"), len(pcm) // 2)
        except (ValueError, TypeError, OverflowError) as failure:
            # Audio that will not decode, or timestamps that are not usable
            # numbers, are this backend failing this request — the same thing an
            # unreachable one is, from the caller's side. `binascii.Error` is a
            # `ValueError`; `OverflowError` is what `round()` answers for a
            # non-finite timestamp, which `json` will happily parse from a literal
            # `Infinity` and which an ordinary huge value reaches by overflowing
            # the multiply silently.
            raise RemoteFailure(f"{url}: unreadable timed audio ({failure})") from None
        return engine.TimedSpeech(
            pcm=pcm,
            sample_rate=WIRE_RATE,
            timings=timings,
            # `word-exact` is what the surface calls a measured alignment and
            # `interpolated` what it calls a spread one, so this reads the
            # backend's own word for it rather than guessing from the numbers.
            measured=published.get("alignment_fidelity") == "word-exact",
        )


def _body(text: str, prosody: engine.Prosody) -> dict:
    """The request a backend expects, carrying only what crossed the engine seam.

    No `model_id`: the voice already decided which backend this went to, and
    naming an engine as well would be a second answer to a question already
    settled — one that a backend running something else would rightly refuse.
    """
    return {"text": text, "voice_settings": {"speed": prosody.speed}}


def _timings(alignment: Any, total_samples: int) -> tuple[engine.Timing, ...]:
    """The backend's character alignment as the durations this seam carries.

    [LAW:one-source-of-truth] `TimedSpeech` promises its timings sum to the audio's
    sample count, and that promise is kept here by construction. Each duration is
    the gap between consecutive boundaries, so rounding cannot accumulate; each
    boundary is clamped to the audio's own length, so the running sum can never
    exceed it; and the leftover is therefore never negative, so adding it to the
    last stretch closes the sum exactly.

    The clamp is what makes "by construction" true rather than usual. Without it
    a backend whose alignment overshoots the audio — a resampler padding on the
    way up to 48 kHz, a trailing character measured past the last sample — would
    leave a negative leftover that a final `max(0, ...)` would silently discard,
    and the sum would come up short. The audio is the fact and the timestamps
    describe it, so where they disagree the audio wins; expressing that as a
    bound on each boundary rather than as a correction at the end removes the
    case instead of handling it.

    A response with no alignment yields no timings, which is the same thing an
    engine that measured nothing reports, and the caller learns it from
    `measured` rather than from the length of this tuple.
    """
    if not isinstance(alignment, dict):
        return ()
    characters = alignment.get("characters")
    ends = alignment.get("character_end_times_seconds")
    if not isinstance(characters, list) or not isinstance(ends, list):
        return ()
    if len(characters) != len(ends) or not characters:
        return ()

    timings: list[engine.Timing] = []
    previous = 0
    for character, end in zip(characters, ends):
        # Non-decreasing and never past the end of the audio, which is what makes
        # every duration below non-negative and their sum at most `total_samples`.
        boundary = min(total_samples, max(previous, round(float(end) * WIRE_RATE)))
        timings.append(
            engine.Timing(
                samples=boundary - previous,
                separates_words=str(character).isspace(),
            )
        )
        previous = boundary

    # Whatever the backend's last timestamp left unaccounted for, charged to the
    # last stretch rather than spread, so no individual measurement is altered.
    # Never negative: `previous` cannot have passed `total_samples`.
    last = timings[-1]
    timings[-1] = engine.Timing(
        samples=last.samples + (total_samples - previous),
        separates_words=last.separates_words,
    )
    return tuple(timings)
