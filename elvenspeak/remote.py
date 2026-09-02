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
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from . import engine
from .discovery import Backend
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
_TIMEOUT_SECONDS = 300.0


class RemoteFailure(RuntimeError):
    """A backend refused, failed, or could not be reached during a request.

    [LAW:no-silent-failure] Raised rather than substituted. The router fronts
    several engines and the tempting recovery — try another one — would answer in
    a voice the caller did not ask for, which is the wrong-engine answer this
    whole epic exists to refuse. A voice belongs to one backend; if that backend
    cannot speak, the caller is told.
    """


def _request(url: str, body: dict | None, accept: str) -> Any:
    """Opens `url`, returning the live response for the caller to drain.

    Left open on purpose: the streaming path needs the socket while it reads, so
    closing is the caller's job and every caller here does it with `with`.
    """
    payload = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"accept": accept} | (
        {} if payload is None else {"content-type": "application/json"}
    )
    request = urllib.request.Request(
        url,
        data=payload,
        headers=headers,
        method="GET" if payload is None else "POST",
    )
    try:
        return urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS)
    except (urllib.error.URLError, TimeoutError) as failure:
        raise RemoteFailure(f"{url}: {failure}") from None


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
    labels = published.get("labels")
    return engine.Voice(
        id=voice_id,
        name=str(published.get("name") or voice_id),
        description=str(published.get("description") or ""),
        labels=tuple(
            (str(key), str(value))
            for key, value in sorted(labels.items() if isinstance(labels, dict) else ())
        ),
    )


@dataclass(frozen=True)
class Description:
    """What a backend says it is and what it will honour.

    One value because it comes from one request. `GET /v1/models` answers both
    questions at once, and asking it twice — once per field — would let a router
    pair one backend's name with another moment's capabilities.
    """

    #: The engine this server runs, which `/v1/models` lists first precisely so a
    #: reader learns what it *is* before what it will answer to. Carried for
    #: messages: an operator told two backends collide wants that word, not two
    #: ports.
    engine_name: str
    capabilities: frozenset[engine.Capability]


@dataclass(frozen=True)
class Remote:
    """The elvenspeak server at one [`Backend`], asked over HTTP.

    Holds no voices and no capabilities. Both are the server's own answers, and a
    copy kept here would be a second one, free to describe a fleet that has moved
    on ([LAW:one-source-of-truth]). The router snapshots them where a snapshot is
    the honest thing — at boot, into the catalog the surface is built over.
    """

    backend: Backend

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
            with _request(url, None, "application/json") as response:
                return json.load(response)
        except (RemoteFailure, ValueError) as failure:
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

    def describe(self) -> Description:
        """What this server runs and what it says it will honour.

        Read from `GET /v1/models`, which reports the deployment's capabilities
        beside every model id it serves — the same set that decides its own 501s,
        so a router built on it refuses exactly what the backend would have
        refused rather than keeping a second opinion.

        [LAW:no-silent-failure] A server too old to answer this endpoint is a
        refusal to boot naming it, not an assumption. Treating a 404 as "declares
        nothing" would silently turn every timestamp request into a 501 for a
        backend that can in fact measure, and the operator would be left to infer
        an image version from missing captions. Every image built since
        `GET /v1/models` landed answers it; one that does not is behind, and
        saying so at boot is cheaper than saying it in the audio.
        """
        published = self._asked(
            "/v1/models", "could not be asked what engine it runs"
        )
        if not isinstance(published, list) or not published:
            raise ConfigError(
                [f"{self.backend.service}: /v1/models did not answer with any model"]
            )
        listed = published[0]
        declared = listed.get("capabilities") if isinstance(listed, dict) else None
        if not isinstance(declared, list):
            raise ConfigError(
                [f"{self.backend.service}: /v1/models named no capabilities"]
            )
        return Description(
            engine_name=str(listed.get("model_id") or self.backend.service),
            capabilities=frozenset(
                item for item in engine.Capability if item.name.lower() in declared
            ),
        )

    def speak(
        self, voice: engine.Voice, text: str, prosody: engine.Prosody
    ) -> engine.Speech:
        """Streams `text` back from the backend that owns `voice`.

        The response is drained as it arrives rather than collected, which is what
        makes a routed `/stream` stream: the router's own encoder starts on the
        first chunk, so the caller hears the backend's first words while the
        backend is still making its last.
        """
        url = (
            f"{self.backend.base_url}/v1/text-to-speech/{voice.id}/stream"
            f"?output_format={WIRE_FORMAT}"
        )
        response = _request(url, _body(text, prosody), "audio/pcm")

        def chunks() -> Iterator[bytes]:
            with response:
                while True:
                    block = response.read(_CHUNK)
                    if not block:
                        return
                    yield block

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
        url = (
            f"{self.backend.base_url}/v1/text-to-speech/{voice.id}/with-timestamps"
            f"?output_format={WIRE_FORMAT}"
        )
        with _request(url, _body(text, prosody), "application/json") as response:
            published = json.load(response)
        if not isinstance(published, dict) or "audio_base64" not in published:
            raise RemoteFailure(f"{url}: no timed audio in the response")
        pcm = base64.b64decode(published["audio_base64"])
        return engine.TimedSpeech(
            pcm=pcm,
            sample_rate=WIRE_RATE,
            timings=_timings(published.get("alignment"), len(pcm) // 2),
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
    sample count, and that promise is kept here by construction: each duration is
    the gap between consecutive rounded boundaries, so the rounding cannot
    accumulate, and the last one absorbs whatever the backend's final timestamp
    and the actual byte count disagree about. Rounding each duration
    independently would drift, and the alignment built from it would end before
    or after the audio it describes.

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
        boundary = round(float(end) * WIRE_RATE)
        timings.append(
            engine.Timing(
                samples=max(0, boundary - previous),
                separates_words=str(character).isspace(),
            )
        )
        previous = max(previous, boundary)

    # The audio is the fact; the timestamps describe it. Where they disagree the
    # audio wins, and the difference is charged to the last stretch rather than
    # spread, so no individual measurement is quietly altered.
    shortfall = total_samples - sum(timing.samples for timing in timings)
    last = timings[-1]
    timings[-1] = engine.Timing(
        samples=max(0, last.samples + shortfall),
        separates_words=last.separates_words,
    )
    return tuple(timings)
