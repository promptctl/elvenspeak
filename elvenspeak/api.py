"""The ElevenLabs text-to-speech surface, served from local Piper voices.

The shape of this module is dictated by an API this project does not own, which
is the whole point: a client written against ElevenLabs should reach this server
by changing a base URL and nothing else. So the paths, the request bodies, the
response fields and the status codes are theirs, transcribed from the published
reference rather than invented here.

# What "compatible" is allowed to mean

[LAW:no-silent-failure] Compatible cannot mean "accepts the request and answers
200 regardless". Three rules keep it honest, and every one of them exists
because the previous version of this service broke it:

1. A parameter that *can* be honoured is honoured. `output_format` selects from
   all 28 published formats; `voice_id` selects a real voice; `speed`
   changes the speech rate.
2. A parameter that *cannot* be honoured is named in the `x-elvenspeak-ignored`
   response header. Piper has no equivalent for `stability` or `seed`, so those
   are dropped — but a caller is told which of the things it asked for did not
   happen, instead of having to infer it from the audio.
3. A request that cannot be served is refused. An unknown `output_format` is a
   422 quoting the offending value, not a silent substitution.

# Why substitution survives rule 3

An unrecognised `voice_id` still answers, in the fallback voice, because clients
hold ElevenLabs voice ids and a server that 404s all of them replaces nothing.
That is a documented contract rather than a swallowed failure, and the
`x-elvenspeak-voice` header names whatever actually spoke — see
[`elvenspeak.voices`].
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import alignment as align_mod
from . import speech, voices
from .formats import (
    DEFAULT_OUTPUT_FORMAT,
    SUPPORTED_OUTPUT_FORMATS,
    OutputFormat,
    UnknownOutputFormat,
)
from .settings import Settings

_LOGGER = logging.getLogger("elvenspeak.api")

#: Body fields ElevenLabs accepts that describe a generative model's sampling,
#: cross-request conditioning, or a pronunciation database — none of which a
#: Piper voice has. Named here, once, so the header in rule 2 above is derived
#: from one list rather than assembled at each endpoint.
_UNSUPPORTED_BODY_FIELDS = (
    "model_id",
    "language_code",
    "seed",
    "previous_text",
    "next_text",
    "previous_request_ids",
    "next_request_ids",
    "pronunciation_dictionary_locators",
    "apply_text_normalization",
    "apply_language_text_normalization",
    "use_pvc_as_ivc",
)

_UNSUPPORTED_VOICE_SETTINGS = (
    "stability",
    "similarity_boost",
    "style",
    "use_speaker_boost",
)


class VoiceSettings(BaseModel):
    """ElevenLabs' `voice_settings`, of which Piper implements `speed`.

    The rest are declared rather than swept into an `extra` bucket so that the
    schema states plainly what may be sent, and so a caller sending `stability`
    gets it reported back as ignored instead of silently discarded by the parser.

    Unmodelled settings are kept for the same reason [`SpeechRequest`] keeps
    unmodelled body fields: enumerating the four ElevenLabs publishes today makes
    rule 2 true only for those four, and a setting added next year would go back
    to being dropped with nothing reporting it.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    speed: float | None = Field(default=None, gt=0.25, le=4.0)
    stability: float | None = None
    similarity_boost: float | None = None
    style: float | None = None
    use_speaker_boost: bool | None = None


#: Longest text one request may synthesize. ElevenLabs' own per-request limit is
#: of this order, and without any bound a single unauthenticated caller can hold
#: a CPU core for as long as it likes — `ELVENSPEAK_API_KEY` is unset by default,
#: so nothing else stands between the network and ONNX inference.
MAX_TEXT_LENGTH = 5000


class SpeechRequest(BaseModel):
    """The request body every text-to-speech endpoint takes.

    Unmodelled fields are *kept*, not rejected. `extra="forbid"` would be the
    stricter choice and the wrong one for a compatibility server: ElevenLabs adds
    body fields over time, and a 422 would break clients that are correct against
    the newer API. Keeping them lets [`ignored`] name them in the response
    instead, which is rule 2 applied to parameters this server has not heard of
    yet rather than only to the ones it already enumerates.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    #: Rejected empty rather than synthesized. Piper's behaviour on an empty
    #: string is unspecified, and finding out mid-stream is not an option on the
    #: streaming endpoints — the 200 is already committed by then.
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)

    @field_validator("text")
    @classmethod
    def _must_say_something(cls, value: str) -> str:
        """Rejects text with nothing in it to speak, and stamps it stripped.

        [LAW:parse-dont-validate] `min_length` measures characters, and `"   "`
        has three of them, so it reached synthesis: the streaming endpoints
        answered 200 with an empty body — `split_sentences` correctly finds no
        sentences — and the others handed whitespace to Piper, whose behaviour on
        it is unspecified. The refusal belongs here, at the one crossing, so that
        no endpoint downstream can hold a `SpeechRequest` with nothing to say.

        Stripped rather than merely checked, so the text carried forward is the
        text that gets spoken — the alignment endpoints return `characters` built
        from this string, and leading blanks would be characters no sound answers.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("text must contain something other than whitespace")
        return stripped
    voice_settings: VoiceSettings | None = None
    model_id: str | None = None
    language_code: str | None = None
    seed: int | None = Field(default=None, ge=0, le=4294967295)
    previous_text: str | None = None
    next_text: str | None = None
    previous_request_ids: list[str] | None = None
    next_request_ids: list[str] | None = None
    pronunciation_dictionary_locators: list[dict] | None = None
    apply_text_normalization: str | None = None
    apply_language_text_normalization: bool | None = None
    use_pvc_as_ivc: bool | None = None

    def prosody(self) -> speech.Prosody:
        speed = self.voice_settings.speed if self.voice_settings else None
        return speech.Prosody(speed=speed if speed is not None else 1.0)

    def ignored(self) -> tuple[str, ...]:
        """Which of the caller's parameters this server could not honour.

        Only fields actually sent are reported — a caller that asked for nothing
        unsupported gets no header at all, so the header's presence means
        something happened rather than being constant noise.

        Three sources, because there are three ways a parameter goes unhonoured:
        a known field with no Piper equivalent, a known voice setting with none,
        and a field this server has never heard of. The last is what keeps rule 2
        true as ElevenLabs' schema grows — an unmodelled field was previously
        dropped by the parser and reported nowhere, which is the exact silent
        discard the rule exists to forbid.
        """
        sent = [name for name in _UNSUPPORTED_BODY_FIELDS if getattr(self, name) is not None]
        if self.voice_settings is not None:
            sent += [
                f"voice_settings.{name}"
                for name in _UNSUPPORTED_VOICE_SETTINGS
                if getattr(self.voice_settings, name) is not None
            ]
            sent += [
                f"voice_settings.{name}"
                for name in sorted(self.voice_settings.model_extra or {})
            ]
        sent += sorted(self.model_extra or {})
        return tuple(sent)


def create_app(settings: Settings) -> FastAPI:
    """Builds the application around an already-installed voice catalog.

    Takes [`Settings`] rather than reading the environment, so tests construct a
    server without touching the process environment and the deployment has one
    place its configuration comes from.
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Voices are installed before the first request, not on demand: a model
        # is ~60 MB, and fetching one inside a call would charge that caller an
        # unbounded, silent delay. A voice that cannot be installed is one clean
        # failure to boot instead.
        catalog = voices.install(
            keys=settings.voices,
            models_dir=settings.models_dir,
            fallback=settings.fallback,
            include_alignments=settings.timestamps,
            allow_download=settings.allow_download,
        )
        # Every configured voice is loaded now, for the same reason its files are
        # fetched now. `PiperVoice.load` builds an ONNX session — seconds of work
        # — and doing it lazily put that on the event loop inside whichever
        # request first named the voice, stalling every other request including
        # /health. Loading here also makes a corrupt or truncated model a failure
        # to boot rather than a surprise on the first call, and it is what lets
        # /health's answer mean the voices are actually usable.
        for voice in catalog.installed:
            catalog.model(voice)
        app.state.catalog = catalog
        _LOGGER.info(
            "serving %s (fallback: %s)",
            ", ".join(settings.voices),
            settings.fallback or "none",
        )
        yield

    app = FastAPI(
        title="elvenspeak",
        summary="ElevenLabs-compatible text-to-speech, served from local Piper voices",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings

    def require_key(xi_api_key: str | None = Header(default=None)) -> None:
        """Checks `xi-api-key` when one is configured.

        A constant-time comparison would be theatre here: the header is compared
        against a value the operator set, over a link they control, and the
        endpoint's timing is dominated by ONNX inference. The check exists so a
        deployment *can* be closed, not because this is an authentication
        system.
        """
        expected = settings.api_key
        if expected is None:
            return
        if xi_api_key != expected:
            raise HTTPException(status_code=401, detail="invalid xi-api-key")

    guarded = [Depends(require_key)]

    def catalog(request: Request) -> voices.Catalog:
        return request.app.state.catalog

    def parse_format(value: str) -> OutputFormat:
        try:
            return OutputFormat.parse(value)
        except UnknownOutputFormat as error:
            # 422 rather than 400 because that is what the published API answers
            # for a malformed parameter, and a client's error handling is keyed
            # to the status it already expects.
            raise HTTPException(
                status_code=422,
                detail={
                    "message": str(error),
                    "supported": list(SUPPORTED_OUTPUT_FORMATS),
                },
            ) from None

    def resolve(cat: voices.Catalog, voice_id: str) -> voices.Resolution:
        try:
            return cat.resolve(voice_id)
        except voices.VoiceNotInstalled as error:
            raise HTTPException(status_code=404, detail=str(error)) from None

    def headers(
        resolution: voices.Resolution,
        body: SpeechRequest,
        extra: dict[str, str] | None = None,
    ) -> dict[str, str]:
        out = {"x-elvenspeak-voice": resolution.voice.key}
        if resolution.substituted:
            out["x-elvenspeak-voice-requested"] = resolution.requested
        ignored = body.ignored()
        if ignored:
            # Comma is the separator, and these names are arbitrary JSON keys —
            # `extra="allow"` is what makes rule 2 hold for fields this server has
            # never heard of, and it accepts a field literally named `a, b`. Left
            # bare, that name arrives looking like two of them, so the header
            # would misreport the thing it exists to report accurately. Escaped
            # in the same form `_ascii_safe` produces, so there is one convention.
            out["x-elvenspeak-ignored"] = ", ".join(
                name.replace(",", "\\x2c") for name in ignored
            )
        # An endpoint with a header of its own passes it here rather than
        # assigning it onto the result, so there is no way to add one that skips
        # the escaping below — the guarantee is structural rather than a habit.
        out.update(extra or {})
        # [LAW:single-enforcer] Every response header this service sends is built
        # here, so this is the one place the wire's encoding has to be satisfied.
        # Applied to all values rather than to the caller-controlled ones, so
        # adding a header later cannot reintroduce the problem by omission.
        return {name: _ascii_safe(value) for name, value in out.items()}

    # ----------------------------------------------------------------- health

    @app.get("/health")
    def health(request: Request) -> dict:
        # No guard on the catalog: the lifespan sets it before the first request,
        # and an empty `voices` list is the healthcheck's signal for "this server
        # cannot speak". Reporting that for a catalog that failed to exist would
        # answer the one question this endpoint is asked with a real-looking
        # answer to a different one.
        cat = catalog(request)
        return {"status": "ok", "voices": [voice.key for voice in cat.installed]}

    # --------------------------------------------------------------- synthesis

    @app.post("/v1/text-to-speech/{voice_id}", dependencies=guarded)
    async def convert(
        voice_id: str,
        body: SpeechRequest,
        request: Request,
        output_format: str = Query(default=DEFAULT_OUTPUT_FORMAT),
    ) -> Response:
        cat = catalog(request)
        fmt = parse_format(output_format)
        resolution = resolve(cat, voice_id)
        model = cat.model(resolution.voice)

        # Off the loop. FastAPI does not thread-pool `async def` handlers, so
        # draining Piper's synchronous generator inline would stall every other
        # request for the whole synthesis — the exact thing speech.py's docstring
        # claims this service does not do, and which was true only of /stream.
        pcm = await asyncio.to_thread(
            lambda: b"".join(speech.stream_pcm(model, body.text, body.prosody()))
        )
        audio = await speech.encode(pcm, resolution.voice.sample_rate, fmt)
        return Response(
            content=audio,
            media_type=fmt.content_type,
            headers=headers(resolution, body),
        )

    @app.post("/v1/text-to-speech/{voice_id}/stream", dependencies=guarded)
    async def convert_stream(
        voice_id: str,
        body: SpeechRequest,
        request: Request,
        output_format: str = Query(default=DEFAULT_OUTPUT_FORMAT),
    ) -> StreamingResponse:
        cat = catalog(request)
        fmt = parse_format(output_format)
        resolution = resolve(cat, voice_id)
        model = cat.model(resolution.voice)

        # The generator is handed over unstarted: synthesis begins when the
        # response body is first read, so a client that disconnects between
        # request and read costs nothing.
        chunks = speech.stream_pcm(model, body.text, body.prosody())
        return StreamingResponse(
            speech.encode_stream(chunks, resolution.voice.sample_rate, fmt),
            media_type=fmt.content_type,
            headers=headers(resolution, body),
        )

    @app.post("/v1/text-to-speech/{voice_id}/with-timestamps", dependencies=guarded)
    async def convert_with_timestamps(
        voice_id: str,
        body: SpeechRequest,
        request: Request,
        output_format: str = Query(default=DEFAULT_OUTPUT_FORMAT),
    ) -> JSONResponse:
        cat = catalog(request)
        fmt = parse_format(output_format)
        resolution = resolve(cat, voice_id)
        _require_timestamps(request)
        model = cat.model(resolution.voice)

        timed = await asyncio.to_thread(
            speech.synthesize_timed,
            model,
            body.text,
            body.prosody(),
            resolution.voice.sample_rate,
        )
        audio = await speech.encode(timed.pcm, timed.sample_rate, fmt)
        aligned = align_mod.align(
            body.text,
            timed.phonemes,
            timed.durations,
            timed.sample_rate,
            measured=timed.measured,
        )
        return JSONResponse(
            content=_timestamped(audio, aligned),
            headers=headers(
                resolution,
                body,
                {"x-elvenspeak-alignment": aligned.fidelity.value},
            ),
        )

    @app.post(
        "/v1/text-to-speech/{voice_id}/stream/with-timestamps", dependencies=guarded
    )
    async def convert_stream_with_timestamps(
        voice_id: str,
        body: SpeechRequest,
        request: Request,
        output_format: str = Query(default=DEFAULT_OUTPUT_FORMAT),
    ) -> StreamingResponse:
        cat = catalog(request)
        fmt = parse_format(output_format)
        resolution = resolve(cat, voice_id)
        _require_timestamps(request)
        model = cat.model(resolution.voice)
        prosody = body.prosody()
        sample_rate = resolution.voice.sample_rate

        async def stream() -> AsyncIterator[bytes]:
            # Split here rather than letting Piper split internally, because an
            # alignment has to be measured against a known stretch of text and
            # Piper's chunks do not say which words they came from. Owning the
            # split is what makes each emitted object's timings meaningful.
            elapsed = 0.0
            for sentence in speech.split_sentences(body.text):
                timed = await asyncio.to_thread(
                    speech.synthesize_timed, model, sentence, prosody, sample_rate
                )
                audio = await speech.encode(timed.pcm, sample_rate, fmt)
                aligned = align_mod.align(
                    sentence,
                    timed.phonemes,
                    timed.durations,
                    sample_rate,
                    elapsed,
                    measured=timed.measured,
                )
                # [LAW:one-source-of-truth] The next sentence starts where this
                # alignment says this one ended. Deriving it instead from
                # `len(pcm)/2/rate` would be a second, independent answer to "how
                # long was this" — computed from the audio while the alignment is
                # computed from summed phoneme durations — and the two drift the
                # moment any sample is unattributed, silently sliding every later
                # sentence against its own audio.
                # Unguarded: `align` returns empty lists only for empty text, and
                # `split_sentences` strips and filters, so no empty sentence
                # reaches here. A guard would not protect anything — it would
                # carry the previous sentence's `elapsed` forward and lay this
                # one over audio already accounted for, which is the drift the
                # line above was rewritten to stop.
                elapsed = aligned.ends[-1]
                yield json.dumps(_timestamped(audio, aligned)).encode() + b"\n"

        return StreamingResponse(
            stream(),
            media_type="application/json",
            # No `x-elvenspeak-alignment` here, deliberately. Fidelity is decided
            # per sentence and can differ between the objects of one response, so
            # a single header could only report one of several answers — and it
            # would have to be sent before any of them were known. Each streamed
            # object carries its own `alignment_fidelity` instead, which is the
            # only place the value is true.
            headers=headers(resolution, body),
        )

    # ------------------------------------------------------------------ voices

    @app.get("/v1/voices", dependencies=guarded)
    def list_voices(request: Request) -> dict:
        cat = catalog(request)
        return {
            "voices": [
                voice.as_elevenlabs(cat.aliases_for(voice.key))
                for voice in cat.installed
            ]
        }

    @app.get("/v1/voices/settings/default", dependencies=guarded)
    def default_settings() -> dict:
        # ElevenLabs' documented defaults, reported unchanged. Only `speed` has
        # any effect here; the others are echoed so a settings screen built
        # against the real API renders with the values it expects rather than
        # with blanks.
        return {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
            "speed": 1.0,
        }

    @app.get("/v1/voices/{voice_id}", dependencies=guarded)
    def get_voice(voice_id: str, request: Request) -> dict:
        cat = catalog(request)
        # Exact match only: a listing endpoint that answered for every id would
        # report a voice the caller does not have, which is the opposite of what
        # it is for. Synthesis substitutes; discovery does not.
        voice = cat.get(voice_id)
        if voice is None:
            raise HTTPException(status_code=404, detail=f"unknown voice {voice_id!r}")
        return voice.as_elevenlabs(cat.aliases_for(voice.key))

    @app.get("/v1/voices/{voice_id}/settings", dependencies=guarded)
    def voice_settings(voice_id: str, request: Request) -> dict:
        cat = catalog(request)
        if cat.get(voice_id) is None:
            raise HTTPException(status_code=404, detail=f"unknown voice {voice_id!r}")
        return default_settings()

    return app


def _ascii_safe(value: str) -> str:
    """Renders a string so it can survive being sent as a header value.

    Two of these values are the caller's own text — the `voice_id` from the URL
    and the field names of the JSON body — and Starlette encodes header values as
    latin-1. A voice id like `日本語` therefore raised `UnicodeEncodeError` while
    building the response, turning the documented "unknown voice still gets
    audio" substitution into a 500 that named nothing.

    Escaped rather than dropped, because these headers exist to tell a caller
    what it asked for: `\\u65e5` is still recognisably the id they sent, while a
    stripped one would report a request nobody made.

    The rule is the printable range, not encodability. CR and LF are ASCII, so
    escaping only what fails an ASCII encode let them through untouched — and a
    voice id of `foo%0D%0AX-Injected:%20evil` arrives at the handler already
    percent-decoded, putting a bare CRLF in a header value. Whether the server
    below would refuse that on the way out is not something this function should
    be leaning on unstated. Everything outside `\\x20`-`\\x7e` is escaped, which
    covers C0, DEL, and the non-ASCII case together.
    """
    return "".join(
        char if " " <= char <= "~" else char.encode("unicode_escape").decode("ascii")
        for char in value
    )


def _require_timestamps(request: Request) -> None:
    """Refuses the timestamp endpoints when models were loaded without alignment.

    [LAW:no-silent-failure] The alternative is answering with plausible timings
    derived from nothing, which a caption renderer would trust.
    """
    if not request.app.state.settings.timestamps:
        raise HTTPException(
            status_code=501,
            detail=(
                "timestamps are disabled; set ELVENSPEAK_TIMESTAMPS=1 and restart so "
                "voices load with alignment support"
            ),
        )


def _timestamped(audio: bytes, aligned: align_mod.Alignment) -> dict:
    """The body shape both timestamp endpoints return."""
    return {
        "audio_base64": base64.b64encode(audio).decode("ascii"),
        "alignment": aligned.as_elevenlabs(),
        # ElevenLabs distinguishes the alignment of the text as written from the
        # text as normalized for speech. Piper is fed the text as written, so the
        # two are the same object here rather than a second, invented one.
        "normalized_alignment": aligned.as_elevenlabs(),
        # Not an ElevenLabs field. Added because the alternative is worse: these
        # timings are measured at word boundaries in the good case and evenly
        # spread in the bad one, and a caller handed only floats cannot tell
        # which it got. The non-streaming endpoint reports this in a header; on
        # the streaming one, where it varies per object, this is the only place
        # it can be said truthfully. A client that does not know the field
        # ignores it, exactly as it ignores any field it was not expecting.
        "alignment_fidelity": aligned.fidelity.value,
    }
