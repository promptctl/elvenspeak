"""The ElevenLabs text-to-speech surface, served by whatever engine it is given.

The shape of this module is dictated by an API this project does not own, which
is the whole point: a client written against ElevenLabs should reach this server
by changing a base URL and nothing else. So the paths, the request bodies, the
response fields and the status codes are theirs, transcribed from the published
reference rather than invented here.

The audio comes from an [`elvenspeak.engine.Engine`] handed to [`create_app`],
and nothing below names a particular one. That is the other half of the point:
this surface is the reusable part, and which engine speaks is a choice made where
the process starts.

# What "compatible" is allowed to mean

[LAW:no-silent-failure] Compatible cannot mean "accepts the request and answers
200 regardless". Three rules keep it honest, and every one of them exists
because the previous version of this service broke it:

1. A parameter that *can* be honoured is honoured. `output_format` selects from
   all 28 published formats; `voice_id` selects a real voice; `speed`
   changes the speech rate.
2. A parameter that *cannot* be honoured is named in the `x-elvenspeak-ignored`
   response header. Nothing here has an equivalent for `stability` or `seed`,
   so those are dropped — but a caller is told which of the things it asked for
   did not happen, instead of having to infer it from the audio.
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

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import alignment as align_mod
from . import encoding, text, voices
from .engine import Engine, Prosody, Voice
from .formats import (
    DEFAULT_OUTPUT_FORMAT,
    SUPPORTED_OUTPUT_FORMATS,
    OutputFormat,
    UnknownOutputFormat,
)
from .settings import Settings

_LOGGER = logging.getLogger("elvenspeak.api")

#: Body fields ElevenLabs accepts that describe a generative model's sampling,
#: cross-request conditioning, or a pronunciation database — none of which this
#: server can honour. Named here, once, so the header in rule 2 above is derived
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
    """ElevenLabs' `voice_settings`, of which only `speed` reaches the engine.

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
#: so nothing else stands between the network and the engine.
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

    #: Rejected empty rather than synthesized. An engine's behaviour on an empty
    #: string is nobody's contract, and finding out mid-stream is not an option
    #: on the streaming endpoints — the 200 is already committed by then.
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)

    @field_validator("text")
    @classmethod
    def _must_say_something(cls, value: str) -> str:
        """Rejects text with nothing in it to speak, and stamps it stripped.

        [LAW:parse-dont-validate] `min_length` measures characters, and `"   "`
        has three of them, so it reached synthesis: the streaming endpoints
        answered 200 with an empty body — `split_sentences` correctly finds no
        sentences — and the others handed whitespace to the engine, whose
        behaviour on it is nobody's contract. The refusal belongs here, at the
        one crossing, so no endpoint downstream can hold a `SpeechRequest` with
        nothing to say.

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

    def prosody(self) -> Prosody:
        speed = self.voice_settings.speed if self.voice_settings else None
        return Prosody(speed=speed if speed is not None else 1.0)

    def ignored(self) -> tuple[str, ...]:
        """Which of the caller's parameters this server could not honour.

        Only fields actually sent are reported — a caller that asked for nothing
        unsupported gets no header at all, so the header's presence means
        something happened rather than being constant noise.

        Three sources, because there are three ways a parameter goes unhonoured:
        a known field this server cannot honour, a known voice setting it cannot,
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


def create_app(settings: Settings, engine: Engine) -> FastAPI:
    """Builds the ElevenLabs surface over a ready engine.

    [LAW:one-way-deps] `engine` is the only way audio gets made here, and it
    arrives already able to make it. Nothing in this module names a concrete
    engine or knows how one is built: whoever starts the process chooses, so
    adding a second engine is a change to the entry point rather than to the API
    surface — which is the whole point of the seam.

    Takes [`Settings`] rather than reading the environment, so tests construct a
    server without touching the process environment and the deployment has one
    place its configuration comes from.
    """
    # Built once, here, rather than per request: it reads `aliases.toml`, and an
    # operator's malformed edit should stop the process rather than surface on
    # whichever call first needed an alias.
    cat = voices.Catalog.for_engine(engine, settings.fallback)

    app = FastAPI(
        title="elvenspeak",
        summary="ElevenLabs-compatible text-to-speech, served from local voices",
        version="1.0.0",
    )
    _LOGGER.info(
        "serving %s (fallback: %s)",
        ", ".join(voice.id for voice in cat.installed) or "no voices",
        settings.fallback or "none",
    )

    def require_key(xi_api_key: str | None = Header(default=None)) -> None:
        """Checks `xi-api-key` when one is configured.

        A constant-time comparison would be theatre here: the header is compared
        against a value the operator set, over a link they control, and the
        endpoint's timing is dominated by synthesis. The check exists so a
        deployment *can* be closed, not because this is an authentication
        system.
        """
        expected = settings.api_key
        if expected is None:
            return
        if xi_api_key != expected:
            raise HTTPException(status_code=401, detail="invalid xi-api-key")

    guarded = [Depends(require_key)]

    def require_timestamps() -> None:
        """Refuses the timestamp endpoints when voices loaded without timings.

        [LAW:no-silent-failure] The alternative is answering with plausible
        timings derived from nothing, which a caption renderer would trust.
        """
        if not settings.timestamps:
            raise HTTPException(
                status_code=501,
                detail=(
                    "timestamps are disabled; set ELVENSPEAK_TIMESTAMPS=1 and "
                    "restart so voices load with alignment support"
                ),
            )

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

    def resolve(voice_id: str) -> voices.Resolution:
        try:
            return cat.resolve(voice_id)
        except voices.VoiceNotInstalled as error:
            raise HTTPException(status_code=404, detail=str(error)) from None

    def headers(
        resolution: voices.Resolution,
        body: SpeechRequest,
        extra: dict[str, str] | None = None,
    ) -> dict[str, str]:
        # Every value is a list of parts, single-valued headers being lists of
        # one. `x-elvenspeak-ignored` is genuinely a list and the rest are not,
        # and giving them two shapes meant escaping ran in two places — the list
        # escaped its separator before the join, everything else after it — so a
        # value could be escaped twice and a backslash come out mangled. One
        # shape means one pass, below, with nothing to sequence.
        out: dict[str, tuple[str, ...]] = {
            "x-elvenspeak-voice": (resolution.voice.id,)
        }
        if resolution.substituted:
            out["x-elvenspeak-voice-requested"] = (resolution.requested,)
        ignored = body.ignored()
        if ignored:
            out["x-elvenspeak-ignored"] = ignored
        # An endpoint with a header of its own passes it here rather than
        # assigning it onto the result, so there is no way to add one that skips
        # the escaping below — the guarantee is structural rather than a habit.
        out.update({name: (value,) for name, value in (extra or {}).items()})
        # [LAW:single-enforcer] Every response header this service sends is built
        # here, so this is the one place the wire's encoding has to be satisfied.
        # Applied to all values rather than to the caller-controlled ones, so
        # adding a header later cannot reintroduce the problem by omission.
        return {
            name: ", ".join(_ascii_safe(part) for part in parts)
            for name, parts in out.items()
        }

    # ----------------------------------------------------------------- health

    @app.get("/health")
    def health() -> dict:
        # An empty `voices` list is the healthcheck's signal for "this server
        # cannot speak" — reachable, because an engine is entitled to offer
        # nothing, and reported rather than dressed up as a 500.
        return {"status": "ok", "voices": [voice.id for voice in cat.installed]}

    # --------------------------------------------------------------- synthesis

    @app.post("/v1/text-to-speech/{voice_id}", dependencies=guarded)
    async def convert(
        voice_id: str,
        body: SpeechRequest,
        output_format: str = Query(default=DEFAULT_OUTPUT_FORMAT),
    ) -> Response:
        fmt = parse_format(output_format)
        resolution = resolve(voice_id)

        def synthesize() -> tuple[int, bytes]:
            spoken = engine.speak(resolution.voice, body.text, body.prosody())
            return spoken.sample_rate, b"".join(spoken.audio)

        # Off the loop, which is the obligation the engine interface places on
        # every caller. FastAPI does not thread-pool `async def` handlers, so
        # draining a synchronous engine inline would stall every other request
        # for the whole synthesis — and this endpoint did exactly that until the
        # streaming path, which gets its thread from the encoder's pump, made the
        # omission visible here.
        sample_rate, pcm = await asyncio.to_thread(synthesize)
        audio = await encoding.encode(pcm, sample_rate, fmt)
        return Response(
            content=audio,
            media_type=fmt.content_type,
            headers=headers(resolution, body),
        )

    @app.post("/v1/text-to-speech/{voice_id}/stream", dependencies=guarded)
    async def convert_stream(
        voice_id: str,
        body: SpeechRequest,
        output_format: str = Query(default=DEFAULT_OUTPUT_FORMAT),
    ) -> StreamingResponse:
        fmt = parse_format(output_format)
        resolution = resolve(voice_id)

        # On a thread even though the interface promises the audio arrives lazily
        # — `speak` itself is synchronous, and an engine that opens a connection
        # or picks a model inside it would otherwise do that on the loop. The
        # samples are still pulled later, by the encoder's pump, so a client that
        # disconnects between request and read costs nothing.
        spoken = await asyncio.to_thread(
            engine.speak, resolution.voice, body.text, body.prosody()
        )
        return StreamingResponse(
            encoding.encode_stream(spoken.audio, spoken.sample_rate, fmt),
            media_type=fmt.content_type,
            headers=headers(resolution, body),
        )

    @app.post("/v1/text-to-speech/{voice_id}/with-timestamps", dependencies=guarded)
    async def convert_with_timestamps(
        voice_id: str,
        body: SpeechRequest,
        output_format: str = Query(default=DEFAULT_OUTPUT_FORMAT),
    ) -> JSONResponse:
        fmt = parse_format(output_format)
        resolution = resolve(voice_id)
        require_timestamps()

        spoken = await asyncio.to_thread(
            engine.speak_timed, resolution.voice, body.text, body.prosody()
        )
        audio = await encoding.encode(spoken.pcm, spoken.sample_rate, fmt)
        aligned = align_mod.align(body.text, spoken)
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
        output_format: str = Query(default=DEFAULT_OUTPUT_FORMAT),
    ) -> StreamingResponse:
        fmt = parse_format(output_format)
        resolution = resolve(voice_id)
        require_timestamps()
        prosody = body.prosody()

        async def stream() -> AsyncIterator[bytes]:
            # Split here rather than letting the engine split internally, because an
            # alignment has to be measured against a known stretch of text and an
            # engine's chunks do not say which words they came from. Owning the
            # split is what makes each emitted object's timings meaningful.
            elapsed = 0.0
            for sentence in text.split_sentences(body.text):
                spoken = await asyncio.to_thread(
                    engine.speak_timed, resolution.voice, sentence, prosody
                )
                audio = await encoding.encode(spoken.pcm, spoken.sample_rate, fmt)
                aligned = align_mod.align(sentence, spoken, elapsed)
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
    def list_voices() -> dict:
        return {
            "voices": [
                _voice_json(voice, cat.aliases_for(voice.id))
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
    def get_voice(voice_id: str) -> dict:
        # Exact match only: a listing endpoint that answered for every id would
        # report a voice the caller does not have, which is the opposite of what
        # it is for. Synthesis substitutes; discovery does not.
        voice = cat.get(voice_id)
        if voice is None:
            raise HTTPException(status_code=404, detail=f"unknown voice {voice_id!r}")
        return _voice_json(voice, cat.aliases_for(voice.id))

    @app.get("/v1/voices/{voice_id}/settings", dependencies=guarded)
    def voice_settings(voice_id: str) -> dict:
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

    Backslash and comma are escaped too, which is what makes the rendering
    reversible. Backslash introduces every escape, so leaving it alone let a
    caller send the literal text `\\x2c` and receive it back indistinguishable
    from a comma this function escaped — the header could no longer say which
    characters were really asked for. Comma separates the parts of a
    multi-valued header, so a comma inside one part would read as two.
    """
    return "".join(
        char if " " <= char <= "~" and char not in "\\," else _escaped(char)
        for char in value
    )


def _escaped(char: str) -> str:
    """One character as a `\\xNN`, `\\uNNNN` or `\\UNNNNNNNN` escape.

    Spelled here rather than via `unicode_escape`, which renders some characters
    as short forms (`\\r`, `\\n`) and leaves printable ones like comma untouched —
    two shapes and an exception, where the header wants one shape and none.
    """
    code = ord(char)
    if code <= 0xFF:
        return f"\\x{code:02x}"
    if code <= 0xFFFF:
        return f"\\u{code:04x}"
    return f"\\U{code:08x}"


def _voice_json(voice: Voice, aliases: tuple[str, ...]) -> dict:
    """One voice in the shape the voice endpoints return.

    Field names are ElevenLabs', not any engine's, because the whole point is
    that an unmodified client can read the response. Rendered here rather than by
    [`elvenspeak.engine`], which supplies the facts and holds no opinion about
    the wire: an engine author should not have to learn this schema to be usable.

    [`Voice.labels`] is spread into ElevenLabs' own free-form `labels` — the
    field it publishes for exactly this — so an engine's extra facts reach the
    caller without either side inventing a field for them.
    """
    return {
        "voice_id": voice.id,
        "name": voice.name,
        "category": "premade",
        "labels": dict(voice.labels),
        "description": voice.description,
        "preview_url": None,
        "available_for_tiers": [],
        "high_quality_base_model_ids": [],
        "samples": None,
        "settings": None,
        "sharing": None,
        "fine_tuning": {
            "is_allowed_to_fine_tune": False,
            "state": {},
            "verification_failures": [],
            "verification_attempts_count": 0,
            "manual_verification_requested": False,
        },
        # Not an ElevenLabs field. Round-trips the alias table so a client can
        # discover that the ElevenLabs id it holds will reach this voice, instead
        # of finding out by trying it.
        "aliases": list(aliases),
    }


def _timestamped(audio: bytes, aligned: align_mod.Alignment) -> dict:
    """The body shape both timestamp endpoints return."""
    return {
        "audio_base64": base64.b64encode(audio).decode("ascii"),
        "alignment": aligned.as_elevenlabs(),
        # ElevenLabs distinguishes the alignment of the text as written from the
        # text as normalized for speech. The engine is fed the text as written, so
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
