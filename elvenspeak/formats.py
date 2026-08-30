"""The audio shape a caller asked for, and how to produce it.

ElevenLabs names an output format with one string — `mp3_44100_128`,
`pcm_16000`, `ulaw_8000`. That string is a closed set of thirty values,
published in the API reference, so it is treated here as an enumeration to
*parse into*, never a string to inspect later.

# Why parsing happens once, at the edge

[LAW:parse-dont-validate] An unparsed `output_format` is the bug this module
exists to make unrepresentable. The previous version of this service accepted
the parameter, ignored it, and answered 200 with MP3 at whatever rate the voice
happened to run at — a caller asking for `pcm_16000` got a plausible-looking
response in the wrong format, with nothing anywhere reporting that its request
had not been honoured.

The fix is not a validation check next to the old code; it is that no code past
[`OutputFormat.parse`] can hold a format that has not already been proven to be
one of the thirty. Downstream takes an [`OutputFormat`], so "unsupported format"
is not a state a synthesis path can reach, and there is no second place where
someone could forget to look.

# Why every format is one ffmpeg invocation

[LAW:composability] Thirty formats across six codec families could be six encode
functions and a dispatch, which makes the codec a *name* — `encode_mp3`,
`encode_opus` — and every new format a new function. Instead the codec is a
*value*: Piper always emits signed 16-bit PCM at its voice's native rate, and
ffmpeg turns that into any of the thirty in a single pass that resamples and
encodes together. A thirty-first format is a row in [`_FORMATS`], not code.

That also *removes* a dependency rather than adding one. This service used to
shell out to `lame`, which knows exactly one codec — the single-format
limitation was the dependency choice showing through the API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Codec(str, Enum):
    """The six container/codec families ElevenLabs' `output_format` names.

    A `str` enum so it can be compared and logged as its wire spelling without
    a conversion at every use.
    """

    MP3 = "mp3"
    PCM = "pcm"
    WAV = "wav"
    OPUS = "opus"
    ULAW = "ulaw"
    ALAW = "alaw"


@dataclass(frozen=True)
class OutputFormat:
    """One of the thirty formats ElevenLabs publishes, already known to be legal.

    Frozen because it is a parsed fact about a request, not a workspace: two
    handlers holding the same format are holding the same value, and neither can
    edit it out from under the other.

    `bitrate_kbps` is present exactly when the codec is lossy. It is not an
    `Optional[int]` that some paths remember to check — [`ffmpeg_args`] is the
    only reader, and it is derived from the same table that decided the codec.
    """

    codec: Codec
    sample_rate: int
    bitrate_kbps: int | None

    @property
    def wire_name(self) -> str:
        """The spelling ElevenLabs uses, reconstructed from the parsed parts."""
        if self.bitrate_kbps is None:
            return f"{self.codec.value}_{self.sample_rate}"
        return f"{self.codec.value}_{self.sample_rate}_{self.bitrate_kbps}"

    @property
    def content_type(self) -> str:
        """What to send back in `Content-Type`.

        Raw PCM and the two G.711 formats have no container, so the type
        describes the sample encoding rather than a file format — a caller
        cannot infer rate or width from the response and is expected to know
        them from the format it asked for, exactly as with ElevenLabs.
        """
        return _CONTENT_TYPES[self.codec]

    def ffmpeg_args(self) -> list[str]:
        """The output half of an ffmpeg command line for this format.

        Excludes input flags and the output target, which belong to the caller
        that owns the process — see [`elvenspeak.speech`].
        """
        args = ["-ar", str(self.sample_rate), "-ac", "1"]
        args += _CODEC_ARGS[self.codec]
        if self.bitrate_kbps is not None:
            args += ["-b:a", f"{self.bitrate_kbps}k"]
        if self.codec is Codec.MP3:
            # ffmpeg writes an ID3v2 header by default. ElevenLabs' MP3 has no
            # tag, and a decoder reading a fixed number of leading bytes to find
            # the first frame is a real (if sloppy) client — matching the wire
            # exactly costs nothing here and removes a whole class of "works
            # against ElevenLabs, fails against this" report.
            args += ["-write_xing", "0", "-id3v2_version", "0", "-map_metadata", "-1"]
        args += ["-f", _MUXERS[self.codec]]
        return args

    @staticmethod
    def parse(value: str) -> "OutputFormat":
        """Turns the wire spelling into a format, or refuses.

        Raises [`UnknownOutputFormat`] rather than returning `None`, because
        every caller of this function is an HTTP handler that must answer 422 —
        an optional return would put the same `if` in each of them and let one
        of them forget.
        """
        try:
            return _FORMATS[value]
        except KeyError:
            raise UnknownOutputFormat(value) from None

    @staticmethod
    def default() -> "OutputFormat":
        """ElevenLabs' documented default for every text-to-speech endpoint."""
        return _FORMATS[DEFAULT_OUTPUT_FORMAT]


class UnknownOutputFormat(ValueError):
    """A format string outside the published set.

    Carries the offending value so the handler can name it back to the caller;
    a bare "invalid format" leaves someone diffing their string against the docs
    by eye.
    """

    def __init__(self, value: str) -> None:
        super().__init__(f"unsupported output_format: {value!r}")
        self.value = value


DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"

_CONTENT_TYPES: dict[Codec, str] = {
    Codec.MP3: "audio/mpeg",
    Codec.OPUS: "audio/ogg",
    Codec.WAV: "audio/wav",
    Codec.PCM: "audio/pcm",
    Codec.ULAW: "audio/basic",
    Codec.ALAW: "audio/x-alaw-basic",
}

_CODEC_ARGS: dict[Codec, list[str]] = {
    Codec.MP3: ["-c:a", "libmp3lame"],
    Codec.OPUS: ["-c:a", "libopus"],
    Codec.WAV: ["-c:a", "pcm_s16le"],
    Codec.PCM: ["-c:a", "pcm_s16le"],
    Codec.ULAW: ["-c:a", "pcm_mulaw"],
    Codec.ALAW: ["-c:a", "pcm_alaw"],
}

# The muxer differs from the codec for exactly the containerless families, which
# is why this is a second table rather than a field on Codec: `pcm_s16le` is the
# codec for both `wav` (RIFF container) and `pcm` (raw samples), and collapsing
# them would lose the distinction the caller is actually asking about.
_MUXERS: dict[Codec, str] = {
    Codec.MP3: "mp3",
    Codec.OPUS: "ogg",
    Codec.WAV: "wav",
    Codec.PCM: "s16le",
    Codec.ULAW: "mulaw",
    Codec.ALAW: "alaw",
}


def _build_formats() -> dict[str, OutputFormat]:
    """Every format ElevenLabs publishes, transcribed from the API reference.

    [LAW:one-source-of-truth] Written out rather than generated from a rate x
    bitrate cross product, because the published set is not a cross product —
    MP3 offers 32 kbps only at 22.05 kHz, and 128 only at 44.1 — and a
    generator would invent formats the real API rejects. The list is the map;
    `tests/test_formats.py` pins it against the spelling rules so a typo is a
    test failure rather than a 422 in production.
    """
    formats: dict[str, OutputFormat] = {}

    def add(codec: Codec, rate: int, bitrate: int | None = None) -> None:
        fmt = OutputFormat(codec=codec, sample_rate=rate, bitrate_kbps=bitrate)
        formats[fmt.wire_name] = fmt

    for rate, bitrate in (
        (22050, 32),
        (24000, 48),
        (44100, 32),
        (44100, 64),
        (44100, 96),
        (44100, 128),
        (44100, 192),
    ):
        add(Codec.MP3, rate, bitrate)

    for bitrate in (32, 64, 96, 128, 192):
        add(Codec.OPUS, 48000, bitrate)

    for rate in (8000, 16000, 22050, 24000, 32000, 44100, 48000):
        add(Codec.PCM, rate)
        add(Codec.WAV, rate)

    add(Codec.ULAW, 8000)
    add(Codec.ALAW, 8000)

    return formats


_FORMATS: dict[str, OutputFormat] = _build_formats()

SUPPORTED_OUTPUT_FORMATS: tuple[str, ...] = tuple(sorted(_FORMATS))

# Guards the transcription above against a shape that would parse but mean
# nothing — a name with the wrong number of parts, or a codec spelled two ways.
_WIRE_NAME = re.compile(r"^(mp3|pcm|wav|opus|ulaw|alaw)_\d{4,5}(_\d{2,3})?$")
assert all(_WIRE_NAME.match(name) for name in _FORMATS), "malformed format name"
assert DEFAULT_OUTPUT_FORMAT in _FORMATS, "default is not a supported format"
