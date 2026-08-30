"""The published format set, pinned.

These tests exist because `output_format` is the parameter this service used to
accept and ignore. A regression there would not crash anything — it would answer
200 with the wrong audio — so the table gets checked rather than trusted.
"""

from __future__ import annotations

import pytest

from piper_server.formats import (
    DEFAULT_OUTPUT_FORMAT,
    SUPPORTED_OUTPUT_FORMATS,
    Codec,
    OutputFormat,
    UnknownOutputFormat,
)

#: Transcribed from ElevenLabs' API reference, independently of the table in
#: formats.py. Two hand-copies of the same published list disagree loudly if
#: either is edited carelessly, which a single copy checked against itself
#: cannot do.
PUBLISHED = {
    "mp3_22050_32",
    "mp3_24000_48",
    "mp3_44100_32",
    "mp3_44100_64",
    "mp3_44100_96",
    "mp3_44100_128",
    "mp3_44100_192",
    "opus_48000_32",
    "opus_48000_64",
    "opus_48000_96",
    "opus_48000_128",
    "opus_48000_192",
    "pcm_8000",
    "pcm_16000",
    "pcm_22050",
    "pcm_24000",
    "pcm_32000",
    "pcm_44100",
    "pcm_48000",
    "wav_8000",
    "wav_16000",
    "wav_22050",
    "wav_24000",
    "wav_32000",
    "wav_44100",
    "wav_48000",
    "ulaw_8000",
    "alaw_8000",
}


def test_supports_exactly_the_published_set():
    assert set(SUPPORTED_OUTPUT_FORMATS) == PUBLISHED


def test_default_is_elevenlabs_default():
    assert DEFAULT_OUTPUT_FORMAT == "mp3_44100_128"
    assert OutputFormat.default().wire_name == "mp3_44100_128"


@pytest.mark.parametrize("name", sorted(PUBLISHED))
def test_every_format_round_trips_its_name(name):
    """Parsing then re-spelling returns the caller's own string.

    Guards the reconstruction in `wire_name`: a format whose parts do not spell
    back to its key would be unfindable by the caller that asked for it.
    """
    assert OutputFormat.parse(name).wire_name == name


@pytest.mark.parametrize(
    "name,codec,rate,bitrate",
    [
        ("mp3_44100_128", Codec.MP3, 44100, 128),
        ("opus_48000_64", Codec.OPUS, 48000, 64),
        ("pcm_16000", Codec.PCM, 16000, None),
        ("wav_22050", Codec.WAV, 22050, None),
        ("ulaw_8000", Codec.ULAW, 8000, None),
        ("alaw_8000", Codec.ALAW, 8000, None),
    ],
)
def test_parses_into_its_parts(name, codec, rate, bitrate):
    fmt = OutputFormat.parse(name)
    assert (fmt.codec, fmt.sample_rate, fmt.bitrate_kbps) == (codec, rate, bitrate)


@pytest.mark.parametrize(
    "bogus",
    [
        "",
        "mp3",
        "mp3_44100",  # real codec, real rate, missing the bitrate MP3 requires
        "mp3_9999_1",
        "pcm_16000_128",  # bitrate on a lossless codec
        "flac_44100",
        "MP3_44100_128",  # case matters: the wire spelling is lowercase
        "pcm_11025",  # plausible rate that the published set does not include
    ],
)
def test_refuses_anything_unpublished(bogus):
    """[LAW:no-silent-failure] The refusal is the feature.

    Every one of these is a string a caller could plausibly send, and answering
    any of them with a substituted format is the bug this module was written to
    remove.
    """
    with pytest.raises(UnknownOutputFormat) as raised:
        OutputFormat.parse(bogus)
    assert raised.value.value == bogus


def test_bitrate_is_present_exactly_for_lossy_codecs():
    for name in SUPPORTED_OUTPUT_FORMATS:
        fmt = OutputFormat.parse(name)
        lossy = fmt.codec in (Codec.MP3, Codec.OPUS)
        assert (fmt.bitrate_kbps is not None) is lossy, name


def test_ffmpeg_args_carry_rate_codec_and_muxer():
    args = OutputFormat.parse("opus_48000_96").ffmpeg_args()
    assert args[args.index("-ar") + 1] == "48000"
    assert args[args.index("-c:a") + 1] == "libopus"
    assert args[args.index("-b:a") + 1] == "96k"
    assert args[args.index("-f") + 1] == "ogg"


def test_mp3_strips_the_id3_header():
    """ElevenLabs' MP3 begins at a frame sync; ffmpeg's would begin at a tag."""
    args = OutputFormat.parse("mp3_44100_128").ffmpeg_args()
    assert args[args.index("-id3v2_version") + 1] == "0"


def test_content_types_are_distinct_per_container():
    types = {OutputFormat.parse(n).content_type for n in SUPPORTED_OUTPUT_FORMATS}
    assert types == {
        "audio/mpeg",
        "audio/ogg",
        "audio/wav",
        "audio/pcm",
        "audio/basic",
        "audio/x-alaw-basic",
    }
