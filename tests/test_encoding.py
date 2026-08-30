"""Encoder properties provable without a voice model.

Its own module because tests/test_api.py is gated on an installed Piper voice,
and this needs none — it feeds the encoder a buffer it constructs itself. Living
there meant the module-level skip removed it in exactly the case it was written
for: a CI machine with no baked-in model, where nothing else covers the claim
that an output format's name describes the bytes it returns.
"""

from __future__ import annotations

import pytest

from elvenspeak.formats import OutputFormat
from elvenspeak.speech import encode

NATIVE_RATE = 22050
#: One second of a cheap non-silent waveform. Non-silent because some encoders
#: special-case digital silence, and a test that passes only on silence proves
#: less than it appears to.
ONE_SECOND = b"\x00\x20" * NATIVE_RATE


@pytest.mark.parametrize("name,rate", [("pcm_8000", 8000), ("pcm_16000", 16000),
                                       ("pcm_22050", 22050), ("pcm_44100", 44100)])
async def test_pcm_length_matches_its_declared_rate(name, rate):
    """The rate in a format's name is the rate of the samples returned.

    Raw PCM carries no header, so a caller handed the wrong rate cannot discover
    it — the audio just plays at the wrong pitch. That is the silent wrong answer
    this service exists to stop producing.

    One buffer encoded several ways, rather than several synthesis calls: Piper
    is not deterministic (the same sentence gave 36352, 37888 and 37376 samples
    across three runs, since VITS samples from a noise distribution), so
    comparing two synthesized responses could never be tighter than that
    variance — far too coarse to catch a rate that is subtly wrong.
    """
    encoded = await encode(ONE_SECOND, NATIVE_RATE, OutputFormat.parse(name))
    assert len(encoded) / 2 / rate == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize(
    "name,magic",
    [
        ("mp3_44100_128", b"\xff"),
        ("wav_22050", b"RIFF"),
        ("opus_48000_64", b"OggS"),
    ],
)
async def test_container_matches_the_format_asked_for(name, magic):
    encoded = await encode(ONE_SECOND, NATIVE_RATE, OutputFormat.parse(name))
    assert encoded.startswith(magic)


async def test_mp3_carries_no_id3_tag():
    """ElevenLabs' MP3 begins at a frame sync; ffmpeg's would begin at a tag."""
    encoded = await encode(ONE_SECOND, NATIVE_RATE, OutputFormat.parse("mp3_44100_128"))
    assert not encoded.startswith(b"ID3")


async def test_every_published_format_encodes():
    """No format in the table is one this server cannot actually produce.

    The table is transcribed from a published spec, so a row could name a codec
    ffmpeg was not built with — which would surface as a 500 on the first caller
    to ask for it rather than as anything visible here.
    """
    from elvenspeak.formats import SUPPORTED_OUTPUT_FORMATS

    for name in SUPPORTED_OUTPUT_FORMATS:
        encoded = await encode(ONE_SECOND, NATIVE_RATE, OutputFormat.parse(name))
        assert encoded, f"{name} produced no bytes"
