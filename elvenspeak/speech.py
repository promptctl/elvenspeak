"""Text into samples.

Piper always produces the same thing — signed 16-bit mono PCM at its voice's
native rate — plus, when asked, the phoneme durations that account for it.
Turning those samples into one of the 28 wire formats is [`encoding`]'s job and
happens on the other side of that seam, so synthesis knows nothing about codecs
and encoding knows nothing about voices.

What is left here after that cut is the whole of what this service needs from a
speech engine: PCM chunks, the rate they are at, and phoneme durations when the
engine has them. That is the interface, discovered rather than designed.

# Why callers run synthesis off the event loop

[LAW:effects-at-boundaries] Piper is synchronous, CPU-bound ONNX inference, so
every function here blocks for as long as the audio takes to make. Run inline in
an `async def` handler it would stall every other request on the worker for the
length of the synthesis.

Nothing here may therefore be called on the event loop. A caller discharges that
one of two ways: await it through `asyncio.to_thread`, or hand the generator to
[`encoding.encode_stream`], which pulls it on a worker one chunk at a time. Which
mechanism suits a given endpoint is that endpoint's business — this states the
obligation rather than listing who currently meets it, because a roster of
callers is true until the next endpoint lands and silently false afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator

    from piper import PiperVoice


@dataclass
class Prosody:
    """The knobs Piper exposes, in the shape ElevenLabs' `voice_settings` names.

    Only the settings with a real Piper equivalent appear. ElevenLabs' others —
    `stability`, `similarity_boost`, `style`, `use_speaker_boost` — describe a
    generative model's sampling and have no meaning for a Piper voice, so they
    are accepted at the edge and dropped there, in one documented place, rather
    than being threaded down here to be ignored somewhere less visible.
    """

    #: Inverse of speed: Piper's `length_scale` stretches audio, so a caller
    #: asking for speed 2.0 wants each phoneme to last half as long.
    speed: float = 1.0

    def as_piper(self):
        # No speaker_id. Piper's multi-speaker models take an index, and there is
        # no ElevenLabs body field to source one from — so the knob was declared,
        # forwarded, and set by nothing, which claims a capability this API does
        # not have. Reaching it would mean inventing a request field only this
        # server understands, and a field no ElevenLabs client will ever send is
        # the opposite of what this service is for. A multi-speaker voice speaks
        # as its default, which is what Piper does with no id.
        from piper.config import SynthesisConfig

        return SynthesisConfig(length_scale=1.0 / self.speed if self.speed else 1.0)


@dataclass
class Timed:
    """Everything one synthesis produced, when the caller wants timings too.

    The phoneme lists are parallel and cumulative across chunks, so a multi-chunk
    utterance yields one continuous timeline rather than several restarting at
    zero. Their durations sum to the sample count of `pcm` — every sample is
    accounted for, whether or not the model said what produced it.
    """

    pcm: bytes
    sample_rate: int
    phonemes: list[str] = field(default_factory=list)
    durations: list[int] = field(default_factory=list)
    #: False when some audio arrived without the model saying which phonemes
    #: produced it. The timeline still spans the whole utterance, but its word
    #: boundaries are no longer measurements, so nothing downstream may present
    #: them as such.
    measured: bool = True

    @property
    def has_timings(self) -> bool:
        return bool(self.durations)


def stream_pcm(
    model: "PiperVoice", text: str, prosody: Prosody
) -> "Iterator[bytes]":
    """Piper's raw samples, one chunk at a time, as they are produced."""
    for chunk in model.synthesize(text, syn_config=prosody.as_piper()):
        yield chunk.audio_int16_bytes


def synthesize_timed(
    model: "PiperVoice", text: str, prosody: Prosody, sample_rate: int
) -> Timed:
    """Synthesizes in one piece, collecting phoneme durations alongside.

    Whole rather than streamed because the timestamp endpoints must send the
    alignment with the audio, and an alignment is only complete once the last
    phoneme has been measured.
    """
    result = Timed(pcm=b"", sample_rate=sample_rate)
    audio: list[bytes] = []

    for chunk in model.synthesize(
        text, syn_config=prosody.as_piper(), include_alignments=True
    ):
        samples = chunk.audio_int16_bytes
        audio.append(samples)
        alignments = chunk.phoneme_alignments
        if alignments:
            for item in alignments:
                result.phonemes.append(item.phoneme)
                result.durations.append(int(item.num_samples))
        elif samples:
            # Audio the model produced without saying which phonemes made it.
            # Dropping it would leave `durations` summing to less than the audio
            # it describes, and every timing derived from that sum would be
            # short — a whole timeline quietly compressed against real audio.
            # Recorded instead as an unattributed span, marked with a boundary
            # phoneme so it reads as silence between words rather than as part
            # of one, and the result stops claiming to be measured.
            result.phonemes.append(" ")
            result.durations.append(len(samples) // 2)
            result.measured = False

    result.pcm = b"".join(audio)
    return result
