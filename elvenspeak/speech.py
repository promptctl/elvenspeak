"""Text to samples, and samples to the format the caller asked for.

Two steps that stay separate on purpose. Piper always produces the same thing —
signed 16-bit mono PCM at its voice's native rate — and every one of the 28
output formats is that same PCM put through one ffmpeg pass that resamples and
encodes together. Synthesis therefore knows nothing about codecs, and encoding
knows nothing about voices.

# Why `/stream` streams

[LAW:no-silent-failure] The endpoint is named `stream` and the previous version
did not: it joined every chunk into one buffer, encoded the whole thing, and
answered with a single body. A caller that had gone to the trouble of consuming
the response incrementally got its audio in one piece at the end anyway, and
nothing said so.

Here Piper's chunks are written into ffmpeg as they are produced and ffmpeg's
output is yielded as it appears, so the first audio reaches the caller while the
rest of the sentence is still being synthesized. That is worth real latency:
Piper's cost is proportional to the audio it is producing, so on a long clause
the difference between first-byte and last-byte is most of the request.

# Why synthesis runs off the event loop

[LAW:effects-at-boundaries] Piper is synchronous, CPU-bound ONNX inference. Run
inline it would stall every other request on the worker for the length of the
synthesis. It runs in a thread that feeds the encoder's stdin, so the event loop
only ever waits on pipes.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator, Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .formats import OutputFormat

if TYPE_CHECKING:  # pragma: no cover
    from piper import PiperVoice

_LOGGER = logging.getLogger("elvenspeak.speech")

#: How much encoded audio to hand upward at a time. Small enough that the first
#: sound leaves promptly, large enough not to spend a syscall per few samples.
_READ_SIZE = 16 * 1024

#: How long a finished encoder gets to exit before it is killed. Generous,
#: because reaching it means something is wedged rather than slow — ffmpeg has
#: already closed stdout by this point and has nothing left to do.
_EXIT_TIMEOUT = 10.0


class SynthesisFailed(RuntimeError):
    """The encoder refused the audio, or synthesis died before it was complete.

    Distinct from an empty result: this is a fault, not a quiet answer.

    Becomes a 500 only on the non-streaming endpoints. The streaming ones have
    already committed a 200 by the time synthesis can fail, so all they can do is
    abort the body — a streaming caller must treat a truncated response as a
    failure, because no status code is coming to say so.
    """


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
    #: Piper's per-voice speaker index, for the multi-speaker models.
    speaker_id: int | None = None

    def as_piper(self):
        from piper.config import SynthesisConfig

        return SynthesisConfig(
            speaker_id=self.speaker_id,
            length_scale=1.0 / self.speed if self.speed else 1.0,
        )


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
) -> Iterator[bytes]:
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


async def encode(pcm: bytes, native_rate: int, fmt: OutputFormat) -> bytes:
    """Converts one complete buffer into `fmt`."""
    chunks = [part async for part in encode_stream(_once(pcm), native_rate, fmt)]
    return b"".join(chunks)


async def encode_stream(
    pcm_chunks: Iterator[bytes], native_rate: int, fmt: OutputFormat
) -> AsyncIterator[bytes]:
    """Converts samples into `fmt`, emitting encoded bytes as they are ready.

    `pcm_chunks` is a *synchronous* iterator because that is what Piper gives and
    what a caller can most easily supply; it is drained on a worker thread so
    that producing it cannot block the loop.

    # Three pipes, and why each is watched

    stdin is fed by [`_pump`], stdout is read here, and stderr is drained by its
    own task. The last of those is not tidiness: stderr is an OS pipe of about
    64 KB, and ffmpeg blocks when it fills. Draining it only after the stdout
    loop ends means a chatty failure can wedge ffmpeg before it has produced the
    output that would end that loop, and the request hangs rather than fails.

    # Why a synthesis failure cannot pass as a short reply

    [LAW:no-silent-failure] Because that is the failure this whole service was
    rebuilt to stop producing. If synthesis dies halfway, ffmpeg encodes what it
    received and exits 0 — a clean 200 carrying half an answer. So the pump's
    outcome is awaited and its exception raised, rather than the process's exit
    status being trusted to describe something it never saw.
    """
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(native_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        *fmt.ffmpeg_args(),
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    assert process.stderr is not None

    pump = asyncio.create_task(_pump(process, pcm_chunks))
    errors = asyncio.create_task(process.stderr.read())
    failure: BaseException | None = None

    try:
        while True:
            block = await process.stdout.read(_READ_SIZE)
            if not block:
                break
            yield block
    finally:
        pump.cancel()
        try:
            await pump
        except asyncio.CancelledError:
            # The pump was still running when the response ended. Expected on an
            # abandoned request, and says nothing about whether synthesis worked.
            pass
        except (BrokenPipeError, ConnectionResetError):
            # ffmpeg closed its input first. Its exit status describes that
            # better than this side of the pipe can.
            _LOGGER.debug("encoder closed its input early")
        except BaseException as error:  # noqa: BLE001 - re-raised below, once
            # Captured rather than raised here: this is a `finally`, and raising
            # from it during an abort would replace the reason the response
            # actually ended.
            failure = error
        stderr = await errors
        # Reaped, not killed. `returncode` is still None here on the ordinary
        # path — stdout reaching EOF means ffmpeg closed the pipe, not that
        # anything has collected its status yet — so killing on "no returncode"
        # SIGKILLs a process that had finished, and turns its real exit code into
        # whatever the race produces. Waited for first; killed only if it will
        # not leave, which is what the consumer having gone away looks like.
        try:
            await asyncio.wait_for(process.wait(), timeout=_EXIT_TIMEOUT)
        except TimeoutError:
            process.kill()
            await process.wait()

    # Only reached when the generator ran to completion — an abandoned response
    # exits through the `finally` above and never gets here.
    if failure is not None:
        raise SynthesisFailed(
            f"synthesis failed before the {fmt.wire_name} audio was complete"
        ) from failure
    if process.returncode not in (0, -9):
        raise SynthesisFailed(
            f"ffmpeg exited {process.returncode} encoding {fmt.wire_name}: "
            f"{stderr.decode(errors='replace')[:500]}"
        )


async def _pump(process, pcm_chunks: Iterator[bytes]) -> None:
    """Feeds synthesized samples into the encoder without blocking the loop.

    One chunk is pulled per await, each on a worker thread, and written before
    the next is asked for. That ordering is the whole design: it is what keeps
    synthesis from running ahead of the encoder, so no queue is needed to bound
    it and there is no queue to deadlock against when the consumer goes away.
    An exception from the generator propagates out of this task intact rather
    than being flattened into an end-of-input that the encoder cannot tell from
    a finished sentence.
    """
    chunks = iter(pcm_chunks)
    done = object()
    try:
        while True:
            chunk = await asyncio.to_thread(next, chunks, done)
            if chunk is done:
                break
            process.stdin.write(chunk)
            await process.stdin.drain()
    finally:
        # Unconditional: ffmpeg waits on stdin forever if it is never closed, so
        # skipping this on the error path would turn a synthesis failure into a
        # hung request.
        with suppress(BrokenPipeError, ConnectionResetError, RuntimeError):
            process.stdin.close()


def _once(pcm: bytes) -> Iterator[bytes]:
    yield pcm


#: Sentence-final punctuation followed by whitespace. Deliberately simple: this
#: decides where a *streamed* reply is cut, and a wrong cut costs a slightly odd
#: pause, not a wrong result. Anything cleverer (abbreviations, decimals) would
#: be a second opinion about sentence boundaries competing with espeak's own.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Cuts text where a streamed response may be broken into pieces.

    Used only by the streaming timestamp endpoint, which must align each emitted
    object against a known stretch of text.

    Returns an empty list for text with nothing to say. The previous version
    ended `or [text]`, which cannot produce an empty list — a one-element list
    literal is always truthy — so empty input became `[""]` and reached
    synthesis. The endpoints refuse empty text at the edge; this returning
    nothing is the second half of that, so a whitespace-only string cannot slip
    through as one blank sentence.
    """
    pieces = [piece.strip() for piece in _SENTENCE_END.split(text.strip())]
    return [piece for piece in pieces if piece]
