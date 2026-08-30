"""Text to samples, and samples to the format the caller asked for.

Two steps that stay separate on purpose. Piper always produces the same thing —
signed 16-bit mono PCM at its voice's native rate — and every one of the thirty
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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .formats import OutputFormat

if TYPE_CHECKING:  # pragma: no cover
    from piper import PiperVoice

_LOGGER = logging.getLogger("elvenspeak.speech")

#: How much encoded audio to hand upward at a time. Small enough that the first
#: sound leaves promptly, large enough not to spend a syscall per few samples.
_READ_SIZE = 16 * 1024

#: Bounds how far synthesis may run ahead of the encoder. Piper is faster than
#: real time, so without a bound a long reply would be fully synthesized into
#: memory while its first second was still being sent.
_QUEUE_DEPTH = 8


class SynthesisFailed(RuntimeError):
    """The encoder refused the audio, or was not there to refuse it.

    Distinct from an empty result: this is a fault, and the endpoints turn it
    into a 500 rather than an empty 200.
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
    zero.
    """

    pcm: bytes
    sample_rate: int
    phonemes: list[str] = field(default_factory=list)
    durations: list[int] = field(default_factory=list)

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
        audio.append(chunk.audio_int16_bytes)
        alignments = chunk.phoneme_alignments
        if alignments:
            for item in alignments:
                result.phonemes.append(item.phoneme)
                result.durations.append(int(item.num_samples))

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

    feeder = asyncio.create_task(_feed(process, pcm_chunks))
    try:
        while True:
            block = await process.stdout.read(_READ_SIZE)
            if not block:
                break
            yield block
    finally:
        # The consumer may have gone away mid-response — a caller that hung up,
        # or an interrupted reply. Whatever happened, this process does not get
        # to outlive the request that started it.
        if process.returncode is None:
            process.kill()
        feeder.cancel()
        stderr = await process.stderr.read() if process.stderr else b""
        await process.wait()

    if process.returncode not in (0, -9):
        raise SynthesisFailed(
            f"ffmpeg exited {process.returncode}: {stderr.decode(errors='replace')[:500]}"
        )


async def _feed(process, pcm_chunks: Iterator[bytes]) -> None:
    """Pumps synthesized samples into the encoder without blocking the loop."""
    queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=_QUEUE_DEPTH)
    loop = asyncio.get_running_loop()

    def produce() -> None:
        try:
            for chunk in pcm_chunks:
                asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    producer = loop.run_in_executor(None, produce)
    try:
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            process.stdin.write(chunk)
            await process.stdin.drain()
    except (BrokenPipeError, ConnectionResetError):
        # ffmpeg died, or the response was abandoned. The encode path already
        # reports the exit status; there is nothing to add by raising here.
        _LOGGER.debug("encoder closed its input early")
    finally:
        try:
            process.stdin.close()
        except (BrokenPipeError, RuntimeError):
            pass
        await producer


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
    object against a known stretch of text. Returns the whole text as one piece
    when it contains no boundary, so a caller always gets at least one element.
    """
    pieces = [piece.strip() for piece in _SENTENCE_END.split(text.strip())]
    return [piece for piece in pieces if piece] or [text]
