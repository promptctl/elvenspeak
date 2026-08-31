"""Samples into the format the caller asked for.

Signed 16-bit mono PCM in, encoded bytes out. Every one of the 28 output formats
is that same PCM put through one ffmpeg pass that resamples and encodes together,
so this module knows nothing about voices, models, or who produced the samples —
it is handed an iterator of bytes and a rate to interpret them at.

That ignorance is the point rather than an accident of the current design. Every
speech engine emits PCM and ffmpeg does not care which one did, so this half of
the pipeline is already engine-agnostic for every engine that will ever exist.

# Why `/stream` streams

[LAW:no-silent-failure] The endpoint is named `stream` and an earlier version
did not: it joined every chunk into one buffer, encoded the whole thing, and
answered with a single body. A caller that had gone to the trouble of consuming
the response incrementally got its audio in one piece at the end anyway, and
nothing said so.

Here samples are written into ffmpeg as they are produced and ffmpeg's output is
yielded as it appears, so the first audio reaches the caller while the rest of
the sentence is still being synthesized. That is worth real latency: synthesis
costs time in proportion to the audio it is producing, so on a long clause the
difference between first-byte and last-byte is most of the request.

# Why the input iterator is drained on a thread

[LAW:effects-at-boundaries] `pcm_chunks` is a *synchronous* iterator, because
that is what a speech engine most easily produces and what a caller can most
easily supply. Pulling from it inline would stall every other request on the
worker for as long as the producer takes to answer. [`_pump`] pulls each chunk on
a worker thread instead, so the event loop only ever waits on pipes.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import suppress

from .formats import OutputFormat

_LOGGER = logging.getLogger("elvenspeak.encoding")

#: How much encoded audio to hand upward at a time. Small enough that the first
#: sound leaves promptly, large enough not to spend a syscall per few samples.
_READ_SIZE = 16 * 1024

#: How long a finished encoder gets to exit before it is killed. Generous,
#: because reaching it means something is wedged rather than slow — ffmpeg has
#: already closed stdout by this point and has nothing left to do.
_EXIT_TIMEOUT = 10.0


class EncodingFailed(RuntimeError):
    """The encoded audio could not be completed.

    Two causes, deliberately one exception: the source of samples raised before
    it was done, or ffmpeg exited non-zero. This module is handed an iterator of
    bytes and does not know who produced it — naming the failure after the
    caller's domain, as `SynthesisFailed` did, put a fact from the other side of
    the seam into the one module that must not hold one.

    Distinct from an empty result: this is a fault, not a quiet answer.

    Becomes a 500 only on the non-streaming endpoints. The streaming ones have
    already committed a 200 by the time it can be raised, so all they can do is
    abort the body — a streaming caller must treat a truncated response as a
    failure, because no status code is coming to say so.
    """


async def encode(pcm: bytes, native_rate: int, fmt: OutputFormat) -> bytes:
    """Converts one complete buffer into `fmt`."""
    chunks = [part async for part in encode_stream(_once(pcm), native_rate, fmt)]
    return b"".join(chunks)


async def encode_stream(
    pcm_chunks: Iterator[bytes], native_rate: int, fmt: OutputFormat
) -> AsyncIterator[bytes]:
    """Converts samples into `fmt`, emitting encoded bytes as they are ready.

    # Three pipes, and why each is watched

    stdin is fed by [`_pump`], stdout is read here, and stderr is drained by its
    own task. The last of those is not tidiness: stderr is an OS pipe of about
    64 KB, and ffmpeg blocks when it fills. Draining it only after the stdout
    loop ends means a chatty failure can wedge ffmpeg before it has produced the
    output that would end that loop, and the request hangs rather than fails.

    # Why a broken sample source cannot pass as a short reply

    [LAW:no-silent-failure] Because that is the failure this whole service was
    rebuilt to stop producing. If the source dies halfway, ffmpeg encodes what it
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
    #: Whether the SIGKILL below came from this function. An exit code cannot
    #: carry that, and it is the only thing that makes -9 acceptable.
    killed_here = False

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
        # Bounded for the same reason the wait below is. An abandoned response
        # leaves the `while` above without having drained stdout, so ffmpeg can
        # block writing into a full pipe — and a process blocked on stdout never
        # closes stderr, so an unbounded read here waits on a process that is
        # itself waiting on us. That deadlock would hold the subprocess and both
        # tasks forever, never reaching the kill that breaks it.
        try:
            stderr = await asyncio.wait_for(errors, timeout=_EXIT_TIMEOUT)
        except TimeoutError:
            errors.cancel()
            # Only decorates the message below, so losing it costs a detail;
            # the non-zero returncode still reports the failure itself.
            stderr = b""
        # Reaped, not killed. `returncode` is still None here on the ordinary
        # path — stdout reaching EOF means ffmpeg closed the pipe, not that
        # anything has collected its status yet — so killing on "no returncode"
        # SIGKILLs a process that had finished, and turns its real exit code into
        # whatever the race produces. Waited for first; killed only if it will
        # not leave, which is what the consumer having gone away looks like.
        try:
            await asyncio.wait_for(process.wait(), timeout=_EXIT_TIMEOUT)
        except TimeoutError:
            killed_here = True
            process.kill()
            await process.wait()

    # Only reached when the generator ran to completion — an abandoned response
    # exits through the `finally` above and never gets here.
    if failure is not None:
        raise EncodingFailed(
            f"the sample source failed before the {fmt.wire_name} audio was complete"
        ) from failure
    # -9 is tolerated only when the kill above is where it came from. Inferring
    # that from the exit code instead accepts every other SIGKILL as success —
    # and an ffmpeg killed mid-encode, the OOM killer being the realistic cause
    # here, closes stdout as it dies. The read loop sees an ordinary EOF, exits
    # without exception, and `wait()` returns -9 without the timeout branch ever
    # running, so whatever was encoded before the kill ships as a complete 200.
    tolerated = (0, -9) if killed_here else (0,)
    if process.returncode not in tolerated:
        raise EncodingFailed(
            f"ffmpeg exited {process.returncode} encoding {fmt.wire_name}: "
            f"{stderr.decode(errors='replace')[:500]}"
        )


async def _pump(process, pcm_chunks: Iterator[bytes]) -> None:
    """Feeds samples into the encoder without blocking the loop.

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
