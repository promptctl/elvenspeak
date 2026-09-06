"""How much synthesis this process does at once, asserted through the real surface.

[LAW:verifiable-goals] The bound exists because elvenspeak-piper was OOM-killed
at a 2048 MiB limit during an eight-way burst (piper-memory-9rc), and the failure
it prevents is observable only on a memory-limited node under load. The property
underneath it is cheap and exact -- no more than N syntheses are inside the engine
at the same moment -- so it is checked here in seconds rather than inferred from
resident bytes on a runner.

DRIVEN THROUGH A REAL SERVER, over HTTP, against `/stream`. That costs a uvicorn
thread and is worth it twice over.

Once, because `/stream` is the endpoint that would have been missed. `piper.speak`
returns an UNSTARTED generator (piper.py:156) and `encoding._pump` pulls it one
chunk per await, so the synthesis a streaming request pays for happens after its
handler has returned. A gate released when the handler returns bounds the two
eager endpoints and leaves this one exactly as unbounded as it was -- while every
test of the eager paths goes green. That is the shape of fix this project has
already shipped twice: correct in the suite, absent at the endpoint production
calls.

And again, because `TestClient` would not show it. Its requests do not overlap the
way concurrent callers do, and a bound looks perfectly held by a client that never
asks for two things at once.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress

import fleet
import httpx
import pytest
from conftest import DECLARED_VOICES, DeclaredPrepared, declaring

from elvenspeak import api, create_app
from elvenspeak.engine import (
    Capability,
    Prosody,
    Speech,
    TimedSpeech,
    Timing,
    Voice,
)
from elvenspeak.engines import ENGINES
from elvenspeak.settings import Settings
from elvenspeak.text import split_sentences

#: Enough callers to exceed every bound under test, so the ceiling is what the
#: setting says rather than what the test could muster.
_CALLERS = 6

#: Four sentences, and the punctuation is load-bearing. `text.split_sentences`
#: cuts on sentence-final punctuation followed by whitespace, so a body without
#: any -- "one two three four", which this file used -- comes back as ONE
#: sentence, and `/stream/with-timestamps`'s per-sentence loop then takes and
#: gives back its permit exactly once. That is the single-acquire shape the other
#: endpoints already cover, so the coverage this file claimed for the repeated
#: shape was not being exercised at all. `test_the_sentence_loop_really_cycles`
#: holds this to more than one synthesis per request so it cannot quietly revert.
_TEXT = "One. Two. Three. Four."

#: A synthesis long enough that overlapping ones actually overlap in wall time.
#: Split into chunks because a single sleeping chunk would sit entirely inside
#: `_audible`'s first pull, in the handler, and never reach the pump -- which is
#: the half of the window this file exists to cover.
_CHUNKS = 6
_CHUNK_SECONDS = 0.02

_RATE = 22050
#: One chunk of real samples. Silence is fine; `_audible` refuses only emptiness.
_SAMPLES = b"\x00\x01" * 256


class Overlap:
    """How many syntheses are inside the engine now, and the most ever seen.

    Locked because the count is written from worker threads: the first chunk is
    pulled on the thread `_audible` runs on and the rest on whichever thread the
    encoder's pump borrows, so several callers really do write this concurrently.
    An unlocked `most` would under-report exactly when the test matters.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._now = 0
        self.most = 0
        #: How many times the engine was entered at all. Distinct from `most`,
        #: and the thing that says whether a per-request loop really looped.
        self.entries = 0

    @contextmanager
    def one_more(self) -> Iterator[None]:
        with self._lock:
            self._now += 1
            self.entries += 1
            self.most = max(self.most, self._now)
        try:
            yield
        finally:
            with self._lock:
                self._now -= 1


class WatchedEngine:
    """An engine that reports when it is synthesising, and is lazy about starting.

    Lazy on purpose, and it is the whole point of the stand-in: `speak` returns a
    generator that has not run, so nothing is counted until something pulls
    samples. That is piper's shape (piper.py:156), and an eager stand-in would
    move the entire synthesis into the handler and quietly test a window the real
    engine does not use.
    """

    def __init__(self, overlap: Overlap, speak_seconds: float = 0.0) -> None:
        self._overlap = overlap
        #: How long `speak` itself blocks before returning its generator. Zero is
        #: piper's shape; a real delay is what lets a caller hang up while the
        #: engine is still inside `speak`, which is where nearly all of a
        #: streaming request's wall time goes and which no other test here covers.
        self._speak_seconds = speak_seconds
        self._voices = declaring(frozenset(Capability), DECLARED_VOICES)

    def voices(self) -> tuple[Voice, ...]:
        return self._voices

    def speak(self, voice: Voice, text: str, prosody: Prosody) -> Speech:
        # Counted too, because this is engine work like any other -- and it is
        # the work a hang-up orphans, since a thread already running cannot be
        # cancelled. At the default `speak_seconds` of zero this is instant and
        # contributes nothing to the overlap the other tests measure.
        with self._overlap.one_more():
            time.sleep(self._speak_seconds)
        return Speech(sample_rate=_RATE, audio=self._audio())

    def speak_timed(self, voice: Voice, text: str, prosody: Prosody) -> TimedSpeech:
        """The whole utterance at once, counted as the one piece of work it is.

        Eager, like every real `speak_timed`: there is no generator to pull
        later, so the permit around this call covers all of it.
        """
        with self._overlap.one_more():
            time.sleep(self._speak_seconds + _CHUNKS * _CHUNK_SECONDS)
        samples = len(_SAMPLES) // 2
        return TimedSpeech(
            pcm=_SAMPLES,
            sample_rate=_RATE,
            timings=(Timing(samples=samples, separates_words=False),),
        )

    def _audio(self) -> Iterator[bytes]:
        # Counted around each CHUNK, not around the generator, because making a
        # chunk is what the bound is over. A counter wrapping the whole generator
        # would measure in-flight requests instead — a different property, and
        # one this service deliberately does not bound: a stream parked between
        # pulls is waiting on its client's socket, not on the engine.
        for _ in range(_CHUNKS):
            with self._overlap.one_more():
                time.sleep(_CHUNK_SECONDS)
            yield _SAMPLES


def _settings(at_once: int) -> Settings:
    return Settings(
        engine=DeclaredPrepared(),
        engine_name="declared",
        known_engines=frozenset(ENGINES) | {"declared"},
        withheld=frozenset(),
        fallback=DECLARED_VOICES[0].id,
        api_key=None,
        host="127.0.0.1",
        port=0,
        concurrent_syntheses=at_once,
    )


#: Every endpoint that makes audio, because each gates a structurally different
#: shape and one cannot stand for another.
#:
#:   ""                        drains the engine inside one thread
#:   "/stream"                 returns before the audio exists; the rest is
#:                             pulled chunk by chunk by the encoder's pump
#:   "/with-timestamps"        one eager `speak_timed`
#:   "/stream/with-timestamps" one `speak_timed` PER SENTENCE, with a whole
#:                             ffmpeg spawn and a client-paced yield between
#:                             acquisitions -- the only shape here that takes and
#:                             gives back the same permit repeatedly inside one
#:                             request
#:
#: The last of those was untested while the other two were, which is exactly
#: where a bound is worth doubting: a permit reacquired before the previous
#: sentence's synthesis had really finished would look identical from outside.
_SPEAKING_ENDPOINTS = ["", "/stream", "/with-timestamps", "/stream/with-timestamps"]


def _most_overlap(at_once: int, endpoint: str) -> Overlap:
    """Calls `_CALLERS` utterances at once and reports how many ever coincided."""
    overlap = Overlap()
    app = create_app(_settings(at_once), WatchedEngine(overlap))

    with fleet.serving(app) as base:
        voice = DECLARED_VOICES[0].id

        def stream(_: int) -> int:
            # The body is read to completion, because reading it is what drains
            # the engine: a response whose stream nobody pulls has synthesised
            # nothing and would count as a caller that never arrived.
            with httpx.Client(timeout=30.0) as client, client.stream(
                "POST",
                f"{base}/v1/text-to-speech/{voice}{endpoint}",
                json={"text": _TEXT},
            ) as response:
                for _chunk in response.iter_bytes():
                    pass
                return response.status_code

        with ThreadPoolExecutor(max_workers=_CALLERS) as pool:
            codes = list(pool.map(stream, range(_CALLERS)))

    assert codes == [200] * _CALLERS, (
        f"a caller did not get audio ({codes}) — the overlap below is measured over "
        "fewer syntheses than this test thinks it drove"
    )
    return overlap


@pytest.mark.parametrize("endpoint", _SPEAKING_ENDPOINTS)
def test_the_probe_can_see_two_syntheses_overlap(endpoint):
    """Positive control: the harness can observe what the next test forbids.

    Without this, a bound of two passes identically whether it held or whether
    these callers simply never coincided — a green check over a measurement that
    never happened, which is the failure this whole file is written against.
    """
    overlap = _most_overlap(_CALLERS, endpoint)

    assert overlap.most > 2, (
        f"never saw more than {overlap.most} syntheses at once even with the gate "
        f"opened to {_CALLERS} — this probe cannot detect overlap, so the bound "
        "below would pass without being tested"
    )


@pytest.mark.parametrize("endpoint", _SPEAKING_ENDPOINTS)
@pytest.mark.parametrize("at_once", [1, 2, 3])
def test_no_more_syntheses_run_at_once_than_the_setting_allows(at_once, endpoint):
    """[LAW:single-enforcer] The number an operator sets is the number that holds.

    Over several bounds rather than one, because a gate that serialises everything
    satisfies "no more than 3" perfectly while being wrong, and a gate that leaks
    one extra permit satisfies "no more than 1" never but "no more than 3" often.
    """
    overlap = _most_overlap(at_once, endpoint)

    assert overlap.most <= at_once, (
        f"{overlap.most} syntheses ran at once against a bound of {at_once} — "
        f"{_CALLERS} callers reached the engine together, which is how "
        "elvenspeak-piper was OOM-killed"
    )


def test_a_caller_who_hangs_up_mid_utterance_gives_its_permit_back():
    """The leak that would be worse than the bug: permits that never come back.

    `/stream`'s permit outlives its handler by design, which puts its release in
    the response body rather than in a `finally` the handler owns. If that release
    did not happen when a caller disconnects early, every abandoned request would
    retire one permit permanently and the service would wedge after exactly
    `concurrent_syntheses` of them — a slower, more total outage than the OOM this
    bound exists to prevent, and one that needs no burst to trigger.

    Bounded at one permit so a single hang-up is the whole supply: with the leak,
    the call after it never gets a slot and this test fails by timing out rather
    than by asserting.
    """
    overlap = Overlap()
    app = create_app(_settings(1), WatchedEngine(overlap))

    with fleet.serving(app) as base:
        voice = DECLARED_VOICES[0].id
        url = f"{base}/v1/text-to-speech/{voice}/stream"
        body = {"text": _TEXT}

        # Reads one chunk and walks away. Leaving the `stream` block closes the
        # connection under a response that is still being produced, which is what
        # a caller hanging up mid-sentence looks like from here.
        with httpx.Client(timeout=30.0) as client:
            with client.stream("POST", url, json=body) as response:
                for _chunk in response.iter_bytes():
                    break

            # The permit the abandoned call was holding has to be available again.
            # A timeout here IS the failure; the assertion below is just what says
            # so when it is not.
            with client.stream("POST", url, json=body) as second:
                drained = sum(len(chunk) for chunk in second.iter_bytes())

    assert drained > 0, (
        "the call after a hang-up got no audio — the abandoned request's permit "
        "was never given back"
    )


def test_a_caller_who_hangs_up_inside_speak_gives_its_permit_back():
    """A hang-up inside `speak` must not retire the slot it was holding.

    The window is the one that matters: `speak` is where nearly all of a
    streaming request's wall time goes, and a voice UI barging in over the agent
    cancels exactly there. The test above hangs up after the first chunk; this
    one hangs up before any chunk exists.

    HONEST ABOUT WHAT THIS DOES NOT PROVE. It is a property guard, not a
    regression test for an observed bug. The permit-holding shape this file was
    written against — handing the permit to the response body — was measured
    under four hard mid-`speak` hang-ups and returned every permit, and this test
    passes against that shape too. It is kept because the property is worth
    holding whatever the server does underneath, not because it discriminates the
    two designs; what rules the leak out is `api._synthesising`, which releases
    from the worker thread's own `finally` and shields the submitted future so
    that finally always runs — there is no permit left to outlive anything.

    One permit, so a single hang-up is the entire supply: were a slot retired,
    the call after it would never get one and this would fail by timing out.
    """
    overlap = Overlap()
    app = create_app(_settings(1), WatchedEngine(overlap, speak_seconds=1.0))

    with fleet.serving(app) as base:
        voice = DECLARED_VOICES[0].id
        url = f"{base}/v1/text-to-speech/{voice}/stream"
        body = {"text": _TEXT}

        # Gives up long before `speak` returns, so the server is cancelled with
        # no response body ever started.
        with httpx.Client(timeout=0.25) as impatient:
            with pytest.raises(httpx.TimeoutException):
                impatient.post(url, json=body)

        with httpx.Client(timeout=30.0) as client, client.stream(
            "POST", url, json=body
        ) as second:
            drained = sum(len(chunk) for chunk in second.iter_bytes())

    assert drained > 0, (
        "the call after a hang-up inside `speak` got no audio — the cancelled "
        "request's permit was never given back"
    )


def test_the_sentence_loop_really_cycles_its_permit():
    """The per-sentence coverage above must actually exercise several sentences.

    `/stream/with-timestamps` is in `_SPEAKING_ENDPOINTS` for one reason: it is
    the only endpoint that takes and gives back the same permit repeatedly inside
    a single request. That is a property of the REQUEST BODY, not of the
    endpoint — with a body carrying no sentence-final punctuation the loop runs
    once and the whole point is lost, which is what happened when this coverage
    was first added.

    So the shape is asserted rather than assumed: one caller, and the engine has
    to have been entered once per sentence.
    """
    overlap = Overlap()
    app = create_app(_settings(2), WatchedEngine(overlap))

    with fleet.serving(app) as base:
        voice = DECLARED_VOICES[0].id
        with httpx.Client(timeout=30.0) as client, client.stream(
            "POST",
            f"{base}/v1/text-to-speech/{voice}/stream/with-timestamps",
            json={"text": _TEXT},
        ) as response:
            for _chunk in response.iter_bytes():
                pass

    sentences = len(split_sentences(_TEXT))
    assert sentences > 1, f"_TEXT is not multiple sentences: {_TEXT!r}"
    assert overlap.entries == sentences, (
        f"one request entered the engine {overlap.entries} times for {sentences} "
        "sentences — the per-sentence permit cycle this endpoint is here to cover "
        "is not being exercised"
    )


async def _free_permits(gate: asyncio.Semaphore) -> int:
    """How many permits `gate` will hand out right now, counted by taking them.

    [LAW:behavior-not-structure] Rather than reading `asyncio.Semaphore._value`,
    which is CPython's private representation of this fact and not the fact. The
    law does not stop applying because the structure belongs to the standard
    library — and the counter is the wrong observation on its own terms anyway:
    the property under test is what the gate will hand out, and taking them is
    what answers that.

    Every permit taken is given straight back, so the probe leaves the gate as it
    found it. A release arriving from a worker thread mid-count is not lost: the
    count only ever understates by one, and the assertions below are written
    against a settled gate.
    """
    taken = 0
    while True:
        try:
            await asyncio.wait_for(gate.acquire(), timeout=0.05)
        except (TimeoutError, asyncio.TimeoutError):
            break
        taken += 1
    for _ in range(taken):
        gate.release()
    return taken


@pytest.mark.parametrize(
    "occupy_the_pool",
    # The two states a cancellation can find the submitted work in, and they fail
    # in opposite directions if the release is wired wrong: RUNNING releases too
    # early (the slot is handed on while the engine is still in it), PENDING
    # never releases at all (the work is cancelled outright, so a `finally`
    # inside it never runs). One mechanism has to answer both.
    [pytest.param(False, id="cancelled-while-running"),
     pytest.param(True, id="cancelled-while-pending")],
)
def test_a_cancelled_synthesis_gives_its_permit_back_exactly_once(occupy_the_pool):
    """[LAW:no-silent-failure] A permit that is never returned is an outage on a timer.

    Driven directly rather than through the server because the PENDING case needs
    the executor saturated, which a test cannot arrange through HTTP but can
    arrange exactly here with a one-worker pool. Measured before it was believed:
    with the release inside the submitted callable and nothing shielding it, a
    gate of two came back one and stayed there for the life of the loop.

    That is the worse of the two failure directions. Over-subscription is
    transient and self-heals; a lost permit is monotonic, so a service leaks its
    way to a gate of zero and stops synthesising while looking perfectly healthy.
    """

    async def run() -> tuple[int, bool]:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        gate = asyncio.Semaphore(2)
        finished = threading.Event()

        def work() -> str:
            time.sleep(0.4)
            finished.set()
            return "spoken"

        if occupy_the_pool:
            asyncio.create_task(asyncio.to_thread(lambda: time.sleep(0.6)))
            await asyncio.sleep(0.05)

        task = asyncio.create_task(api._synthesising(gate, work))
        await asyncio.sleep(0.15)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

        # Read before the work can have finished: a permit back this early means
        # it was handed on while the engine was still holding it.
        await asyncio.sleep(0.02)
        released_early = await _free_permits(gate) == 2 and not finished.is_set()

        await asyncio.sleep(2.0)
        return await _free_permits(gate), released_early

    permits, released_early = asyncio.run(run())

    assert not released_early, (
        "the permit came back before the synthesis it was guarding had finished — "
        "a cancelled caller handed its slot to the next one mid-utterance"
    )
    assert permits == 2, (
        f"{permits} of 2 permits came back after a cancelled synthesis — the rest "
        "are gone for the life of the process, and the gate shrinks toward zero"
    )


def test_a_synthesis_that_fails_after_its_caller_left_still_says_so(caplog):
    """[LAW:no-silent-failure] A hang-up must not be a way to lose an engine failure.

    `asyncio.shield` unregisters its inner callback the moment the outer await is
    cancelled, so nothing reads what the shielded work did next. The permit still
    comes back — that is the `finally` — but an engine that then RAISES has its
    exception discarded, resurfacing whenever the task is collected as a bare
    "Task exception was never retrieved" with nothing tying it to a request.
    Measured; the loop's exception handler saw exactly that message and nothing
    else.

    The caller is gone either way. What has to survive is the operator learning
    that synthesis broke, which is the difference between a fault that gets fixed
    and one that is only ever seen as a graph.
    """

    async def run() -> None:
        gate = asyncio.BoundedSemaphore(1)

        def work() -> str:
            time.sleep(0.3)
            raise RuntimeError("engine blew up after the caller left")

        task = asyncio.create_task(api._synthesising(gate, work))
        await asyncio.sleep(0.1)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        # Long enough for the orphaned thread to finish and be read.
        await asyncio.sleep(1.0)

    with caplog.at_level(logging.ERROR, logger="elvenspeak.api"):
        asyncio.run(run())

    # Filtered to this service's own logger, and that is not fussiness. Asyncio's
    # GC-time "Task exception was never retrieved" carries the whole traceback,
    # so an unfiltered search for the message text passes whether or not anything
    # here reported it — the test would have been green against the very bug it
    # exists to catch. Verified: it was.
    ours = [r for r in caplog.records if r.name == "elvenspeak.api"]
    assert any("engine blew up after the caller left" in r.getMessage() for r in ours), (
        "an engine failure that landed after its caller hung up was never "
        f"reported by elvenspeak.api — its records: {[r.getMessage() for r in ours]}"
    )
