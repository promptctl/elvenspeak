"""How many syntheses this process runs at once, asserted through the real surface.

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

import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import fleet
import httpx
import pytest
from conftest import DECLARED_VOICES, DeclaredPrepared, declaring

from elvenspeak import create_app
from elvenspeak.engine import Capability, Prosody, Speech, Voice
from elvenspeak.engines import ENGINES
from elvenspeak.settings import Settings

#: Enough callers to exceed every bound under test, so the ceiling is what the
#: setting says rather than what the test could muster.
_CALLERS = 6

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

    @contextmanager
    def one_more(self) -> Iterator[None]:
        with self._lock:
            self._now += 1
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

    def __init__(self, overlap: Overlap) -> None:
        self._overlap = overlap
        # TIMESTAMPS deliberately not declared: this stand-in has no
        # `speak_timed`, and an engine that declares what it cannot do is the
        # dishonesty `DeclaredEngine`'s docstring exists to refuse. It also keeps
        # the endpoints below to the two that need only `speak`.
        self._voices = declaring(frozenset({Capability.SPEED}), DECLARED_VOICES)

    def voices(self) -> tuple[Voice, ...]:
        return self._voices

    def speak(self, voice: Voice, text: str, prosody: Prosody) -> Speech:
        return Speech(sample_rate=_RATE, audio=self._audio())

    def _audio(self) -> Iterator[bytes]:
        with self._overlap.one_more():
            for _ in range(_CHUNKS):
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


#: The two endpoints that need only `speak`, and they are gated differently on
#: purpose — which is why both are driven rather than one standing for the pair.
#: `/convert` drains the engine inside its own thread, so its permit is an
#: ordinary `async with` in the handler. `/stream` returns before the audio
#: exists, so its permit is handed to the response body. A single test over
#: either one alone would leave the other's shape unmeasured.
_SPEAKING_ENDPOINTS = ["", "/stream"]


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
                json={"text": "one two three four"},
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
        body = {"text": "one two three four"}

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
