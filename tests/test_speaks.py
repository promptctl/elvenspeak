"""What `speaks.py` refuses, driven against a server built to get it wrong.

`speaks.py` is the artifact-side half of the seam's contract: since
piper-pipeline-4mx the three engines that own a model are no longer subjects of
`tests/test_conformance.py`, and the properties they used to answer there are
asked of the running container instead. That makes this file the thing standing
between a real defect and a green publish, so the question it has to settle is
not "does it pass against a good server" — a check that has never gone red gates
nothing ([LAW:no-silent-failure]) — but "does it go red against each specific
way an engine can be wrong".

So every test below serves a deliberately broken answer and asserts the refusal
names it. The happy path is here too, as the positive control that keeps the
rest honest: without it, a `conform` that raised unconditionally would pass every
other test in this file.

# Why a real socket

`speaks.py` speaks HTTP over `urllib` to an address, because in CI it is talking
to a container. Substituting a transport here would test a `speaks.py` that does
not exist. `http.server` in a thread is the smallest thing that is genuinely the
same code path, and it needs no model, no network and no image — which is the
whole point of the move this file exists to protect.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import speaks

#: A voice that can do everything, as `/v1/voices` publishes one.
EVERYTHING = {"voice_id": "able", "capabilities": ["speed", "timestamps"]}

#: A voice that can do nothing, which is a real answer rather than a broken one —
#: `elvenspeak/chatterbox.py` ships exactly this.
NOTHING = {"voice_id": "plain", "capabilities": []}


@dataclass
class Fake:
    """A server that answers `speaks.py` however a test needs it to.

    [LAW:dataflow-not-control-flow] Every deviation is a *value* on this object
    rather than a subclass or a flag inside the handler, so the handler below has
    one shape and each test says what is wrong by naming a field. Adding the next
    kind of brokenness is a field, never a branch.
    """

    voices: list[dict] = field(default_factory=lambda: [dict(EVERYTHING)])

    #: Served on the second `/v1/voices` call, when a test wants the catalogue to
    #: change under `speaks.py` the way an engine rebuilding its list would.
    voices_again: list[dict] | None = None

    #: Samples returned per character of text, so "more text makes more audio"
    #: holds by default.
    samples_per_character: int = 100

    #: A sample count that ignores the text entirely, which is the engine that
    #: returns something plausible without having spoken what it was given. Not
    #: expressed as `samples_per_character=0`: that returns no audio at all, and
    #: is caught one check earlier as a voice that answered 200 with nothing.
    fixed_samples: int | None = None

    #: Multiplier applied when a speed is asked for. 0.5 is a real halving; 1.0
    #: is the engine that accepted the parameter and did nothing.
    speed_effect: float = 0.5

    #: Whether a speed that was not honoured gets named back to the caller.
    names_ignored_speed: bool = True

    #: An odd byte count, which is not a whole number of 16-bit samples.
    odd_bytes: bool = False

    #: Status for the timestamps endpoint; 501 is the refusal a voice that does
    #: not measure owes.
    timestamps_status: int = 200

    #: End times served for the alignment, when one is served at all.
    ends: list[float] | None = None

    _calls: int = 0

    def voice_listing(self) -> dict:
        self._calls += 1
        if self._calls > 1 and self.voices_again is not None:
            return {"voices": self.voices_again}
        return {"voices": self.voices}

    def audio(self, text: str, speed: float | None) -> bytes:
        count = (
            self.fixed_samples
            if self.fixed_samples is not None
            else len(text) * self.samples_per_character
        )
        if speed is not None:
            count = int(count * self.speed_effect)
        return b"\x00" * (count * speaks.BYTES_PER_SAMPLE + (1 if self.odd_bytes else 0))

    def alignment(self, text: str) -> dict:
        ends = self.ends if self.ends is not None else [
            round(0.1 * (index + 1), 3) for index in range(len(text))
        ]
        return {
            "alignment": {
                "characters": list(text),
                "character_end_times_seconds": ends,
            }
        }


def _handler(fake: Fake) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args) -> None:
            """Silence: a passing test should not print an access log."""

        def _send(self, status: int, payload: bytes, headers: dict[str, str]) -> None:
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802 - http.server's spelling
            if self.path.startswith("/v1/voices"):
                body = json.dumps(fake.voice_listing()).encode()
                self._send(200, body, {"content-type": "application/json"})
                return
            self._send(404, b"", {})

        def do_POST(self) -> None:  # noqa: N802 - http.server's spelling
            length = int(self.headers.get("content-length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            text = body.get("text", "")
            speed = (body.get("voice_settings") or {}).get("speed")

            if "/with-timestamps" in self.path:
                if fake.timestamps_status != 200:
                    self._send(fake.timestamps_status, b'{"detail":"no"}',
                               {"content-type": "application/json"})
                    return
                payload = json.dumps(fake.alignment(text)).encode()
                self._send(200, payload, {"content-type": "application/json"})
                return

            headers = {"content-type": "audio/pcm"}
            # Named back only when the engine did not act on it, which is the
            # same rule `elvenspeak/api.py` follows.
            if speed is not None and fake.speed_effect == 1.0 and fake.names_ignored_speed:
                headers["x-elvenspeak-ignored"] = "voice_settings.speed"
            self._send(200, fake.audio(text, speed), headers)

    return Handler


@contextmanager
def serving(fake: Fake) -> Iterator[str]:
    """`fake` answering on a real loopback port, torn down afterwards."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(fake))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def refusal(fake: Fake) -> str:
    """The message `conform` refuses `fake` with.

    [LAW:parse-dont-validate] A helper rather than a `pytest.raises` in every
    test, so that a test which *fails to refuse* reports that rather than an
    attribute error on the exception it did not get.
    """
    with serving(fake) as url:
        with pytest.raises(speaks.ConformanceFailure) as raised:
            speaks.conform(url, timeout=5.0)
    return str(raised.value)


# ------------------------------------------------------------ positive control


def test_a_server_that_keeps_every_promise_is_accepted():
    """The control the rest of this file rests on.

    Without it, a `conform` that raised on everything would satisfy every other
    test here — which is the exact shape of a check that looks thorough and
    proves nothing.
    """
    with serving(Fake()) as url:
        speaks.conform(url, timeout=5.0)


def test_a_voice_that_declares_nothing_is_also_accepted():
    """Declaring no capability is a real answer, not a broken engine.

    Chatterbox declares the empty set, so an implementation that treated
    "capabilities: []" as a defect would refuse a shipping image.
    """
    fake = Fake(
        voices=[dict(NOTHING)], speed_effect=1.0, timestamps_status=501
    )
    with serving(fake) as url:
        speaks.conform(url, timeout=5.0)


# ------------------------------------------------------- the catalogue's promises


def test_a_catalogue_that_changes_between_calls_is_refused():
    """The voice list is a promise, and a server rebuilding it is breaking one."""
    message = refusal(Fake(voices_again=[dict(EVERYTHING), dict(NOTHING)]))
    assert "changed between two calls" in message


def test_two_voices_sharing_an_id_are_refused():
    """A caller naming the id cannot be answered predictably, so it is not legal."""
    message = refusal(Fake(voices=[dict(EVERYTHING), dict(EVERYTHING)]))
    assert "offered more than once" in message
    assert "able" in message


def test_an_empty_catalogue_is_refused():
    """A server with nothing to say should not have answered /health 200."""
    message = refusal(Fake(voices=[]))
    assert "empty list" in message


# ---------------------------------------------------------- what speaking proves


def test_audio_that_is_not_whole_samples_is_refused():
    """`pcm_*` is raw signed 16-bit, so an odd byte count is not that format."""
    message = refusal(Fake(odd_bytes=True))
    assert "not a whole number of 16-bit samples" in message


def test_more_text_that_does_not_make_more_audio_is_refused():
    """The engine is not speaking what it was given, whatever it returned."""
    message = refusal(Fake(fixed_samples=2000))
    assert "more text did not make more audio" in message


def test_a_declared_speed_that_does_nothing_is_refused():
    """The silent drop the declaration exists to rule out."""
    message = refusal(Fake(speed_effect=1.0))
    assert "accepted and did nothing" in message


def test_a_speed_honoured_without_being_declared_is_refused():
    """The same defect from the other side, and the arm a one-sided check misses.

    A voice that applies a parameter it disclaimed cannot be described accurately
    to any caller, so this is a failure even though the audio "worked".
    """
    message = refusal(
        Fake(voices=[dict(NOTHING)], speed_effect=0.5, timestamps_status=501)
    )
    assert "honoured it" in message


def test_a_dropped_speed_that_is_not_named_back_is_refused():
    """[LAW:no-silent-failure] Accepted-and-dropped is reported, not hidden."""
    message = refusal(
        Fake(
            voices=[dict(NOTHING)],
            speed_effect=1.0,
            names_ignored_speed=False,
            timestamps_status=501,
        )
    )
    assert "x-elvenspeak-ignored" in message


# ----------------------------------------------------- what timings owe the caller


def test_timestamps_refused_by_a_voice_that_declared_them_is_refused():
    """A capability is a promise the endpoint has to keep."""
    message = refusal(Fake(timestamps_status=501))
    assert "declares" in message
    assert "501" in message


def test_timestamps_answered_by_a_voice_that_declared_none_is_refused():
    """Numbers that cannot have been measured, handed to a caller as if they were.

    This is the defect `Capability.TIMESTAMPS` was added to prevent, and the one
    an engine is most tempted to commit: inventing an even spread is easy and
    looks exactly like a measurement.
    """
    message = refusal(
        Fake(voices=[dict(NOTHING)], speed_effect=1.0, timestamps_status=200)
    )
    assert "does not declare" in message
    assert "cannot have been measured" in message


def test_an_alignment_that_goes_backwards_is_refused():
    """A timeline that is not monotonic does not describe an utterance."""
    message = refusal(Fake(ends=[0.5, 0.2, 0.9] + [1.0] * (len(speaks.TEXT) - 3)))
    assert "go backwards" in message


def test_an_empty_alignment_is_refused():
    """Text went in, so a timeline covering none of it is not an answer."""
    message = refusal(Fake(ends=[]))
    assert "empty alignment" in message


# ------------------------------------------------------------- the transport itself


def test_a_server_that_is_not_there_is_refused_by_name():
    """[LAW:no-silent-failure] The failure names the request, not just `None`.

    Whoever reads this in a publish log is looking at a container that did
    something unexpected, and the first thing they need is which call went
    unanswered.
    """
    with pytest.raises(speaks.ConformanceFailure) as raised:
        speaks.conform("http://127.0.0.1:1", timeout=2.0)
    assert "/v1/voices" in str(raised.value)


def test_a_listing_without_the_fields_it_promises_is_refused():
    """A malformed catalogue is reported as that, not as a KeyError later on."""
    message = refusal(Fake(voices=[{"voice_id": "able"}]))
    assert "capabilities" in message
