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

import base64
import itertools
import json
import threading
import time
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

    #: Samples added to each successive synthesis, per voice, so no two of that
    #: voice's answers are the same length and two identical requests never come
    #: back identical. What a model that samples does — the spread of one text's
    #: draws is measured in `speaks.APART`, and it is why `speaks.py` reads no
    #: length against another unless the voice repeats itself. Negative counts
    #: down, which is how a test puts a later draw below an earlier one without
    #: depending on the order four threads happen to run in.
    #:
    #: Per voice rather than per server because that is the case that matters:
    #: behind the router one voice's backend samples while the next one's does
    #: not, in one deployment, and a server that could only be all of one or all
    #: of the other could not have caught reading one voice's repeatability off
    #: another.
    redraws: dict[str, int] = field(default_factory=dict)

    #: Voices whose successive draws carry different waveforms at whatever length
    #: they came out — the other half of what a sampler does, and the half
    #: [`redraws`] cannot express. `speaks._repeatable` compares audio rather than
    #: sample counts precisely because counts are quantised to whole frames and
    #: two draws land on one length by coincidence, never on one waveform. A
    #: server that could only vary its lengths made equal length equal bytes, so
    #: a `_repeatable` rewritten to compare `.samples` passed every test here
    #: while reinstating the flake it was written to retire: the claim lived only
    #: in a docstring, and a claim only a comment defends is one that decays.
    rewaveforms: set[str] = field(default_factory=set)

    #: Draws of [`silent_on`] answered before the silence starts. 0 is a text this
    #: server never speaks. 1 is the intermittent shape Kokoro actually had — 15
    #: of 16 short lines silent — and it is the one that reaches the repeatability
    #: probe, whose second draw of `SHORT_TEXT` is what decides whether the
    #: voice's lengths are read at all. A server silent from the first draw is
    #: refused in the loop above and never gets that far, so it could not have
    #: caught a glitched probe buying a voice an exemption from its own checks.
    answers_before_silence: int = 0

    #: Multiplier applied when a speed is asked for. 0.5 is a real halving; 1.0
    #: is the engine that accepted the parameter and did nothing.
    speed_effect: float = 0.5

    #: Whether a speed that was not honoured gets named back to the caller.
    names_ignored_speed: bool = True

    #: An odd byte count, which is not a whole number of 16-bit samples.
    odd_bytes: bool = False

    #: Bytes a streamed answer promises and never sends: a 200 whose body stops
    #: short. What an image that cannot encode answers, because the stream's
    #: headers are out before the encoder is found missing — measured on
    #: elvenspeak-piper and -router 2026.09.08.1 run with ffmpeg off their PATH.
    cut_short_by: int = 0

    #: A text answered with no audio at all. Kokoro's measured zero-sample defect
    #: on short input, in the shape it would reach a caller if `Silence` did not
    #: catch it first: a 200 carrying nothing.
    silent_on: str | None = None

    #: Texts answered at another text's length, per voice —
    #: `{(voice_id, sent): measured_as}`. The crossed response, which is what a
    #: caller reading somebody else's answer off a shared socket receives; per
    #: voice because what two callers cross is the speakers they named.
    answers_as: dict[tuple[str, str], str] = field(default_factory=dict)

    #: The fraction of the utterance the timeline claims to cover. 1.0 is a
    #: timeline that ends where the audio ends; anything less is one that stops
    #: short, which is the drift `elvenspeak/alignment.py` closes deliberately.
    covers: float = 1.0

    #: Whether the timestamps body arrives without the audio its timeline
    #: measured, leaving the numbers with nothing to be checked against.
    omits_audio: bool = False

    #: The draws [`redraws`], [`rewaveforms`] and [`answers_before_silence`] step,
    #: and the one lock they are stepped under. The concurrent section puts four
    #: callers inside this server at once, and two of them advancing one unguarded
    #: counter would be the fake inventing a race of its own to then fail on.
    #: Taken and released around each `next` rather than held across the draw,
    #: because a lock held while another is taken is the deadlock a test double
    #: has no business teaching anyone about.
    _drawn: Iterator[int] = field(default_factory=itertools.count, init=False, repr=False)
    _waved: Iterator[int] = field(default_factory=itertools.count, init=False, repr=False)
    _hushed: Iterator[int] = field(default_factory=itertools.count, init=False, repr=False)
    _drawing: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    #: Status for the timestamps endpoint; 501 is the refusal a voice that does
    #: not measure owes.
    timestamps_status: int = 200

    #: End times served for the alignment, when one is served at all.
    ends: list[float] | None = None

    #: Seconds each character of text occupies the engine, which is held to a
    #: single caller at a time — `ELVENSPEAK_CONCURRENT_SYNTHESES=1`, a correct
    #: and common deployment, in the smallest shape that reproduces it. Per
    #: character because a real engine's work is the text it was given. At 0.0
    #: the engine is occupied for no time, so the serialisation is unobservable
    #: and every other test here is answered as immediately as before.
    seconds_per_character: float = 0.0

    #: The engine itself: one caller inside it at a time. Taken unconditionally
    #: ([LAW:dataflow-not-control-flow]) so there is no second code path that
    #: only the timing test runs.
    engine: threading.Lock = field(default_factory=threading.Lock, repr=False)

    _calls: int = 0

    def _stepped(self, counter: Iterator[int]) -> int:
        """The next value of one of this server's draw counters, taken once.

        [LAW:single-enforcer] The one place a counter is advanced, so "guarded by
        `_drawing`" is a fact about this method rather than a convention three
        call sites keep. Three counters and one lock: each orders a draw against
        the others of its own kind, never against another kind.
        """
        with self._drawing:
            return next(counter)

    def voice_listing(self) -> dict:
        self._calls += 1
        if self._calls > 1 and self.voices_again is not None:
            return {"voices": self.voices_again}
        return {"voices": self.voices}

    def samples(self, voice: str, text: str, speed: float | None) -> int:
        """How many samples one synthesis of `text` runs to.

        [LAW:one-source-of-truth] The streamed audio, the audio inside the
        timestamped body and the timeline measured over it are three renderings
        of one utterance, so all three are divided out of this. Stated apart they
        drifted by construction — the alignment ran 2.4s over 0.109s of audio,
        which no server could return and which the accounting `speaks.py` now
        checks could never have been asserted against.
        """
        measured = self.answers_as.get((voice, text), text)
        count = (
            self.fixed_samples
            if self.fixed_samples is not None
            else len(measured) * self.samples_per_character
        )
        if speed is not None:
            count = int(count * self.speed_effect)
        if text == self.silent_on and self._stepped(self._hushed) >= self.answers_before_silence:
            return 0
        step = self.redraws.get(voice, 0)
        if not step:
            return count
        return count + step * self._stepped(self._drawn)

    def audio(self, voice: str, text: str, speed: float | None) -> bytes:
        """The bytes one synthesis returns, filled with one repeated sample.

        The filler is what a draw of a [`rewaveforms`] voice varies: a different
        byte each time, so two draws of one voice can land on the same sample
        count and still be two different utterances. Every other voice answers in
        zeros, which is what this has always returned — the filler is a *value*
        this method reads and never a second way of building a body
        ([LAW:dataflow-not-control-flow]).

        Counted before the filler is drawn, because [`samples`] takes the same
        lock and a single `threading.Lock` does not nest.
        """
        counted = self.samples(voice, text, speed) * speaks.BYTES_PER_SAMPLE + (
            1 if self.odd_bytes else 0
        )
        filler = self._stepped(self._waved) % 256 if voice in self.rewaveforms else 0
        return bytes([filler]) * counted

    def timed(self, voice: str, text: str) -> dict:
        """The body `/with-timestamps` returns: the audio and its timeline.

        The boundaries are divided out of the *sample count* so the last lands
        exactly on the last sample, the same arithmetic `fleetstub.py` does and
        for the same reason — a fake that relied on the slack `speaks.py` allows
        would be a fake whose alignment is quietly wrong wherever the slack hides it.
        """
        spoken = self.audio(voice, text, None)
        # Counted off the very bytes carried below rather than drawn a second
        # time. Two calls agreed only while a synthesis was a pure function of
        # its request, which [`redraws`] is precisely the end of — and a timeline
        # divided out of one draw while the audio came from another is the drift
        # this method's docstring exists to rule out.
        samples = len(spoken) // speaks.BYTES_PER_SAMPLE
        spread = [
            self.covers * (index + 1) * samples / len(text) / speaks.SAMPLE_RATE
            for index in range(len(text))
        ]
        carried = {} if self.omits_audio else {
            "audio_base64": base64.b64encode(spoken).decode("ascii")
        }
        return {
            **carried,
            "alignment": {
                "characters": list(text),
                "character_end_times_seconds": (
                    self.ends if self.ends is not None else spread
                ),
            },
        }


def _handler(fake: Fake) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args) -> None:
            """Silence: a passing test should not print an access log."""

        def _send(
            self, status: int, payload: bytes, headers: dict[str, str], missing: int = 0
        ) -> None:
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            # Promising `missing` bytes past what is written is how a body that
            # stops short looks on the wire: the connection closes still owing them.
            self.send_header("content-length", str(len(payload) + missing))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802 - http.server's spelling
            if self.path.startswith("/v1/voices"):
                body = json.dumps(fake.voice_listing()).encode()
                self._send(200, body, {"content-type": "application/json"})
                return
            self._send(404, b"", {})

        def do_POST(self) -> None:  # noqa: N802 - http.server's spelling
            match self.path.partition("?")[0].split("/"):
                case ["", "v1", "text-to-speech", voice, ("stream" | "with-timestamps") as endpoint]:
                    self._synthesize(voice, endpoint)
                case _:
                    self._send(404, b"", {})

        def _synthesize(self, voice: str, endpoint: str) -> None:
            length = int(self.headers.get("content-length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            text = body.get("text", "")
            speed = (body.get("voice_settings") or {}).get("speed")

            with fake.engine:
                time.sleep(fake.seconds_per_character * len(text))

            if endpoint == "with-timestamps":
                if fake.timestamps_status != 200:
                    self._send(fake.timestamps_status, b'{"detail":"no"}',
                               {"content-type": "application/json"})
                    return
                payload = json.dumps(fake.timed(voice, text)).encode()
                self._send(200, payload, {"content-type": "application/json"})
                return

            headers = {"content-type": "audio/pcm"}
            # Named back when the voice spoken did not declare it, whatever length
            # the audio came back — the rule `elvenspeak/api.py` follows, which
            # decides from the declaration and never from the audio.
            declares = any(
                listed["voice_id"] == voice and "speed" in listed["capabilities"]
                for listed in fake.voices
            )
            if speed is not None and not declares and fake.names_ignored_speed:
                headers["x-elvenspeak-ignored"] = "voice_settings.speed"
            self._send(200, fake.audio(voice, text, speed), headers, missing=fake.cut_short_by)

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


def test_a_voice_that_goes_silent_on_a_short_utterance_is_refused():
    """The one defect a paragraph-length check cannot see.

    Kokoro returned nothing for 15 of 16 one- and two-word Spanish lines while
    reading a paragraph perfectly, so a conformance that only ever asked for
    `TEXT` would have called that engine conformant. The refusal has to name the
    text as well as the voice — "answered with nothing" sends a reader to the
    voice, and the fault is in what it was asked to say.
    """
    message = refusal(Fake(silent_on=speaks.SHORT_TEXT))
    assert repr(speaks.SHORT_TEXT) in message
    assert "no audio at all" in message


def test_a_voice_that_goes_silent_only_on_the_repeatability_probe_is_refused(capsys):
    """A glitched probe must not buy a voice an exemption from its own checks.

    The probe draws `SHORT_TEXT` a second time to ask whether the voice answers
    an identical request identically, and that answer decides whether any length
    of that voice is read against another. So a voice that answered the first
    draw and goes silent on the second — Kokoro's intermittent defect, one draw
    in sixteen speaking — differs from itself for a reason that is not sampling,
    and reading that as "this voice samples" switches off the very comparison
    that would have caught it. A defect disabling its own detector, which is the
    shape of the bug this whole check was written to retire.

    Refused as the silence it is, then: the probe is drawn through `_speak` like
    every other synthesis in the file and audited before anything reads it.

    The verdict is what this asserts, and the message alone would not have. Left
    unaudited, the probe's silence still ended in a refusal naming this very text
    — three draws later, from the concurrent section, which audited its own
    answers — so a test reading only the message passed against the defect it was
    written for. What separates the two is whether this voice was ever called a
    sampler: that line is printed only by a run that read an empty probe as a
    redraw, and the exemption is granted at the moment it is printed
    ([LAW:behavior-not-structure]).
    """
    message = refusal(Fake(silent_on=speaks.SHORT_TEXT, answers_before_silence=1))
    assert repr(speaks.SHORT_TEXT) in message
    assert "no audio at all" in message
    assert "with two different utterances" not in capsys.readouterr().out


def test_a_voice_that_goes_silent_when_asked_to_speak_faster_is_refused():
    """Zero samples is shorter than three quarters of anything.

    The pace check asks whether a declared `speed` shortened the audio, and an
    engine that answers a speeded request with no audio at all satisfies it
    completely — measured: `conform` passed this server and printed "speed varies
    the pace as declared (2400 -> 0 samples at 2.0x)" about an answer carrying
    nothing. A check a defect passes by being worse than the defect it looks for
    gates nothing ([LAW:no-silent-failure]).

    Served as `speed_effect=0.0` rather than a field of its own: "the engine's
    speed handling returns no audio" is already sayable as a value this fake
    reads, and the deviation belongs in the value ([LAW:no-mode-explosion]).
    """
    message = refusal(Fake(speed_effect=0.0))
    assert "no audio at all" in message


def test_a_silence_on_the_longer_text_is_named_as_silence_not_as_a_short_answer():
    """The refusal has to name the defect, not the first check it happens to trip.

    `LONGER_TEXT` is drawn only by the two comparisons, never by the loop that
    speaks every voice, so a voice silent on it arrives at a length comparison
    first. Unaudited, that read `0 <= 400` and refused it as "more text did not
    make more audio" — true of the numbers and wrong about the engine, sending a
    reader after a length defect in a server that had gone silent. Two different
    faults are not owed one message.
    """
    message = refusal(Fake(silent_on=speaks.LONGER_TEXT))
    assert repr(speaks.LONGER_TEXT) in message
    assert "no audio at all" in message
    assert "more text did not make more audio" not in message


def test_a_declared_speed_that_does_nothing_is_refused():
    """The silent drop the declaration exists to rule out."""
    message = refusal(Fake(speed_effect=1.0))
    assert "accepted and did nothing" in message


def test_a_voice_that_declares_no_speed_is_not_judged_by_its_length():
    """A disclaimed speed owes a name in the ignored header, not a length.

    The server withholds it, so the engine is asked the same thing twice and a
    sampling engine's two answers can land well apart. Chatterbox samples, and
    elvenspeak-chatterbox:2026.09.08.1 was refused as "does not declare `speed`
    and yet honoured it" on exactly this shape: the speed withheld and named,
    one draw shorter than the other.
    """
    fake = Fake(voices=[dict(NOTHING)], speed_effect=0.6, timestamps_status=501)
    with serving(fake) as url:
        speaks.conform(url, timeout=5.0)


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


def test_a_timeline_that_stops_short_of_its_audio_is_refused():
    """The check that reads the alignment against the audio rather than itself.

    `engine.TimedSpeech` promises its timings sum to the audio's sample count and
    `elvenspeak/alignment.py` ends the last character exactly there, so a
    half-length timeline is a broken promise — and one that every other check
    here passes, because non-empty and ascending are both true of it. It is also
    the defect with a downstream: `/stream/with-timestamps` lays each sentence
    after where the last one said it ended, so the shortfall compounds.
    """
    message = refusal(Fake(covers=0.5))
    assert "does not account for the utterance it describes" in message


def test_an_alignment_answered_without_the_audio_it_measured_is_refused():
    """A timeline handed over with nothing to check it against.

    Refused as a malformed answer rather than surfacing as a `KeyError` inside
    the accounting check, which is the same rule the catalogue is parsed under.
    """
    message = refusal(Fake(omits_audio=True))
    assert "decodable audio" in message


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


def test_a_stream_cut_off_after_its_200_is_refused_by_name_not_traced():
    """[LAW:no-silent-failure] A body that stops short is a refusal, not a traceback.

    The one failure a smoke of a healthy image that cannot speak actually
    produces: the status was already 200, so there is no refusal to catch, only
    a body that ends early — `http.client.IncompleteRead`, which is not an
    `OSError`. `_speak` caught socket errors alone, and on
    elvenspeak-piper:2026.09.08.1 with ffmpeg off its PATH `conform` left as a
    bare traceback naming neither the voice nor the text.
    """
    message = refusal(Fake(cut_short_by=1))
    assert repr(speaks.TEXT) in message
    assert "IncompleteRead" in message


def test_a_listing_without_the_fields_it_promises_is_refused():
    """A malformed catalogue is reported as that, not as a KeyError later on."""
    message = refusal(Fake(voices=[{"voice_id": "able"}]))
    assert "capabilities" in message


# ------------------------------------------------------- what contention proves


def test_two_voices_asked_at_once_are_each_answered():
    """The positive control for the concurrent section, on the shape it targets.

    Every other happy-path test here offers one voice, so `subjects` is that
    voice twice and the two-different-speakers case — the one Chatterbox's
    `conds` write makes dangerous — never runs. Two voices puts it under the
    check that is supposed to cover it.
    """
    fake = Fake(voices=[dict(EVERYTHING), {"voice_id": "other", "capabilities": ["speed", "timestamps"]}])
    with serving(fake) as url:
        speaks.conform(url, timeout=5.0)


def test_a_concurrent_caller_answered_at_the_wrong_length_is_refused():
    """The crossed response: a plausible utterance that is not the one asked for.

    Crossed for the last voice on offer, whose short text comes back at the long
    one's length. Every serial comparison of two lengths is asked of the first
    voice, so `_conform_concurrently` is the only place in `speaks.py` where the
    last voice's two lengths are read against each other. Served on every request
    rather than only the concurrent ones, and that is deliberate: a fake that
    misbehaved only while two callers overlapped would be asserting its own
    timing rather than the refusal — and would go green whenever the overlap it
    depends on failed to happen ([LAW:no-ambient-temporal-coupling]).
    """
    fake = Fake(
        voices=[dict(EVERYTHING), {"voice_id": "other", "capabilities": ["speed", "timestamps"]}],
        answers_as={("other", speaks.SHORT_TEXT): speaks.LONGER_TEXT},
    )
    message = refusal(fake)
    assert "callers at once" in message
    assert "crossed or truncated" in message


def test_a_deployment_that_redraws_is_not_held_to_lengths_it_never_promised(capsys):
    """The long tail, which is sampling and not a crossing.

    Chatterbox forces EOS on a long tail and pads a short utterance toward a
    ceiling the long text is bounded by too, so an engine that samples answers
    four characters at more length than sixty-five with nothing crossed and no
    caller truncated — and refusing that would refuse every image chatterbox can
    build. Every leg measured does it, the greenest of them most of all; the
    draws that say so are in `speaks.APART`, which is the one place they are
    written down.

    Served with the same inversion
    [`test_a_concurrent_caller_answered_at_the_wrong_length_is_refused`] is
    refused for, which is the entire point of the pair: one deployment repeats an
    identical request and is held to its lengths, this one redraws and is not,
    and the two verdicts differ on that fact alone. Inverted for the *first*
    voice, so the serial comparison is put under the same question as the
    concurrent one — both read `APART`, and a premise fixed in one of them would
    have gone on failing in the other.
    """
    fake = Fake(
        voices=[dict(NOTHING), {"voice_id": "other", "capabilities": []}],
        answers_as={(NOTHING["voice_id"], speaks.SHORT_TEXT): speaks.LONGER_TEXT},
        redraws={NOTHING["voice_id"]: -10, "other": -10},
        # Neither voice declares `timestamps`, which is Chatterbox's own
        # declaration, so the endpoint owes the 501 that goes with it.
        timestamps_status=501,
    )
    with serving(fake) as url:
        speaks.conform(url, timeout=5.0)

    # A comparison that quietly did not run reads exactly like one that ran and
    # passed, so the skip is only honest if it is said out loud.
    printed = capsys.readouterr().out
    assert "plain answered one request twice with two different utterances" in printed
    assert "its lengths are not read against each other" in printed
    assert "a length was read against another for no voice here" in printed


def test_a_steady_voice_is_still_held_to_its_lengths_behind_a_sampling_one():
    """One voice's redrawing must not excuse the next voice's crossed answer.

    The router puts these two in one deployment: each voice reaches its own
    backend (`elvenspeak/router.py` sends `speak` to `self._speakers[voice.id]`),
    so the first can sample while the last runs a fixed graph. Repeatability read
    once off `voices[0]` and spent on both would skip the comparison here — and
    what it skips is a real crossed response on a voice that never redraws
    anything, which is the whole defect `_conform_concurrently` exists to catch.
    """
    fake = Fake(
        voices=[dict(NOTHING), {"voice_id": "steady", "capabilities": []}],
        redraws={NOTHING["voice_id"]: -10},
        answers_as={("steady", speaks.SHORT_TEXT): speaks.LONGER_TEXT},
        timestamps_status=501,
    )
    message = refusal(fake)
    assert "callers at once" in message
    assert "crossed or truncated" in message
    assert "steady" in message


def test_a_sampling_voice_is_not_refused_for_a_steady_one_s_repeatability():
    """The same borrowing, in the direction that refuses a conformant image.

    `voices[0]` runs a fixed graph and `voices[-1]` samples. One flag taken from
    the first would hold the second to a length its backend never promised, which
    is the false positive this whole change exists to retire — arriving under the
    one name that sends a reader hunting a lock in `elvenspeak/chatterbox.py`.
    """
    fake = Fake(
        voices=[dict(NOTHING), {"voice_id": "drifty", "capabilities": []}],
        redraws={"drifty": -10},
        answers_as={("drifty", speaks.SHORT_TEXT): speaks.LONGER_TEXT},
        timestamps_status=501,
    )
    with serving(fake) as url:
        speaks.conform(url, timeout=5.0)


def test_two_draws_of_one_length_carrying_two_waveforms_are_not_a_repeat(capsys):
    """A coincidence of length is not an identical answer.

    Every draw this server makes comes back at the *same sample count* and a
    different waveform, which is the case `_repeatable` compares audio to survive:
    sample counts are quantised to whole frames, so two draws of a sampler land on
    one length by coincidence often enough to matter and never on one waveform.
    Read as a repeat, this voice would then be held to lengths its backend never
    promised — and it is served with the inversion that makes that refusal happen,
    so the verdict here is a refusal or a pass and never a log line alone.

    That is the point of asserting it this way. `_repeatable` rewritten to compare
    `first.samples == again.samples` is a one-word edit that passed every other
    test in this file, because a fake filling its bodies with zeros made equal
    length equal bytes — the defect could not be expressed, so the whole argument
    for comparing audio lived in a docstring and nothing held the code to it
    ([LAW:behavior-not-structure]).
    """
    fake = Fake(
        voices=[dict(NOTHING)],
        rewaveforms={NOTHING["voice_id"]},
        answers_as={(NOTHING["voice_id"], speaks.SHORT_TEXT): speaks.LONGER_TEXT},
        timestamps_status=501,
    )
    with serving(fake) as url:
        speaks.conform(url, timeout=5.0)

    printed = capsys.readouterr().out
    assert "plain answered one request twice with two different utterances" in printed
    assert "a length was read against another for no voice here" in printed


def test_each_request_is_bounded_by_the_work_in_front_of_it(monkeypatch):
    """A deployment that answers one caller at a time, slowly, still conforms.

    Two ways to inherit a bound rather than compute it, and both refuse an image
    for being what it is ([LAW:no-ambient-temporal-coupling]). One figure for
    every request is sized for one length: 180s was sized when these texts were a
    couple of seconds of speech, and `speaks.LONGER_TEXT` took 100s of it on a
    4-cpu workstation, leaving a 2-cpu runner less than 2x. And a caller among
    four in flight waits out
    the other three: held to the budget of a request that waits behind nothing,
    elvenspeak-chatterbox:2026.09.08.1 answered every serial synthesis of a smoke
    run and then timed out under contention on `builtin-es` saying "Yes.".

    The engine here spends 70% of the budget's rate on each character, so every
    request fits its own budget, `speaks.LONGER_TEXT` overruns `speaks.TEXT`'s,
    and four serialised callers outlast any one budget while staying inside their
    sum. Shrunk to fractions of a second, because what is under test is those
    ratios and not the real figures.
    """
    monkeypatch.setattr(speaks, "SECONDS_BEFORE_SPEECH", 0.1)
    monkeypatch.setattr(speaks, "SECONDS_PER_CHARACTER", 0.02)
    monkeypatch.setattr(speaks, "HEADROOM", 1.0)
    with serving(Fake(seconds_per_character=0.014)) as url:
        speaks.conform(url, timeout=5.0)
