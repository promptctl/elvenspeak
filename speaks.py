#!/usr/bin/env python3
"""Every property `elvenspeak.engine` promises, asked of a server over HTTP.

`tests/test_conformance.py` asks these same questions of an engine object in
process, and for the three engines that own a model it can only do so by
fetching ~3.4 GB onto the runner and synthesizing on cpu — 22m54s on gitea run
40, gating all four publish legs. piper-pipeline-4mx took the models out of that
suite. This file is where the questions they answered went, and the move is an
upgrade rather than a salvage: in process the subject is the *program*, and here
it is the *artifact* — the image that is about to be pushed, with the checkpoints
it really baked, run as a container. The gap between those two is the one that
has shipped a working fix twice and served nothing.

It costs no download. The image already carries its assets — `acquire` runs at
build time (Dockerfile:227) — so the bytes that made the pytest suite expensive
are paid by the bake either way, and what is added here is inference alone.

# One code path for every engine

Nothing below asks which engine it is talking to, and nothing may start. Voices,
and what each voice can do, are read from `/v1/voices`, which publishes
`capabilities` per voice — so the difference between an engine that measures
utterances and one that cannot is a *value this file reads*, never a branch it
selects ([LAW:dataflow-not-control-flow]). That is also the strongest available
statement of the seam: an engine no one anticipated is conformant here if it
answers, and needs no entry in any table to be asked.

# Standard library only

Same constraint as `smoke.py`, for the same reason: this runs on the CI runner
beside it, before — and independently of — any project install. It never imports
`elvenspeak`, so it cannot accidentally assert the implementation against itself.

# What it deliberately does not check

"The audio is mono at exactly the rate it declares" is not answerable from bytes,
exactly as `tests/test_conformance.py` says of the same property. What is
checkable is every consequence that shows up as arithmetic, and `pcm_*` is
requested precisely so that arithmetic is available: raw signed 16-bit
little-endian samples at a rate named in the format itself, with no container to
subtract and no codec to make the byte count mean something else.

Which speaker the audio is in belongs to the same list, and is stated because the
concurrency check below is otherwise easy to read as proving it. An utterance
returned in the wrong person's voice is fluent, the right length, and identical
under every arithmetic here to a correct one. Separating those needs a speaker
embedding, which is a model — the thing this file exists to do without.
[`_conform_concurrently`] says what it does establish instead.
"""

from __future__ import annotations

import base64
import http.client
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from email.message import Message
from typing import Any

#: Raw signed 16-bit samples at a rate the name states, so a byte count is a
#: sample count. Every codec that is not PCM makes the length a fact about the
#: encoder instead of about the utterance.
PCM_FORMAT = "pcm_22050"
BYTES_PER_SAMPLE = 2

#: The rate every duration below is read at, taken out of the format that asks
#: for it rather than written beside it. A second literal would be a second
#: clock, and the one free to drift is the one no request ever carries.
SAMPLE_RATE = int(PCM_FORMAT.removeprefix("pcm_"))

#: Long enough that halving the pace moves the length well outside the jitter of
#: every engine that declares `speed`, and ordinary enough that no engine needs a
#: pronunciation dictionary. Not far enough from [`SHORT_TEXT`] to be told apart
#: from it by length alone — see [`APART`].
TEXT = "One two three four five."

#: A whole utterance in four characters, asked of every voice beside [`TEXT`].
#: Kokoro has a measured zero-sample defect on short text — `ef_dora` returned
#: nothing for 15 of 16 one- and two-word Spanish lines, which is why no Kokoro
#: Spanish voice is baked (elvenspeak/chatterbox.py). Chatterbox produced healthy
#: audio for every short line measured, and asserting that is not a courtesy to
#: the engine that passes: whatever eventually renders mixed language feeds these
#: engines spans this short, and one that goes silent on them is useless for it
#: however well it reads a paragraph. The server refuses a silent synthesis with
#: `Silence` rather than a 200 of nothing, so the defect arrives here as a named
#: status on a named text.
SHORT_TEXT = "Yes."

#: Longer by whole words rather than by punctuation: a trailing "!" is not
#: reliably more audio in any engine here.
LONGER_TEXT = "One two three four five, six seven eight nine ten, eleven twelve."

#: The two texts every comparison of two utterances' lengths in this file is made
#: between, far enough apart that no draw of any engine here puts them in the
#: wrong order. Chatterbox samples, so a bare `<` between two of its answers is a
#: coin weighted by how far apart the texts are. Measured on
#: elvenspeak-chatterbox:2026.09.08.1: six syntheses of [`TEXT`] that reached the
#: engine identical ran 39690 to 59094 samples, and "Yes." has run to 44100 —
#: past [`TEXT`]'s shortest, so those two overlap. [`LONGER_TEXT`] ran 113778,
#: which a third's spread still leaves 1.7x the longest "Yes.".
APART = (SHORT_TEXT, LONGER_TEXT)

#: The speed to compare against 1.0. The same figure `tests/test_conformance.py`
#: uses, and for the reason stated there: fast enough that the shortening is
#: unmistakable, slow enough that no engine refuses it.
FASTER = 2.0

#: How much shorter speaking twice as fast must make it. Not 0.5 — engines round
#: to frames, pad, and trail off — but far enough below 1.0 that "the parameter
#: was quietly dropped" cannot pass. Matches `test_speed_actually_changes_the_audio`.
PACE_CHANGED = 0.75

#: How far the end of the timeline may sit from the end of the audio it
#: describes. The only legitimate error is the ffmpeg pass that resamples the
#: engine's native rate into [`PCM_FORMAT`], which preserves a duration to within
#: a sample or two — tens of microseconds — so this is three orders of magnitude
#: looser than anything correct can need. It stays far tighter than the defect it
#: rules out: a timeline stopping short of its audio makes the streaming endpoint
#: lay the next sentence over the difference, sliding further with every sentence
#: (elvenspeak/alignment.py).
ACCOUNTED_SLACK = 0.05

#: What the slowest engine here spends on one synthesis before any of it is
#: speech, and what each character of text adds, in seconds. A line through the
#: slowest answers elvenspeak-chatterbox:2026.09.08.1 gave on 4 cpus under
#: Rosetta — "Yes." in 48s and [`LONGER_TEXT`] in 100s — which puts [`TEXT`] at
#: 65.4s against the 62s it took.
SECONDS_BEFORE_SPEECH = 45.0
SECONDS_PER_CHARACTER = 0.85

#: How far past that line a synthesis may run before it is a wedge rather than a
#: slow machine. [`TEXT`]'s six answers took 42.5s to 59.5s, 1.4x apart, and the
#: CI runner has 2 cpus where the line was drawn on 4.
HEADROOM = 3.0


def budget(text: str) -> float:
    """Seconds to wait on one synthesis of `text` that waits behind nothing.

    [LAW:no-ambient-temporal-coupling] Computed from the text a request carries,
    because that is what the wait is made of. One figure for every request was
    sized when these texts were a couple of seconds of speech, and went on
    holding [`LONGER_TEXT`] — 100s on a 4-cpu workstation — to 180s on a 2-cpu
    runner. Its own figure rather than `smoke.DEFAULT_TIMEOUT`: that one covers a
    container that may be loading a model, this a server that has loaded one,
    and collapsing the two is what turned a wedged fleet stub into a 360s hang
    before `FLEET_TIMEOUT` was split out.
    """
    return HEADROOM * (SECONDS_BEFORE_SPEECH + SECONDS_PER_CHARACTER * len(text))


class ConformanceFailure(Exception):
    """A property of the seam did not hold against the running artifact."""


@dataclass(frozen=True)
class SpokenVoice:
    """A voice as the server publishes it, and only the parts asked about here.

    [LAW:parse-dont-validate] The listing is turned into these once, at the one
    place it is read, so nothing below indexes into raw JSON or re-decides what a
    missing `capabilities` key means. A voice that arrives without an id or
    without its capability list is a malformed answer, and it is refused there
    rather than surfacing later as a `KeyError` inside an assertion about
    something else.
    """

    id: str
    capabilities: frozenset[str]

    @property
    def measures(self) -> bool:
        """Whether this voice claims to time the utterance it produces."""
        return "timestamps" in self.capabilities

    @property
    def paces(self) -> bool:
        """Whether this voice claims to honour a speed."""
        return "speed" in self.capabilities


@dataclass(frozen=True)
class Reply:
    """One exchange that ran to the end of its body, whatever its status said.

    [LAW:parse-dont-validate] Built only by [`_exchange`], so holding one is proof
    the transport finished, and nothing below catches a socket error again.
    """

    status: int
    headers: Message
    body: bytes


def _exchange(request: urllib.request.Request, timeout: float, asking: str) -> Reply:
    """`request` answered to the end, or [`ConformanceFailure`] naming `asking`.

    [LAW:single-enforcer] The one place this file meets the transport, so every
    request is held to one list of what counts as unanswered. The member easy to
    leave out is `http.client.IncompleteRead` — not an `OSError` — which is what a
    stream whose 200 went out before its encoder failed arrives as.

    A refusal is an answer rather than a failed exchange: its status and body come
    back for the caller to judge, because a 501 is what a voice that measures
    nothing owes.
    """
    try:
        try:
            answer = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as refused:
            answer = refused
        with answer:
            return Reply(answer.status, answer.headers, answer.read())
    except (OSError, http.client.HTTPException) as unfinished:
        raise ConformanceFailure(f"{asking} got no complete answer: {unfinished!r}") from unfinished


def _refused(asking: str, reply: Reply) -> ConformanceFailure:
    """The failure for a request whose answer was not the 200 it needed."""
    return ConformanceFailure(f"{asking} answered {reply.status}: {reply.body[:400]!r}")


def _decoded(asking: str, reply: Reply) -> Any:
    """`reply`'s body as JSON, or a failure naming the request that answered it."""
    try:
        return json.loads(reply.body)
    except ValueError as unreadable:
        raise ConformanceFailure(
            f"{asking} answered a body that is not JSON: {reply.body[:400]!r}"
        ) from unreadable


def _get(base_url: str, path: str, timeout: float) -> Any:
    """`path` decoded as JSON, or a failure that names what answered."""
    asking = f"GET {path}"
    reply = _exchange(urllib.request.Request(f"{base_url}{path}", method="GET"), timeout, asking)
    if reply.status != 200:
        raise _refused(asking, reply)
    return _decoded(asking, reply)


@dataclass(frozen=True)
class Utterance:
    """What one synthesis returned: the audio, and what the server said about it."""

    audio: bytes
    #: Empty for an answer that carried no such header, which is also what the
    #: timestamp endpoint's body amounts to — it returns its audio inside JSON
    #: and names nothing ignored, so [`_conform_timings`] builds one of these
    #: from the decoded bytes and is held to the same bar as every other
    #: synthesis rather than to a second, laxer reading of the same format.
    ignored: str = ""

    @property
    def samples(self) -> int:
        """How many whole samples the audio runs to.

        Reported even when the byte count is odd, because the check that the
        count *is* whole reads this and has to be able to see the bad case.
        """
        return len(self.audio) // BYTES_PER_SAMPLE

    @property
    def seconds(self) -> float:
        """How long the audio runs, at the rate [`PCM_FORMAT`] asked for."""
        return self.samples / SAMPLE_RATE


def _speak(
    base_url: str,
    voice: str,
    text: str,
    speed: float | None = None,
    *,
    timeout: float,
) -> Utterance:
    """One synthesis of `text` in `voice`, as raw PCM.

    `speed` is sent only when a caller asked for one, so an engine that does not
    honour it is not handed a `voice_settings` it would then name in
    `x-elvenspeak-ignored` and make the pace check unreadable.

    [LAW:no-ambient-temporal-coupling] `timeout` is required and deliberately
    has no default. How long a request may take is not only a fact about its
    text; it is also how many callers the caller has in flight, and only the
    caller knows that. A default of [`budget`] would read as correct at a
    contended call site while handing it the bound of a request that waits
    behind nothing — the failure [`_conform_concurrently`] records against a
    real image. The caller computes its bound; nothing here inherits one.
    """
    body: dict[str, Any] = {"text": text}
    if speed is not None:
        body["voice_settings"] = {"speed": speed}
    request = urllib.request.Request(
        f"{base_url}/v1/text-to-speech/{voice}/stream?output_format={PCM_FORMAT}",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    asking = f"speaking {text!r} in {voice!r}"
    reply = _exchange(request, timeout, asking)
    if reply.status != 200:
        raise _refused(asking, reply)
    return Utterance(audio=reply.body, ignored=reply.headers.get("x-elvenspeak-ignored", ""))


def _audible(voice_id: str, text: str, spoken: Utterance) -> None:
    """Refuse an answer that is not a whole utterance of the format asked for.

    [LAW:single-enforcer] Every synthesis in this file arrives here, so what a
    well-formed answer looks like is decided once and the concurrent callers are
    held to exactly the bar the serial ones are. A second, local reading of the
    same two rules is how contention becomes the one condition under which a
    broken answer passes.

    The text is named rather than only the voice, because a voice is asked for
    more than one utterance now and the short one is where an engine is known to
    go silent. "`ef_dora` answered with nothing" sends a reader to the voice;
    "`ef_dora` was asked to say 'Yes.'" sends them to the defect.
    """
    if not spoken.audio:
        raise ConformanceFailure(
            f"{voice_id!r} was asked to say {text!r} and answered 200 with no "
            "audio at all"
        )
    if len(spoken.audio) % BYTES_PER_SAMPLE:
        raise ConformanceFailure(
            f"{voice_id!r} said {text!r} in {len(spoken.audio)} bytes of "
            f"{PCM_FORMAT}, which is not a whole number of 16-bit samples — the "
            "audio is not the format it was asked for"
        )


def _voices(base_url: str, timeout: float) -> tuple[SpokenVoice, ...]:
    """Every voice the server offers, parsed."""
    listed = _get(base_url, "/v1/voices", timeout)
    try:
        entries = listed["voices"]
    except (KeyError, TypeError) as malformed:
        raise ConformanceFailure(
            f"/v1/voices answered without a `voices` list: {listed!r:.400}"
        ) from malformed
    parsed = []
    for entry in entries:
        try:
            parsed.append(
                SpokenVoice(id=entry["voice_id"], capabilities=frozenset(entry["capabilities"]))
            )
        except (KeyError, TypeError) as malformed:
            raise ConformanceFailure(
                f"/v1/voices published a voice missing `voice_id` or "
                f"`capabilities`: {entry!r:.400}"
            ) from malformed
    if not parsed:
        raise ConformanceFailure(
            "/v1/voices published an empty list — a server with nothing to say "
            "should not have answered /health 200"
        )
    return tuple(parsed)


def conform(base_url: str, timeout: float) -> None:
    """Return only if every property held; raise [`ConformanceFailure`] otherwise.

    Returns nothing, for the reason `smoke.smoke` returns nothing: the failure
    arm is the exception and the success arm is having reached the end.

    The order is deliberate and is cheapest-first — the listing before any
    synthesis, and one voice spoken before every voice is. An engine that is
    broken in a way that shows up in its catalogue says so in milliseconds
    instead of after the slowest engine here has spent a minute proving it.
    """
    voices = _voices(base_url, timeout)
    print(f"speaks: {len(voices)} voices offered: {', '.join(v.id for v in voices)}", flush=True)

    # ------------------------------------------------ what the catalogue promises

    again = _voices(base_url, timeout)
    if voices != again:
        raise ConformanceFailure(
            "the voices on offer changed between two calls with nothing in "
            f"between:\n  first:  {voices}\n  second: {again}"
        )

    ids = [voice.id for voice in voices]
    if len(set(ids)) != len(ids):
        duplicated = sorted({name for name in ids if ids.count(name) > 1})
        raise ConformanceFailure(
            f"the same voice id is offered more than once: {duplicated} — a "
            "caller naming one of these cannot be answered predictably"
        )

    # ------------------------------------------------------- what speaking proves

    # [LAW:dataflow-not-control-flow] The short utterance is another value in the
    # texts this loop speaks, never a second pass with a check of its own. Both
    # lengths reach the same request, the same refusals and the same log line, so
    # an engine that goes silent on four characters fails the check every voice
    # already passes rather than one bolted on beside it.
    for voice in voices:
        for text in (TEXT, SHORT_TEXT):
            spoken = _speak(base_url, voice.id, text, timeout=budget(text))
            _audible(voice.id, text, spoken)
            print(
                f"speaks: {voice.id} said {text!r} in {spoken.samples} samples",
                flush=True,
            )

    # Every property below is a comparison between two utterances, so it is asked
    # of one voice rather than all of them: what is under test is the engine's
    # treatment of its input, and the loop above has already established that
    # every voice on offer can be spoken in at all.
    subject = voices[0]

    less_text, more_text = (
        _speak(base_url, subject.id, text, timeout=budget(text)) for text in APART
    )
    if more_text.samples <= less_text.samples:
        raise ConformanceFailure(
            f"{subject.id!r} made {more_text.samples} samples of "
            f"{len(APART[1])} characters and {less_text.samples} of "
            f"{len(APART[0])} — more text did not make more audio, so the engine "
            "is not speaking what it was given"
        )

    # [LAW:dataflow-not-control-flow] Both arms run the same two requests and
    # differ only in what the declaration says the answer owes. A declared speed
    # owes shorter audio. A disclaimed one owes being named ignored — and not a
    # length, because the server withholds a speed from a voice that did not
    # declare it (`api.prosody`), so both requests reach the engine identical and
    # any difference between them is the engine's own. Chatterbox samples, so two
    # identical requests need not come back the same length, and reading one short
    # draw as a speed honoured failed a correct image. [LAW:single-enforcer] That
    # a disclaimed speed never reaches the engine is held where it is decided, by
    # test_capabilities.py's test_a_speed_the_engine_cannot_vary_never_reaches_it.
    paced = _speak(base_url, subject.id, TEXT, speed=FASTER, timeout=budget(TEXT))
    unpaced = _speak(base_url, subject.id, TEXT, timeout=budget(TEXT))
    if subject.paces and paced.samples >= unpaced.samples * PACE_CHANGED:
        raise ConformanceFailure(
            f"{subject.id!r} declares `speed` and made {paced.samples} samples at "
            f"{FASTER}x against {unpaced.samples} at 1.0 — the parameter was "
            "accepted and did nothing, which is the silent drop the declaration "
            "exists to rule out"
        )
    if not subject.paces and "speed" not in paced.ignored:
        raise ConformanceFailure(
            f"{subject.id!r} does not declare `speed`, was sent one, and did not "
            f"name it in x-elvenspeak-ignored (got {paced.ignored!r}) — accepted "
            "and dropped without saying so"
        )
    print(
        f"speaks: speed {'varies the pace' if subject.paces else 'is named ignored'} "
        f"as declared ({unpaced.samples} -> {paced.samples} samples at {FASTER}x)",
        flush=True,
    )

    # ------------------------------------------- what a measured utterance owes

    _conform_timings(base_url, subject)

    # --------------------------------------------- what holds under contention

    _conform_concurrently(base_url, voices)


def _conform_timings(base_url: str, voice: SpokenVoice) -> None:
    """The timing endpoint answers exactly as `voice`'s declaration says it will.

    Both arms are checked for the reason the pace check checks both: the 501 is
    a promise as much as the 200 is, and an engine that answers timings it never
    measured is the failure `Capability.TIMESTAMPS` was added to prevent.
    """
    request = urllib.request.Request(
        f"{base_url}/v1/text-to-speech/{voice.id}/with-timestamps"
        f"?output_format={PCM_FORMAT}",
        data=json.dumps({"text": TEXT}).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    asking = f"asking {voice.id!r} for timestamps"
    reply = _exchange(request, budget(TEXT), asking)
    if reply.status == 501 and not voice.measures:
        print(
            f"speaks: {voice.id} declares no timestamps and refused them 501",
            flush=True,
        )
        return
    if reply.status != 200:
        raise ConformanceFailure(
            f"{asking} answered {reply.status} while it "
            f"{'declares' if voice.measures else 'does not declare'} the "
            f"capability: {reply.body[:400]!r}"
        )

    if not voice.measures:
        raise ConformanceFailure(
            f"{voice.id!r} does not declare `timestamps` and answered them "
            "anyway — the numbers cannot have been measured, and a caller has no "
            "way to know that"
        )

    body = _decoded(asking, reply)
    try:
        ends = body["alignment"]["character_end_times_seconds"]
    except (KeyError, TypeError) as malformed:
        raise ConformanceFailure(
            f"{voice.id!r} answered timestamps without a character alignment: "
            f"{body!r:.400}"
        ) from malformed
    if not ends:
        raise ConformanceFailure(
            f"{voice.id!r} answered timestamps with an empty alignment for "
            f"{len(TEXT)} characters of text"
        )
    if list(ends) != sorted(ends):
        raise ConformanceFailure(
            f"{voice.id!r} returned character end times that go backwards — a "
            "timeline that is not monotonic cannot describe an utterance"
        )

    # The audio the timeline is a timeline OF, which is in the body beside it and
    # was going unread. Without it every check above is internal to the
    # alignment: an engine that measured a different utterance, or measured half
    # of this one, satisfies "non-empty and ascending" completely.
    try:
        measured = Utterance(base64.b64decode(body["audio_base64"], validate=True))
    except (KeyError, TypeError, ValueError) as malformed:
        raise ConformanceFailure(
            f"{voice.id!r} answered timestamps without decodable audio beside "
            f"them: {malformed}"
        ) from malformed
    _audible(voice.id, TEXT, measured)

    # `engine.TimedSpeech` promises its timings sum to the audio's sample count —
    # every sample accounted for, whether or not the engine said what produced it
    # — and `elvenspeak.alignment` ends the last character exactly there. So this
    # is that seam invariant asked of the artifact, and it is the one check here
    # that reads the audio and the timeline against each other rather than each
    # against itself.
    if abs(ends[-1] - measured.seconds) > ACCOUNTED_SLACK:
        raise ConformanceFailure(
            f"{voice.id!r} measured {len(ends)} characters ending at "
            f"{ends[-1]:.3f}s against {measured.samples} samples of "
            f"{PCM_FORMAT}, which run {measured.seconds:.3f}s — the timeline does "
            "not account for the utterance it describes, so a caller laying the "
            "next one after it slides by the difference every time"
        )
    print(
        f"speaks: {voice.id} measured {len(ends)} characters ending at "
        f"{ends[-1]:.2f}s across {measured.seconds:.2f}s of audio",
        flush=True,
    )


def _conform_concurrently(base_url: str, voices: tuple[SpokenVoice, ...]) -> None:
    """Callers inside the engine at once are each answered in full.

    The engines behind this seam are not reentrant, and one of them says so in
    its own code: Chatterbox selects a speaker by WRITING to the model object and
    then reading it back (`elvenspeak/chatterbox.py`, `_synthesized`), so two
    unserialised callers can have the second one's write land between the first
    one's write and its read. The server calls engines off the event loop, so two
    requests naming two voices really are inside `speak` at the same time. This
    is the shape that puts them there — against the real model, in the real
    process, which a stand-in agrees with by construction.

    WHAT THIS CANNOT SEE, stated because what is asserted below is otherwise easy
    to read as more than it is: which speaker answered. A swapped identity comes
    back fluent, the right length, in the wrong person, and no arithmetic over
    PCM separates it from a correct answer — the module header lists it beside
    mono and rate for that reason. `tests/test_chatterbox.py` proves the lock
    that prevents it at the level of that engine's own code, where the
    conditionals are objects that can be told apart.

    WHAT IT DOES ESTABLISH is everything else contention breaks, none of which
    any other check in this file would survive: a caller refused, a deadlock, a
    truncated body, two responses interleaved on one socket, an answer that is
    somebody else's length. Whether the two overlapped at all is the deployment's
    `speaking_at_once` bound to decide and is not observable from out here; a
    server that serialises them keeps every promise below, which is correct —
    `tests/test_concurrency.py` owns the bound itself.

    Two voices and not one, because a single-voice pair never writes two
    different speakers. The first and the last of the offer rather than a slice,
    so a deployment offering exactly one voice is asked the same question twice
    instead of being quietly skipped ([LAW:dataflow-not-control-flow]).
    """
    subjects = (voices[0], voices[-1])
    callers_at_once = len(subjects) * len(APART)
    # [LAW:no-ambient-temporal-coupling] Each caller's bound is the work of every
    # caller in flight, never the budget of a request that waits behind nothing.
    # The four below share one deployment's cpus, and its `speaking_at_once` may
    # serialise them outright, so the last one to be answered waits out the other
    # three's work either way — the same total either way, which is what makes
    # the widening exactly this sum and not a guess.
    #
    # Measured, not reasoned: elvenspeak-chatterbox:2026.09.08.1 answered every
    # serial synthesis of that run and then failed here, on `builtin-es` saying
    # 'Yes.' — the shortest utterance this file asks for anywhere, timing out at
    # 180s because three other callers were in front of it. A conformant image,
    # refused a publish by the clock it was held to rather than by anything it
    # did.
    contended = sum(budget(text) for _ in subjects for text in APART)

    with ThreadPoolExecutor(max_workers=callers_at_once) as callers:
        # Every caller submitted before any result is taken, which is the whole
        # mechanism: resolving each future as it was created would run these one
        # after another and prove nothing this file does not already know.
        # `.result()` re-raises, so a caller refused under contention arrives as
        # the `ConformanceFailure` `_speak` already built ([LAW:no-silent-failure]).
        #
        # Kept as a list of pairs and never keyed by voice: a deployment offering
        # one voice makes `subjects` that voice twice, and a dictionary would
        # have silently collapsed four answers into two — with the two it
        # discarded being the ones nothing then checked.
        started = [
            (
                voice,
                [
                    callers.submit(
                        _speak, base_url, voice.id, text, timeout=contended
                    )
                    for text in APART
                ],
            )
            for voice in subjects
        ]
        answered = [
            (voice, tuple(caller.result() for caller in pair)) for voice, pair in started
        ]

    for voice, pair in answered:
        for text, spoken in zip(APART, pair, strict=True):
            _audible(voice.id, text, spoken)

    # Compared inside one voice and never across two. Across, this would be
    # arithmetic on two speaking rates, and that two voices of one engine speak
    # at comparable rates is not a property any engine here promises — a check
    # resting on it would go red on a legitimately slow voice and read as a
    # concurrency defect.
    for voice, (shorter, longer) in answered:
        if longer.samples <= shorter.samples:
            raise ConformanceFailure(
                f"under {callers_at_once} callers at once, {voice.id!r} answered "
                f"{len(APART[1])} characters with {longer.samples} samples and "
                f"{len(APART[0])} with {shorter.samples} — each answer is a "
                "plausible utterance and they are not the ones that were asked "
                "for, which is what a crossed or truncated response looks like "
                "from here"
            )
    print(
        f"speaks: {callers_at_once} callers at once across "
        f"{len({voice.id for voice, _ in answered})} voices were each answered in full",
        flush=True,
    )
