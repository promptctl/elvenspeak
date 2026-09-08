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
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

#: Raw signed 16-bit samples at a rate the name states, so a byte count is a
#: sample count. Every codec that is not PCM makes the length a fact about the
#: encoder instead of about the utterance.
PCM_FORMAT = "pcm_22050"
BYTES_PER_SAMPLE = 2

#: Long enough that a rate change moves the length well outside any per-request
#: jitter, and ordinary enough that no engine needs a pronunciation dictionary.
TEXT = "One two three four five."

#: Strictly longer than [`TEXT`], and longer by whole words rather than by
#: punctuation: a trailing "!" is not reliably more audio in any engine here.
LONGER_TEXT = "One two three four five, six seven eight nine ten, eleven twelve."

#: The speed to compare against 1.0. The same figure `tests/test_conformance.py`
#: uses, and for the reason stated there: fast enough that the shortening is
#: unmistakable, slow enough that no engine refuses it.
FASTER = 2.0

#: How much shorter speaking twice as fast must make it. Not 0.5 — engines round
#: to frames, pad, and trail off — but far enough below 1.0 that "the parameter
#: was quietly dropped" cannot pass. Matches `test_speed_actually_changes_the_audio`.
PACE_CHANGED = 0.75

#: Seconds to wait on one synthesis. Chatterbox runs at 8-33x real time on cpu
#: (elvenspeak/chatterbox.py:556) and these utterances are a couple of seconds
#: of speech, so the slowest engine's worst case is around a minute. Its own
#: constant rather than a reuse of `smoke.DEFAULT_TIMEOUT`: that one covers a
#: container that may be loading a model, this covers a request to a server that
#: has already loaded one, and collapsing the two is what turned a wedged fleet
#: stub into a 360s hang before `FLEET_TIMEOUT` was split out.
SPEAK_TIMEOUT = 180.0


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


def _get(base_url: str, path: str, timeout: float) -> Any:
    """`path` decoded as JSON, or a failure that names what answered."""
    request = urllib.request.Request(f"{base_url}{path}", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            return json.loads(answer.read())
    except urllib.error.HTTPError as refused:
        raise ConformanceFailure(
            f"GET {path} answered {refused.code}: {refused.read()[:400]!r}"
        ) from refused
    except (urllib.error.URLError, OSError) as unreachable:
        raise ConformanceFailure(f"GET {path} did not answer: {unreachable}") from unreachable


@dataclass(frozen=True)
class Utterance:
    """What one synthesis returned: the audio, and what the server said about it."""

    audio: bytes
    ignored: str

    @property
    def samples(self) -> int:
        """How many whole samples the audio runs to.

        Reported even when the byte count is odd, because the check that the
        count *is* whole reads this and has to be able to see the bad case.
        """
        return len(self.audio) // BYTES_PER_SAMPLE


def _speak(base_url: str, voice: str, text: str, speed: float | None = None) -> Utterance:
    """One synthesis of `text` in `voice`, as raw PCM.

    `speed` is sent only when a caller asked for one, so an engine that does not
    honour it is not handed a `voice_settings` it would then name in
    `x-elvenspeak-ignored` and make the pace check unreadable.
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
    try:
        with urllib.request.urlopen(request, timeout=SPEAK_TIMEOUT) as answer:
            return Utterance(
                audio=answer.read(),
                ignored=answer.headers.get("x-elvenspeak-ignored", ""),
            )
    except urllib.error.HTTPError as refused:
        raise ConformanceFailure(
            f"speaking {text!r} in {voice!r} answered {refused.code}: "
            f"{refused.read()[:400]!r}"
        ) from refused
    except (urllib.error.URLError, OSError) as unreachable:
        raise ConformanceFailure(
            f"speaking {text!r} in {voice!r} did not answer: {unreachable}"
        ) from unreachable


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

    for voice in voices:
        spoken = _speak(base_url, voice.id, TEXT)
        if not spoken.audio:
            raise ConformanceFailure(
                f"{voice.id!r} is on offer and answered 200 with no audio at all"
            )
        if len(spoken.audio) % BYTES_PER_SAMPLE:
            raise ConformanceFailure(
                f"{voice.id!r} returned {len(spoken.audio)} bytes of "
                f"{PCM_FORMAT}, which is not a whole number of 16-bit samples — "
                "the audio is not the format it was asked for"
            )
        print(f"speaks: {voice.id} spoke {spoken.samples} samples", flush=True)

    # Every property below is a comparison between two utterances, so it is asked
    # of one voice rather than all of them: what is under test is the engine's
    # treatment of its input, and the loop above has already established that
    # every voice on offer can be spoken in at all.
    subject = voices[0]

    shorter = _speak(base_url, subject.id, TEXT)
    longer = _speak(base_url, subject.id, LONGER_TEXT)
    if longer.samples <= shorter.samples:
        raise ConformanceFailure(
            f"{subject.id!r} made {longer.samples} samples of "
            f"{len(LONGER_TEXT)} characters and {shorter.samples} of "
            f"{len(TEXT)} — more text did not make more audio, so the engine is "
            "not speaking what it was given"
        )

    # [LAW:dataflow-not-control-flow] Both arms run the same request and differ
    # only in what they demand of the answer, because the honest failure is
    # symmetric: an engine that silently drops a speed it declared, and one that
    # applies a speed it disclaimed, are the same defect seen from two sides, and
    # a check that only looked at declared voices would see one of them.
    paced = _speak(base_url, subject.id, TEXT, speed=FASTER)
    unpaced = _speak(base_url, subject.id, TEXT)
    changed = paced.samples < unpaced.samples * PACE_CHANGED
    if subject.paces and not changed:
        raise ConformanceFailure(
            f"{subject.id!r} declares `speed` and made {paced.samples} samples at "
            f"{FASTER}x against {unpaced.samples} at 1.0 — the parameter was "
            "accepted and did nothing, which is the silent drop the declaration "
            "exists to rule out"
        )
    if not subject.paces and changed:
        raise ConformanceFailure(
            f"{subject.id!r} does not declare `speed` and yet honoured it — a "
            "voice that applies a parameter it disclaimed cannot be described "
            "accurately to any caller"
        )
    if not subject.paces and "speed" not in paced.ignored:
        raise ConformanceFailure(
            f"{subject.id!r} does not declare `speed`, was sent one, and did not "
            f"name it in x-elvenspeak-ignored (got {paced.ignored!r}) — accepted "
            "and dropped without saying so"
        )
    print(
        f"speaks: pace {'varies' if subject.paces else 'is fixed'} as declared "
        f"({unpaced.samples} -> {paced.samples} samples at {FASTER}x)",
        flush=True,
    )

    # ------------------------------------------- what a measured utterance owes

    _conform_timings(base_url, subject)


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
    try:
        with urllib.request.urlopen(request, timeout=SPEAK_TIMEOUT) as answer:
            body = json.loads(answer.read())
    except urllib.error.HTTPError as refused:
        if refused.code == 501 and not voice.measures:
            print(
                f"speaks: {voice.id} declares no timestamps and refused them 501",
                flush=True,
            )
            return
        raise ConformanceFailure(
            f"asking {voice.id!r} for timestamps answered {refused.code} while it "
            f"{'declares' if voice.measures else 'does not declare'} the "
            f"capability: {refused.read()[:400]!r}"
        ) from refused

    if not voice.measures:
        raise ConformanceFailure(
            f"{voice.id!r} does not declare `timestamps` and answered them "
            "anyway — the numbers cannot have been measured, and a caller has no "
            "way to know that"
        )

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
    print(
        f"speaks: {voice.id} measured {len(ends)} characters ending at "
        f"{ends[-1]:.2f}s",
        flush=True,
    )
