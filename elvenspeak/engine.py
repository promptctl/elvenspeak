"""What this server needs from a speech engine.

The rest of this package is an ElevenLabs API surface — 28 output formats, four
synthesis endpoints, the voice listing, character timings — and none of that is
specific to how the audio gets made. This module is the seam between the two: the
server depends on the types here and never on an engine, and an engine supplies
them and never learns what an endpoint is.

# Derived from the endpoints, not from any engine

Every member below exists because an endpoint stated a requirement:

    GET /v1/voices                    -> [`Engine.voices`]
    POST .../{voice}                  -> [`Engine.speak`], drained whole
    POST .../{voice}/stream           -> [`Engine.speak`], drained incrementally
    POST .../{voice}/with-timestamps  -> [`Engine.speak_timed`]
    every response's ignored header   -> [`Voice.capabilities`]
    every endpoint's 501              -> [`Voice.capabilities`]

Nothing here exists because an engine happened to offer it, which is the way this
seam fails: an interface read off one engine's methods — a model file on disk, a
phonemizer, phoneme-level alignments — is one only that engine can implement, and
the second engine then forces the interface to change. Every member must be
answerable by a remote HTTP API and by a local ONNX model alike.

Two carves that follow from that rule, both of which the first version of this
service got the other way round:

[LAW:types-are-the-program] The sample rate travels with the audio, not with the
voice. It reads like a voice property because one engine's rate is fixed per
voice, but ElevenLabs' own voice object has no sample rate, and an engine whose
rate varies per utterance would have to state a value it cannot honour. Returned
with the samples it describes, it is true for every engine.

A [`Voice`] carries no file path, no model handle, and nothing else an engine
would need to speak with. An engine turns an id into whatever it uses; the server
only ever holds what it puts in a response.

# What is not here

Resolving an unrecognised voice id is server policy, not engine business — the
alias table, the fallback, and the substitution contract live in
[`elvenspeak.voices`]. An engine answers about the voices it has and is asked to
speak in one of them; it never has to decide what an id means.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class Capability(Enum):
    """Something a caller may ask for that not every engine can do.

    A closed set, because the server has to know what to do about each one:
    refuse an endpoint, or name a parameter in the ignored header. An engine that
    does something not listed here has nothing to declare it to — adding a member
    and giving the server a use for it is one edit, not two.

    [LAW:no-mode-explosion] A set of values rather than a `can_x` method per
    capability. A method per capability puts the variability in the interface's
    *structure*: every engine grows a member for every capability any engine ever
    gains, and every place that cares grows a branch to ask about it. As data, an
    engine states what it does once and every answer the server gives — the 501,
    the ignored header, the startup log — is derived from that one statement.

    Each value is a phrase completing "this service cannot …", because telling a
    caller is the only thing the server does with an absent capability. Written
    here so it is written once: a message assembled at an endpoint would be a
    second answer to what a capability means, from the layer least able to give
    it — an endpoint has never heard of any engine. The member's `name` is the
    identifier for logs and tests; the value is the sentence.
    """

    #: `voice_settings.speed` really changes the rate: 2.0 is twice as fast. An
    #: engine with a fixed rate does not declare it, and the caller that asked is
    #: told so rather than left to infer it from the audio.
    SPEED = "vary its speaking rate"

    #: [`Engine.speak_timed`] returns durations it measured. Without it the
    #: timestamp endpoints refuse, because the alternative is an alignment
    #: derived from nothing that a caption renderer would trust.
    TIMESTAMPS = "report how long each part of an utterance took"


@dataclass(frozen=True)
class Voice:
    """One voice an engine can speak in, as the API surface has to show it.

    Display metadata and the id to name it by — that is the whole of what the
    server does with a voice. `id` is what a caller sends and what `GET
    /v1/voices` returns, so an engine's ids must be stable across restarts: a
    client that reads the listing and echoes an id back has to reach the same
    voice.

    `labels` is free-form on purpose. ElevenLabs publishes it as an open map, so
    it is where an engine puts the facts that have no field of their own — a
    quality tier, a speaker count, which engine spoke — without either side
    inventing a schema for them.

    Pairs rather than a dict, so that `frozen=True` is true instead of
    decorative. A `dict` field leaves the value mutable — and a `Voice` lives in
    a process-wide [`elvenspeak.voices.Catalog`] that hands the same object to
    every request, so one handler writing a label would change what every later
    caller is told. It also makes the auto-generated `__hash__` raise, which is
    the same lie caught from the other side. The cost is that pairs admit a
    duplicate key where a mapping could not; a shared value that cannot be
    written is worth more than that.
    """

    id: str
    name: str
    description: str
    labels: tuple[tuple[str, str], ...] = ()
    #: Everything in [`Capability`] that speaking in this voice really does.
    #:
    #: [LAW:one-source-of-truth] On the voice rather than on the engine, because a
    #: capability is a fact about *what will speak* and the voice is what names
    #: that. It was an `Engine.capabilities()` method, which was the same thing
    #: only while one process meant one engine — behind
    #: [`elvenspeak.router`] a deployment holds voices from several, and a single
    #: process-wide answer has to choose between refusing calls that would have
    #: worked and promising ones that will not. Neither is true; per-voice is.
    #:
    #: An engine-wide set is still available and is *derived*: the union over the
    #: voices on offer, computed where it is wanted. Kept as its own method it
    #: would be a second source free to disagree with the voices it summarises.
    #:
    #: Fixed for as long as the voice is offered, which is what the endpoints
    #: need: `/stream/with-timestamps` commits its 200 before it calls
    #: [`speak_timed`], so a capability discoverable only by trying could never be
    #: refused honestly. That guarantee used to be asked of the engine by fiat and
    #: is now a property of the value.
    #:
    #: Empty by default, because absence is the safe answer: a capability not
    #: declared is reported to callers as not honoured and is never asked of the
    #: engine, so a voice that undersells itself is merely pessimistic while one
    #: that oversells lies in the audio.
    capabilities: frozenset["Capability"] = frozenset()
    #: Every `model_id` the server that speaks this voice answers to — its engine's
    #: own name, plus the foreign ids that engine declares.
    #:
    #: [LAW:one-source-of-truth] On the voice for the reason `capabilities` is, and
    #: discovered to be the same reason: which engine will speak is a fact about
    #: *what will speak*, and the voice is what names that. It was
    #: `models.Directory`, built from the deployment's own engine name, which was
    #: the same thing only while one process meant one engine. Behind
    #: [`elvenspeak.router`] it was not: a router's name declares nothing, so it
    #: advertised itself as the only engine it served while routing to two, and
    #: every `model_id` its own backends honour came back reported as ignored
    #: (`piper-routing-7e2.17`, measured against the running cluster).
    #:
    #: The deployment-wide set is still available and is *derived*: the union over
    #: the voices on offer ([`elvenspeak.models.Directory.over`]). Held instead, it
    #: would be a second source free to disagree with the voices it summarises —
    #: which is exactly how it came to disagree.
    #:
    #: [LAW:types-are-the-program] Required, and the one field here with no
    #: default, because there is no such thing as a voice no engine speaks: the
    #: engine that speaks it has a name, and that name alone is already a model id
    #: it answers to. An empty set is not the cautious answer it looks like — it
    #: refuses the caller who names the very engine about to speak, since
    #: [`elvenspeak.models.Directory.reach`] finds that name among the build's
    #: engines and reads the disagreement as a request for a different one.
    #:
    #: `kw_only` so this stays beside the field it belongs with rather than moving
    #: ahead of the defaulted ones to satisfy the constructor.
    models: frozenset[str] = field(kw_only=True)


@dataclass(frozen=True)
class Prosody:
    """What a request may ask for that any engine could plausibly honour.

    Only `speed` so far. ElevenLabs' other `voice_settings` — `stability`,
    `similarity_boost`, `style`, `use_speaker_boost` — describe a generative
    model's sampling rather than the delivery of a line, so they are accepted at
    the HTTP edge and dropped there, in one documented place, rather than being
    carried across this seam for an engine to ignore.

    A field gated by a [`Capability`] the speaking voice did not declare arrives
    holding its neutral value, so an engine reads every field here without first
    checking what that voice said it could do — see [`Voice.capabilities`]. That
    is what keeps the report honest: a server that named an option as ignored and
    passed it anyway would be telling the truth only for as long as every engine
    ignored what it never claimed.
    """

    #: A multiplier on ordinary speaking rate: 2.0 is twice as fast. Expressed
    #: this way because it is the ElevenLabs parameter's meaning; an engine whose
    #: own knob runs the other way inverts it on its own side.
    speed: float = 1.0


@dataclass(frozen=True)
class Speech:
    """Audio arriving as it is made, and the rate to interpret it at.

    Signed 16-bit mono PCM, which is what [`elvenspeak.encoding`] takes and what
    every engine can produce — an engine whose native output is something else
    converts on its own side, because a seam that carried several sample formats
    would make every consumer negotiate one.

    The rate comes first and is known before any audio does, which is what the
    streaming endpoint needs: the encoder is started from it, and no engine can
    be asked to synthesize a whole utterance just to say what rate it will be at.

    `audio` is drained exactly once, and draining it is what does the work — a
    generator handed over unstarted costs nothing if the caller goes away.
    """

    sample_rate: int
    audio: Iterator[bytes]


class Silence(Exception):
    """An engine finished without producing a single sample.

    Its own type rather than a bare `ValueError` so one handler can recognise it
    without matching on a message — which would be a second, weaker copy of the
    decision about what a caller is told.

    It lives here, beside the [`Speech`] it is the absence of, rather than in
    [`elvenspeak.api`] where it is answered. Both sides of this seam raise it and
    neither imports the other: an engine reports that it produced nothing, and a
    remote engine ([`elvenspeak.remote`]) reports the same fact after reading it
    off a backend's response. Putting it in the module that answers it would mean
    the engines importing the web layer to say a thing about themselves.
    """

    #: The header by which one elvenspeak process tells another that *this* is
    #: what it means, present on that response and carrying no value worth
    #: reading — its presence is the whole message.
    #:
    #: [LAW:one-source-of-truth] Here, on the type it names, because both sides of
    #: the wire must spell it identically and both already import this module:
    #: [`elvenspeak.api`] writes it on the one response it answers a `Silence`
    #: with, and [`elvenspeak.remote`] reads it back into this type. Two literals
    #: would be two clocks, and the day they disagreed a routed silence would
    #: quietly stop being reported as one — the exact regression 7e2.12 was.
    #:
    #: A status code cannot carry this fact. 502 says "something behind me
    #: failed", which is true of a mute engine and equally true of any proxy,
    #: ingress or sidecar that ever lands between two of these processes and
    #: synthesises its own — and a `Silence` inferred from *their* 502 would tell
    #: an operator the engine went mute while the real fault was that the backend
    #: was unreachable. This header is written by this service and nothing else,
    #: so reading it is a fact about who answered rather than about what is
    #: currently deployed between them.
    WIRE_HEADER = "x-elvenspeak-silence"

    def __init__(self, voice: Voice, text: str) -> None:
        super().__init__(
            f"engine produced no audio for voice {voice.id!r} "
            f"from {len(text)} characters of text"
        )


@dataclass(frozen=True)
class Timing:
    """One stretch of an utterance the engine can account for.

    Deliberately not "a phoneme". [`elvenspeak.alignment`] asks a timed synthesis
    exactly two questions — how long did this stretch last, and does it fall
    between words — and an engine reporting phonemes, characters or whole words
    can answer both. Carrying the unit's text instead would be carrying a fact
    only a phoneme-level engine has, for a consumer that never reads it.
    """

    #: Duration, in samples at the synthesis's own rate.
    samples: int
    #: True for the silence between spoken words, and for an engine's lead-in and
    #: run-out. A run of consecutive false stretches is one word.
    separates_words: bool


@dataclass(frozen=True)
class TimedSpeech:
    """One complete utterance, with the engine's account of what took how long.

    Whole rather than streamed because the timestamp endpoints send an alignment
    with the audio, and an alignment is only complete once the last stretch has
    been measured. A caller wanting both timings and incremental delivery cuts
    the text up itself and synthesizes the pieces — which is what
    `/stream/with-timestamps` does, since only the server knows which words a
    piece contains.

    The timings' durations sum to `pcm`'s sample count: every sample is
    accounted for, whether or not the engine said what produced it.
    """

    pcm: bytes
    sample_rate: int
    timings: tuple[Timing, ...] = ()
    #: False when some audio arrived that the engine could not attribute. The
    #: timeline still spans the whole utterance — the unexplained samples are
    #: recorded as a separator — but its boundaries are no longer measurements,
    #: and nothing downstream may present them as such.
    measured: bool = True


class Engine(Protocol):
    """A source of speech, and the only thing the server knows about one.

    An implementation is constructed by whoever starts the process, already
    holding whatever it needs — models on disk, an HTTP client, credentials — so
    that a failure to get ready is one clean failure to boot rather than a
    surprise inside somebody's first request.

    [LAW:effects-at-boundaries] Both synthesis methods are synchronous and may
    block for as long as the audio takes to make, because that is what a local
    model does and what a remote client can be wrapped to do. The server calls
    them off the event loop; an engine does not have to be async to be usable
    here, and one that is wraps its own loop.
    """

    def voices(self) -> tuple[Voice, ...]:
        """Every voice this engine can speak in now, best first.

        Now, not eventually: a voice that would have to be fetched or warmed on
        first use is not offered, because the caller that named it would pay an
        unbounded and silent delay for the privilege of being first.

        The order is stable across calls, and its first element is load-bearing:
        a deployment that names no fallback voice answers unknown ids in
        whichever one an engine lists first. So an engine that has a reason to
        prefer a voice puts it at the front, and one that does not still must
        not reorder between calls — a sort chosen for tidiness silently picks
        the default voice of every deployment that left the setting alone.
        """
        ...

    def speak(self, voice: Voice, text: str, prosody: Prosody) -> Speech:
        """Synthesizes `text`, delivering samples as they are produced.

        `voice` is one this engine returned from [`voices`]; the server resolves
        ids and hands back the value, so an engine never has to answer for an id
        it does not know.
        """
        ...

    def speak_timed(self, voice: Voice, text: str, prosody: Prosody) -> TimedSpeech:
        """Synthesizes `text` in one piece, measuring it as it goes."""
        ...
