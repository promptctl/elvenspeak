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
from dataclasses import dataclass
from typing import Protocol


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


@dataclass(frozen=True)
class Prosody:
    """What a request may ask for that any engine could plausibly honour.

    Only `speed` so far. ElevenLabs' other `voice_settings` — `stability`,
    `similarity_boost`, `style`, `use_speaker_boost` — describe a generative
    model's sampling rather than the delivery of a line, so they are accepted at
    the HTTP edge and dropped there, in one documented place, rather than being
    carried across this seam for an engine to ignore.
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
        """Every voice this engine can speak in now, in a stable order.

        Now, not eventually: a voice that would have to be fetched or warmed on
        first use is not offered, because the caller that named it would pay an
        unbounded and silent delay for the privilege of being first.
        """
        ...

    def can_time(self) -> bool:
        """Whether [`speak_timed`] will really measure what it returns.

        Asked because the answer is a fact about the engine and nowhere else. It
        was previously read off a server setting that happened to be what the
        engine had been built from — two representations of one fact, agreeing
        only because every caller passed the same value to both
        ([LAW:one-source-of-truth]). The timestamp endpoints refuse with a 501
        when this is false, which they can only do honestly if they ask the thing
        that knows.
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
