"""What the Piper engine records when the model does not explain its audio.

This is the branch that fixed a timeline quietly compressed against real audio: a
Piper chunk carrying samples but no `phoneme_alignments` used to contribute
nothing to the durations, so every timing derived from their sum came out short.
It was covered only from the far end — `test_alignment.py` exercises `align()`
once it is *handed* an unmeasured result — leaving the code that decides that
value untested.

No Piper model is needed. `speak_timed` only calls `.synthesize()` on whatever
session it holds, so a fake yielding fabricated chunks reaches the branch
directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from elvenspeak import piper
from elvenspeak.engine import Capability, Prosody, Voice

RATE = 22050
VOICE = Voice(id="test", name="test", description="test")


@dataclass
class FakeAlignment:
    phoneme: str
    num_samples: int


@dataclass
class FakeChunk:
    audio_int16_bytes: bytes
    phoneme_alignments: list | None = None


@dataclass
class FakeSession:
    """Stands in for `PiperVoice`, which is only ever asked to `.synthesize()`."""

    chunks: list = field(default_factory=list)

    def synthesize(self, text, syn_config=None, include_alignments=False):
        return iter(self.chunks)


def engine_over(*chunks: FakeChunk) -> piper.PiperEngine:
    return piper.PiperEngine(
        {
            VOICE.id: piper._Installed(
                voice=VOICE, sample_rate=RATE, model=FakeSession(list(chunks))
            )
        },
        capabilities=frozenset({Capability.SPEED, Capability.TIMESTAMPS}),
    )


def speak_timed(*chunks: FakeChunk):
    return engine_over(*chunks).speak_timed(VOICE, "hi", Prosody(speed=1.0))


def test_audio_with_no_alignments_is_recorded_and_stops_claiming_measurement():
    """The unattributed chunk is counted, and the result says it was not measured.

    Both halves matter. Counting keeps the durations summing to the audio, so no
    derived timing is short; dropping the `measured` claim is what stops a caption
    renderer trusting boundaries that stand for samples nothing explained.
    """
    result = speak_timed(
        FakeChunk(b"\x00\x10" * 100, [FakeAlignment("h", 60), FakeAlignment("ə", 40)]),
        FakeChunk(b"\x00\x10" * 50, phoneme_alignments=[]),
    )

    assert result.measured is False
    # Two bytes per sample: the invariant that keeps every derived timing honest.
    assert sum(t.samples for t in result.timings) * 2 == len(result.pcm)


def test_a_fully_aligned_synthesis_still_claims_measurement():
    """The control case, so the flag means something rather than always being False."""
    result = speak_timed(
        FakeChunk(b"\x00\x10" * 100, [FakeAlignment("h", 60), FakeAlignment("ə", 40)]),
    )

    assert result.measured is True
    assert sum(t.samples for t in result.timings) * 2 == len(result.pcm)


def test_the_unattributed_span_reads_as_a_separator():
    """Reported as a word separator, so `_word_spans` charges it to no word.

    Marked as silence between words rather than as part of one — attributing
    unexplained samples to a neighbouring word would stretch that word's measured
    span by an amount nothing measured.
    """
    result = speak_timed(
        FakeChunk(b"\x00\x10" * 100, [FakeAlignment("h", 100)]),
        FakeChunk(b"\x00\x10" * 50, phoneme_alignments=None),
    )

    assert result.timings[-1].separates_words is True
    assert result.timings[-1].samples == 50


def test_espeak_notation_is_read_here_and_nowhere_else():
    """The engine answers "does this separate words", not "what phoneme is this".

    espeak's boundary symbols are meaningless to any other engine, so the
    alphabet is interpreted on this side of the seam and only the answer crosses
    it. `alignment.py` used to hold this set itself, which made the module that
    must be engine-agnostic the owner of a Piper fact.
    """
    result = speak_timed(
        FakeChunk(
            b"\x00\x10" * 3,
            [FakeAlignment("^", 1), FakeAlignment("h", 1), FakeAlignment(" ", 1)],
        ),
    )

    assert [t.separates_words for t in result.timings] == [True, False, True]
