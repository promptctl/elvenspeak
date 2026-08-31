"""What `synthesize_timed` records when the model does not explain its audio.

This is the branch that fixed a timeline quietly compressed against real audio:
a Piper chunk carrying samples but no `phoneme_alignments` used to contribute
nothing to `durations`, so every timing derived from their sum came out short.
It was covered only from the far end — `test_alignment.py` exercises `align()`
once it is *handed* `measured=False` — leaving the code that decides that value
untested.

No Piper model is needed. `synthesize_timed` annotates `model` as `"PiperVoice"`
with no runtime constraint and only calls `.synthesize()`, so a fake yielding
fabricated chunks reaches the branch directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from elvenspeak.speech import Prosody, synthesize_timed

RATE = 22050


@dataclass
class FakeAlignment:
    phoneme: str
    num_samples: int


@dataclass
class FakeChunk:
    audio_int16_bytes: bytes
    phoneme_alignments: list | None = None


@dataclass
class FakeVoice:
    """Stands in for `PiperVoice`, which is only ever asked to `.synthesize()`."""

    chunks: list = field(default_factory=list)

    def synthesize(self, text, syn_config=None, include_alignments=False):
        return iter(self.chunks)


def test_audio_with_no_alignments_is_recorded_and_stops_claiming_measurement():
    """The unattributed chunk is counted, and the result says it was not measured.

    Both halves matter. Counting keeps `durations` summing to the audio, so no
    derived timing is short; dropping the `measured` claim is what stops a caption
    renderer trusting boundaries that stand for samples nothing explained.
    """
    model = FakeVoice(chunks=[
        FakeChunk(b"\x00\x10" * 100, [FakeAlignment("h", 60), FakeAlignment("ə", 40)]),
        FakeChunk(b"\x00\x10" * 50, phoneme_alignments=[]),
    ])
    result = synthesize_timed(model, "hi", Prosody(speed=1.0), RATE)

    assert result.measured is False
    # Two bytes per sample: the invariant that keeps every derived timing honest.
    assert sum(result.durations) * 2 == len(result.pcm)


def test_a_fully_aligned_synthesis_still_claims_measurement():
    """The control case, so the flag means something rather than always being False."""
    model = FakeVoice(chunks=[
        FakeChunk(b"\x00\x10" * 100, [FakeAlignment("h", 60), FakeAlignment("ə", 40)]),
    ])
    result = synthesize_timed(model, "hi", Prosody(speed=1.0), RATE)

    assert result.measured is True
    assert sum(result.durations) * 2 == len(result.pcm)


def test_the_unattributed_span_reads_as_a_separator():
    """Recorded as a boundary phoneme, so `_word_spans` charges it to no word.

    Marked as silence between words rather than as part of one — attributing
    unexplained samples to a neighbouring word would stretch that word's measured
    span by an amount nothing measured.
    """
    model = FakeVoice(chunks=[
        FakeChunk(b"\x00\x10" * 100, [FakeAlignment("h", 100)]),
        FakeChunk(b"\x00\x10" * 50, phoneme_alignments=None),
    ])
    result = synthesize_timed(model, "hi", Prosody(speed=1.0), RATE)

    assert result.phonemes[-1].isspace()
    assert result.durations[-1] == 50
