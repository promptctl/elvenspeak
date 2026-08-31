"""Character timings, and the honesty of the fidelity they claim.

The properties worth pinning are not "the numbers equal these numbers" — those
would change with the voice — but the invariants a caption renderer depends on:
the timeline covers every character, never goes backwards, never leaves a hole,
and says truthfully whether its word boundaries were measured or guessed.
"""

from __future__ import annotations

import pytest

from elvenspeak.alignment import Fidelity, align
from elvenspeak.engine import Timing, TimedSpeech

RATE = 22050

# "Hello there, friend." as an engine reports it: a leading separator — the
# run-up into the utterance — then three words with a gap between each. Durations
# are invented but plausible; what matters is that they are positive and that the
# separators fall where a real engine puts them.
#
# Written as (separates_words, samples) rather than as phonemes because that is
# the whole of what crosses the seam. The stretches here correspond to espeak's
# `^`, `h ə l ˈ o ʊ`, ` `, `ð ˈ ɛ ɹ ,`, ` `, `f ɹ ˈ ɛ n d .` — but a word-level
# engine reporting one stretch per word would produce the same shape.
STRETCHES = (
    [(True, 1792)]
    + [(False, n) for n in (1280, 1024, 1024, 1024, 1280, 512)]
    + [(True, 768)]
    + [(False, 900)] * 5
    + [(True, 768)]
    + [(False, 900)] * 7
)
TEXT = "Hello there, friend."


def spoken(stretches=None, measured: bool = True) -> TimedSpeech:
    """A synthesis with these stretches, and audio of exactly their length.

    The pcm is fabricated rather than omitted so the fixture keeps
    [`TimedSpeech`]'s own invariant — the durations account for every sample —
    which is the assumption every timing here is derived under.
    """
    chosen = STRETCHES if stretches is None else stretches
    return TimedSpeech(
        pcm=b"\x00\x10" * sum(samples for _, samples in chosen),
        sample_rate=RATE,
        timings=tuple(
            Timing(samples=samples, separates_words=separates)
            for separates, samples in chosen
        ),
        measured=measured,
    )


def test_every_character_gets_a_time():
    result = align(TEXT, spoken())
    assert result.characters == list(TEXT)
    assert len(result.starts) == len(TEXT)
    assert len(result.ends) == len(TEXT)


def test_timeline_never_goes_backwards():
    result = align(TEXT, spoken())
    assert all(
        result.starts[i] <= result.starts[i + 1] for i in range(len(TEXT) - 1)
    )
    assert all(result.starts[i] <= result.ends[i] for i in range(len(TEXT)))


def test_timeline_has_no_holes():
    """Each character ends exactly where the next begins.

    A renderer that maps a playhead onto a character does so by scanning for the
    span containing it. A gap between spans is a moment with no character, which
    shows up as a flicker rather than as an error.
    """
    result = align(TEXT, spoken())
    for i in range(len(TEXT) - 1):
        assert result.ends[i] == pytest.approx(result.starts[i + 1])


def test_word_boundaries_are_measured_not_spread_evenly():
    """The point of using the engine's durations at all.

    An even spread over the whole string would put every character at the same
    width. Real measurements do not, because "Hello" and "there," take different
    amounts of time per character.
    """
    result = align(TEXT, spoken())
    widths = [e - s for s, e in zip(result.starts, result.ends, strict=True)]
    assert len(set(round(w, 6) for w in widths)) > 1
    assert result.fidelity is Fidelity.WORD_EXACT


def test_leading_separator_is_not_charged_to_a_letter():
    """The engine's run-up is silence, and belongs to no character.

    It lasts 1792 samples here, so a first character starting at 0.0 would mean
    that run-up had been billed to the letter H.
    """
    result = align(TEXT, spoken())
    assert result.starts[0] > 0.0


def test_word_count_mismatch_is_reported_not_hidden():
    """[LAW:no-silent-failure] The fallback is worse, so it announces itself.

    An engine that expands "42" into "forty two" produces more spoken words than
    the text has, and no word-level correspondence exists. The result still has
    the same shape — which is exactly why it has to carry its fidelity.
    """
    result = align("42", spoken())
    assert result.fidelity is Fidelity.INTERPOLATED
    assert len(result.characters) == 2


def test_offset_shifts_the_whole_timeline():
    """So a streamed reply lays its pieces end to end without re-deriving them."""
    plain = align(TEXT, spoken())
    shifted = align(TEXT, spoken(), offset=10.0)
    assert shifted.starts[0] == pytest.approx(plain.starts[0] + 10.0)
    assert shifted.ends[-1] == pytest.approx(plain.ends[-1] + 10.0)


def test_empty_text_is_an_empty_alignment_not_a_crash():
    result = align("", spoken([]))
    assert result.characters == []
    assert result.starts == []


def test_characters_before_the_first_word_are_placed():
    """The leading-gap branch, reachable from any text starting with whitespace.

    Untested until now, and its failure would be silent: the leading characters
    keep their initialized 0.0 span, which reads as a valid timeline that happens
    to start with several zero-width characters rather than as a fault.
    """
    result = align("  " + TEXT, spoken())
    assert len(result.characters) == len(TEXT) + 2
    assert result.starts[0] == pytest.approx(0.0)
    # The gap owns real time — the engine's run-up — and hands off to the first
    # letter rather than collapsing onto it.
    assert result.ends[1] == pytest.approx(result.starts[2])
    assert result.ends[1] > 0.0


def test_trailing_characters_are_placed():
    result = align(TEXT + "   ", spoken())
    assert len(result.characters) == len(TEXT) + 3
    assert result.ends[-1] >= result.starts[-1]
    for i in range(len(result.characters) - 1):
        assert result.ends[i] == pytest.approx(result.starts[i + 1])


def test_trailing_duration_survives_when_no_character_is_left_to_hold_it():
    """The timeline covers the audio even when the text ends on a word.

    `\\S+` swallows final punctuation, so "friend." leaves no trailing character
    for a closing separator's duration to be spread across — and `_word_spans`
    charges separator time to no word. This is the ordinary shape of a sentence,
    and it is what `speak_timed` produces whenever a chunk arrives with audio it
    cannot attribute: a trailing separator carrying real samples.

    Dropping that remainder is invisible per call — the timings look plausible,
    just short — and the streaming endpoint takes `ends[-1]` as the next
    sentence's start, so every sentence begins earlier than the audio it labels
    and the error accumulates across the response.
    """
    stretches = STRETCHES + [(True, 4410)]
    result = align(TEXT, spoken(stretches))

    assert len(result.characters) == len(TEXT)
    total = sum(samples for _, samples in stretches)
    assert result.ends[-1] == pytest.approx(total / RATE)


def test_trailing_duration_is_honoured_with_an_offset_too():
    """The streaming path's actual call shape, since it always passes an offset."""
    stretches = STRETCHES + [(True, 4410)]
    result = align(TEXT, spoken(stretches), offset=7.5)
    total = sum(samples for _, samples in stretches)
    assert result.ends[-1] == pytest.approx(7.5 + total / RATE)


def test_unmeasured_audio_forces_interpolated_even_when_words_line_up():
    """[LAW:no-silent-failure] `measured` is the engine's own report.

    Word counts matching is not enough to claim measurement when some of the
    audio was never attributed — the spans still cover the sound, but some of
    them stand for samples nothing explained.
    """
    result = align(TEXT, spoken(measured=False))
    assert result.fidelity is Fidelity.INTERPOLATED
