"""Character timings, and the honesty of the fidelity they claim.

The properties worth pinning are not "the numbers equal these numbers" — those
would change with the voice — but the invariants a caption renderer depends on:
the timeline covers every character, never goes backwards, never leaves a hole,
and says truthfully whether its word boundaries were measured or guessed.
"""

from __future__ import annotations

import pytest

from elvenspeak.alignment import Fidelity, align

RATE = 22050

# "Hello there, friend." as espeak phonemizes it: a leading boundary phoneme,
# then three words separated by spaces. Durations are invented but plausible;
# what matters is that they are positive and that the separators are where a
# real phonemizer puts them.
PHONEMES = [
    "^", "h", "ə", "l", "ˈ", "o", "ʊ",
    " ", "ð", "ˈ", "ɛ", "ɹ", ",",
    " ", "f", "ɹ", "ˈ", "ɛ", "n", "d", ".",
]
DURATIONS = [
    1792, 1280, 1024, 1024, 1024, 1280, 512,
    768, 900, 900, 900, 900, 900,
    768, 900, 900, 900, 900, 900, 900, 900,
]
TEXT = "Hello there, friend."


def test_every_character_gets_a_time():
    result = align(TEXT, PHONEMES, DURATIONS, RATE)
    assert result.characters == list(TEXT)
    assert len(result.starts) == len(TEXT)
    assert len(result.ends) == len(TEXT)


def test_timeline_never_goes_backwards():
    result = align(TEXT, PHONEMES, DURATIONS, RATE)
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
    result = align(TEXT, PHONEMES, DURATIONS, RATE)
    for i in range(len(TEXT) - 1):
        assert result.ends[i] == pytest.approx(result.starts[i + 1])


def test_word_boundaries_are_measured_not_spread_evenly():
    """The point of using phoneme durations at all.

    An even spread over the whole string would put every character at the same
    width. Real measurements do not, because "Hello" and "there," take different
    amounts of time per character.
    """
    result = align(TEXT, PHONEMES, DURATIONS, RATE)
    widths = [e - s for s, e in zip(result.starts, result.ends, strict=True)]
    assert len(set(round(w, 6) for w in widths)) > 1
    assert result.fidelity is Fidelity.WORD_EXACT


def test_leading_boundary_phoneme_is_not_charged_to_a_letter():
    """The model's run-up is silence, and belongs to no character.

    `^` lasts 1792 samples here, so a first character starting at 0.0 would mean
    that run-up had been billed to the letter H.
    """
    result = align(TEXT, PHONEMES, DURATIONS, RATE)
    assert result.starts[0] > 0.0


def test_word_count_mismatch_is_reported_not_hidden():
    """[LAW:no-silent-failure] The fallback is worse, so it announces itself.

    A phonemizer that expands "42" into "forty two" produces more spoken words
    than the text has, and no word-level correspondence exists. The result still
    has the same shape — which is exactly why it has to carry its fidelity.
    """
    result = align("42", PHONEMES, DURATIONS, RATE)
    assert result.fidelity is Fidelity.INTERPOLATED
    assert len(result.characters) == 2


def test_offset_shifts_the_whole_timeline():
    """So a streamed reply lays its pieces end to end without re-deriving them."""
    plain = align(TEXT, PHONEMES, DURATIONS, RATE)
    shifted = align(TEXT, PHONEMES, DURATIONS, RATE, offset=10.0)
    assert shifted.starts[0] == pytest.approx(plain.starts[0] + 10.0)
    assert shifted.ends[-1] == pytest.approx(plain.ends[-1] + 10.0)


def test_empty_text_is_an_empty_alignment_not_a_crash():
    result = align("", [], [], RATE)
    assert result.characters == []
    assert result.starts == []
