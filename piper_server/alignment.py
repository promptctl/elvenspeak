"""Turning Piper's phoneme durations into the character timings clients expect.

ElevenLabs' `/with-timestamps` endpoints return, for every character of the
input text, the second it starts and the second it ends. Piper reports something
adjacent but not the same: how many samples each *phoneme* took. These do not
correspond one-to-one — "Hello" is five characters and six phonemes
(`h ə l ˈ o ʊ`), and espeak may expand a token into several words that share no
characters with the source at all.

# What is measured and what is interpolated

[FRAMING:representation] The map this module draws is honest about its
resolution, because a timing that claims to be measured and is not will be
trusted by a caption renderer to the millisecond.

Word boundaries are *measured*. espeak emits a space phoneme between words, so
the sample offset where each word begins and ends comes straight from the model
and is exact. Character boundaries *within* a word are *interpolated*: the
word's measured span is divided across its characters in proportion to their
count. So a caption that highlights whole words is exact; one that highlights
individual letters is approximate inside the word and exact at its edges.

When the phonemizer's word count disagrees with the text's — an expanded number,
an abbreviation, a symbol read aloud — no word-level correspondence exists at
all, and pretending otherwise would put every later word's timing off by a
compounding drift. That case falls back to distributing the whole utterance
across the whole text, which is markedly worse, so it does not happen quietly:
the result carries its [`Fidelity`] and the endpoints report it in an
`x-piper-alignment` header.

# Why the fidelity travels with the data

[LAW:no-silent-failure] Because the two cases produce the same shape. A caller
handed a bare list of floats cannot tell measured boundaries from a proportional
guess, and the difference is exactly the thing it would want to know before
trusting them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

# Phonemes espeak emits that mark structure rather than sound. They consume real
# time — the leading one is the model's run-up into the utterance — but they
# belong to no character, so their duration lands in the gaps between words
# rather than being charged to a letter.
_BOUNDARY_PHONEMES = frozenset({"^", "$", "\n"})

_WORD = re.compile(r"\S+")


class Fidelity(str, Enum):
    """How much of a result was measured rather than inferred."""

    #: Word boundaries came from the model; characters interpolated within words.
    WORD_EXACT = "word-exact"
    #: No word correspondence existed; the whole utterance was distributed evenly.
    INTERPOLATED = "interpolated"


@dataclass(frozen=True)
class Alignment:
    """Per-character timings for one synthesized utterance.

    The three lists are the same length as `characters`, and `characters` is the
    input text split into characters — including its spaces, because ElevenLabs
    returns those and a client indexing its own string against the response
    would otherwise slip by one at every word.
    """

    characters: list[str]
    starts: list[float]
    ends: list[float]
    fidelity: Fidelity

    def as_elevenlabs(self) -> dict:
        """The `alignment` object as the API publishes it."""
        return {
            "characters": self.characters,
            "character_start_times_seconds": self.starts,
            "character_end_times_seconds": self.ends,
        }


@dataclass(frozen=True)
class _Span:
    """A stretch of audio, in seconds since the start of the utterance."""

    start: float
    end: float


def align(
    text: str,
    phonemes: list[str],
    durations: list[int],
    sample_rate: int,
    offset: float = 0.0,
) -> Alignment:
    """Places every character of `text` on the timeline of its synthesis.

    `phonemes` and `durations` are parallel: the nth phoneme lasted the nth
    count of samples. `offset` shifts the whole result, so a caller synthesizing
    a reply in several chunks can lay them end to end without re-deriving
    anything.
    """
    characters = list(text)
    if not characters:
        return Alignment([], [], [], Fidelity.WORD_EXACT)

    total = _seconds(sum(durations), sample_rate)
    word_spans = _word_spans(phonemes, durations, sample_rate)
    text_words = list(_WORD.finditer(text))

    if len(word_spans) != len(text_words) or not text_words:
        return _spread(characters, _Span(offset, offset + total), Fidelity.INTERPOLATED)

    starts = [0.0] * len(characters)
    ends = [0.0] * len(characters)

    # Walked with a cursor over both the text and the timeline at once, so every
    # character is written exactly once and in order. Characters before the first
    # word, between words, and after the last are silence as far as the model is
    # concerned; each such run is stretched from the end of the previous word to
    # the start of the next, leaving the timeline with no holes and no overlaps.
    # A renderer stepping through it never lands on a character with no time.
    char_cursor = 0
    time_cursor = 0.0
    for match, span in zip(text_words, word_spans, strict=True):
        if match.start() > char_cursor:
            _distribute(
                starts,
                ends,
                range(char_cursor, match.start()),
                time_cursor + offset,
                span.start + offset,
            )
        _distribute(
            starts,
            ends,
            range(match.start(), match.end()),
            span.start + offset,
            span.end + offset,
        )
        char_cursor = match.end()
        time_cursor = span.end

    if char_cursor < len(characters):
        _distribute(
            starts,
            ends,
            range(char_cursor, len(characters)),
            time_cursor + offset,
            total + offset,
        )
    return Alignment(characters, starts, ends, Fidelity.WORD_EXACT)


def _distribute(
    starts: list[float],
    ends: list[float],
    indices: "range",
    begin: float,
    stop: float,
) -> None:
    """Divides `[begin, stop]` evenly across `indices`, leaving no gaps."""
    count = len(indices)
    if not count:
        return
    step = (stop - begin) / count
    for n, index in enumerate(indices):
        starts[index] = begin + step * n
        ends[index] = begin + step * (n + 1)


def _spread(characters: list[str], span: _Span, fidelity: Fidelity) -> Alignment:
    """The no-correspondence fallback: one even division over everything."""
    starts = [0.0] * len(characters)
    ends = [0.0] * len(characters)
    _distribute(starts, ends, range(len(characters)), span.start, span.end)
    return Alignment(characters, starts, ends, fidelity)


def _word_spans(
    phonemes: list[str], durations: list[int], sample_rate: int
) -> list[_Span]:
    """The measured start and end of each spoken word, in seconds.

    A word is a run of phonemes between separators. Separator duration is not
    charged to either neighbour — it is the silence the gap-filling above spans.
    """
    spans: list[_Span] = []
    elapsed = 0
    run_start: int | None = None

    for phoneme, samples in zip(phonemes, durations, strict=True):
        separator = phoneme.isspace() or phoneme in _BOUNDARY_PHONEMES
        if separator:
            if run_start is not None:
                spans.append(
                    _Span(_seconds(run_start, sample_rate), _seconds(elapsed, sample_rate))
                )
                run_start = None
        elif run_start is None:
            run_start = elapsed
        elapsed += samples

    if run_start is not None:
        spans.append(
            _Span(_seconds(run_start, sample_rate), _seconds(elapsed, sample_rate))
        )
    return spans


def _seconds(samples: int, sample_rate: int) -> float:
    return samples / float(sample_rate)
