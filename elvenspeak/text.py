"""Where a streamed reply may be cut into pieces.

Text in, text out. Nothing here synthesizes or encodes anything, and nothing
here knows which engine will speak the pieces — which is why it sits beside the
engine rather than inside it.
"""

from __future__ import annotations

import re

#: Sentence-final punctuation followed by whitespace. Deliberately simple: this
#: decides where a *streamed* reply is cut, and a wrong cut costs a slightly odd
#: pause, not a wrong result. Anything cleverer (abbreviations, decimals) would
#: be a second opinion about sentence boundaries competing with the one the
#: engine's own phonemizer already holds.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Cuts text where a streamed response may be broken into pieces.

    Used only by the streaming timestamp endpoint, which must align each emitted
    object against a known stretch of text.

    Returns an empty list for text with nothing to say. An earlier version ended
    `or [text]`, which cannot produce an empty list — a one-element list literal
    is always truthy — so empty input became `[""]` and reached synthesis. The
    endpoints refuse empty text at the edge; this returning nothing is the second
    half of that, so a whitespace-only string cannot slip through as one blank
    sentence.
    """
    pieces = [piece.strip() for piece in _SENTENCE_END.split(text.strip())]
    return [piece for piece in pieces if piece]
