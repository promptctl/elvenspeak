"""The seam's own contract: what an engine hands over cannot be written back.

A [`Voice`] is minted by an engine and then held for the life of the process by
[`elvenspeak.voices.Catalog`], which hands the same object to every request. That
makes its immutability a property the whole service depends on rather than a
decoration on the dataclass, and `frozen=True` alone does not deliver it — a
mutable field inside a frozen dataclass is still mutable.
"""

from __future__ import annotations

import pytest

from conftest import SERVES
from elvenspeak.engine import Voice


def voice() -> Voice:
    return Voice(
        id="test",
        name="Test",
        description="a voice",
        labels=(("engine", "test"), ("quality", "medium")),
        models=SERVES,
    )


def test_a_voice_cannot_be_written_through_its_labels():
    """The hazard `labels: Mapping` carried: one handler, every later caller.

    With a dict there, a request handler could assign into the labels of a voice
    the catalog goes on serving, and `GET /v1/voices` would report the change to
    everyone afterwards — with nothing raising at the point of the write.
    """
    labels = voice().labels
    with pytest.raises(TypeError):
        labels[0] = ("engine", "something-else")


def test_a_voice_is_hashable_because_it_says_it_is_frozen():
    """`frozen=True` generates `__hash__`, so it has to be true of every field.

    A dict field left that generated method raising `TypeError: unhashable` —
    the type advertising a guarantee it could not keep, which is the same defect
    as the mutability above seen from the other side.
    """
    assert len({voice(), voice()}) == 1
