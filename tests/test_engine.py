"""The seam's own contract: what an engine hands over cannot be written back.

A [`Voice`] is minted by an engine and then held for the life of the process by
[`elvenspeak.voices.Catalog`], which hands the same object to every request. That
makes its immutability a property the whole service depends on rather than a
decoration on the dataclass, and `frozen=True` alone does not deliver it — a
mutable field inside a frozen dataclass is still mutable.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from conftest import SERVES
from elvenspeak.engine import Voice, spoken_language


def voice() -> Voice:
    return Voice(
        id="test",
        name="Test",
        description="a voice",
        labels=(("engine", "test"), ("quality", "medium")),
        models=SERVES,
        language="en",
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


# --------------------------------------------------------------- language


@pytest.mark.parametrize(
    "tag,family",
    [
        ("es", "es"),
        ("ES", "es"),
        ("es-MX", "es"),
        ("es_MX", "es"),
        ("  es  ", "es"),
        ("en-us", "en"),
        (None, None),
    ],
)
def test_a_callers_language_tag_reduces_to_the_family_a_voice_declares(tag, family):
    """Every spelling of "Spanish" a real client sends has to reach the same voice.

    ElevenLabs publishes `language_code` as ISO 639-1, but a client holding a
    locale from a browser or an OS sends `es-MX`, and one holding a constant
    someone typed sends `ES`. A match that failed on the punctuation would report
    the language ignored while a voice that speaks it sat in the catalog — the
    failure being silent is what makes the normalisation worth a test.
    """
    assert spoken_language(tag) == family



def test_a_blank_language_is_no_language_rather_than_one_nothing_speaks():
    """What a form or a JS client sends for "unset".

    `""` reaching `Catalog.speaking` as a literal language matches no voice, so
    the catalog falls back to the whole table and the right voice still speaks —
    the damage is downstream, in `_honoured`, where `spoke` compares the voice's
    language against `""` and reports `language_code` ignored on every such
    request. A caller who expressed no preference is told their preference was
    dropped.
    """
    assert spoken_language("") is None
    assert spoken_language("   ") is None
    assert spoken_language("-") is None
    # Exported, so the untrusted shapes reach it too: a config value, a raw JSON
    # field. Reduced to the same answer rather than crashing inside a `.strip()`,
    # so no caller has to write an `isinstance` in front of the one enforcer.
    assert spoken_language(5) is None
    assert spoken_language(["es"]) is None


@pytest.mark.parametrize("declared", ["ES", " es ", "es_MX", "es-MX"])
def test_a_voice_is_stamped_with_the_language_a_caller_can_ask_for(declared):
    """Whatever an engine spells it, one spelling reaches the comparison.

    The rule had one enforcer per engine and therefore not one: `remote` routed
    its wire value through `spoken_language` and `piper` stored its sidecar's
    `family` raw, so a sidecar saying `ES` produced a voice that no caller's `es`
    could equal — filtered out of `Catalog.speaking`, reported ignored by
    `_honoured`, and still listed as speaking Spanish in `GET /v1/models`. The
    constructor is where that can be said once for every engine, including one
    written outside this repo.
    """
    assert replace(voice(), language=declared).language == "es"


@pytest.mark.parametrize("declared", ["", "   ", "-", 5, None])
def test_a_voice_that_names_no_language_is_refused_rather_than_stamped(declared):
    """There is no honest coercion of a language that was never stated.

    `str(5)` would mint a language out of a typo, and `""` is the answer-shaped
    void this field's required-ness already refuses from the other side: read as
    a real answer it says "this voice does not speak what you asked for", which
    is a claim no engine intended to make.
    """
    with pytest.raises(ValueError):
        replace(voice(), language=declared)
