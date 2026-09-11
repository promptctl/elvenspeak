"""Kokoro's own configuration, and the capability it is entitled not to have.

`speaks.py` asks this engine every question the seam asks of all of them, and it
asks the published image rather than an import. What is left here is what only
this engine can be asked: the environment it parses for itself, the voice ids it
reads a language out of, and — the reason this engine was chosen — that whether
it can place phonemes in time follows the export it was given rather than its
name.

That last one is the ticket's subject and is worth stating plainly, because it
is not what the ticket assumed. Kokoro was picked as an engine with no phoneme
alignment at all. It turned out to have one in some of its published exports and
not in others: `kokoro_onnx` decides it as `"duration" in session.get_outputs()`,
the `model-files-v1.0` export emits only `audio`, and every `model-files-v1.1`
export emits `waveform` and `duration`. So the honest engine derives the
capability from the session, and both halves of that are checked below.

# Nothing in this file opens an export

No test here downloads a model or opens an ONNX session, and the two stand-ins
that make that possible are load-bearing in opposite directions.

[`_pack`] is *real*: a style pack this file writes with numpy, holding the ids it
was asked for and a vector too small to speak with. `_install` reads it the way
it reads the published one — parses it, looks for each configured voice, reports
what it offers — so the install path is exercised rather than skipped, and the
~26 MB of published vectors were never what any of these tests were reading.

[`_Session`] is a *stand-in*: the four things this engine reads off
`kokoro_onnx.Kokoro`, and nothing else. It answers with the spans and the sample
count it was handed, which is how a timeline whose arithmetic is checkable to the
sample gets asserted at all — a real export reports what it reports, and the old
tests could only ask whether the sum came out right.

The cost of a stand-in is that the library may move underneath it, so it is held
against the real class in `test_the_stand_in_answers_the_calls_this_engine_makes`
— the one test here that imports `kokoro_onnx` for its own sake. With no real
session anywhere in the suite, a renamed `create_timed` would otherwise be found
by a caller and not by this file, which is the failure a stand-in earns.

What no stand-in can answer — that the published export really loads, that its
graph really reports durations, that a voice really speaks — is asked of the
built image by `speaks.py`, against the artifact a deployment would run rather
than against a session this machine happened to build.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

import pytest
from conftest import (
    SERVES,
    declared,
    serves,
    KOKORO_VOICES,
    kokoro_prepared,
)

from elvenspeak import kokoro
from elvenspeak.engine import Capability, Prosody, Timing
from elvenspeak.provisioning import ConfigError

if TYPE_CHECKING:  # pragma: no cover - import cost is real, the symbol is not
    import kokoro_onnx

TEXT = "Compatibility is measurable, and this sentence is long enough to measure."

#: The rate `_Session` reports, and the one every expected sample count below is
#: arithmetic against. Not Kokoro's own 24 kHz: a stand-in that answered with the
#: real constant would let a `sample_rate` read off the library instead of off the
#: session agree with every assertion here by coincidence.
#:
#: A power of two, which is the whole reason for this number. Kokoro reports spans
#: in floating-point seconds, so an expected sample count is a product of a float
#: and this — and at 10000 a span written as `0.00015` multiplies out to
#: 1.4999999999999998, which rounds the other way and makes the assertions below
#: read as arbitrary. Here every span used is a dyadic fraction, every product is
#: exact, and a boundary landing on a half sample does so because the test says so.
RATE = 8192


def _span(phoneme: str, start: float, end: float) -> "kokoro_onnx.Timing":
    """One phoneme's span, in the library's own type.

    [LAW:one-source-of-truth] `kokoro_onnx.Timing` rather than a namedtuple of the
    same three fields: `_stretches` reads `.start`, `.end` and `.phoneme` off
    whatever it is handed, so a local description of that shape would be a second
    map of the library's, free to keep agreeing with these tests after the library
    had stopped agreeing with either.
    """
    from kokoro_onnx import Timing

    return Timing(phoneme=phoneme, start=start, end=end)


@dataclass
class _Session:
    """`kokoro_onnx.Kokoro`, reduced to the four things this engine reads off it.

    An engine test needs a session that answers *chosen* audio and *chosen*
    spans. A real export answers what it answers, so the timeline it produces can
    only be checked for self-consistency — every old measurement test here summed
    the durations and compared them to the audio, which is the one property that
    holds just as well when every boundary is in the wrong place. Handed spans,
    the whole timeline is arithmetic, and the assertions below are exact.

    Honest about what it cannot do. `has_timings` is the flag the engine reads to
    decide whether this export can measure, and a session that reports `False`
    returns no spans — the pairing is a fact about the library, and a stand-in
    free to claim one without the other would let `measured` be read back off the
    capability rather than off what came back.

    [LAW:no-shared-mutable-globals] `asked` records the calls this session
    received, which is how forwarding is asserted directly instead of inferred
    from the length of the audio. It is per-instance because the engine under test
    owns exactly one.
    """

    #: How many samples every synthesis answers with. The audio is silence: this
    #: file asserts where the boundaries fall, and `test_encoding.py` is where
    #: sample values mean anything.
    samples: int = 0
    #: The per-phoneme spans `create_timed` reports, in seconds.
    spans: tuple = ()
    has_timings: bool = True
    asked: list[dict] = field(default_factory=list)

    def create(self, text, voice, speed=1.0, lang="en-us"):
        return self._answer("create", text, voice, speed, lang)[:2]

    def create_timed(self, text, voice, speed=1.0, lang="en-us"):
        return self._answer("create_timed", text, voice, speed, lang)

    def _answer(self, call, text, voice, speed, lang):
        import numpy

        self.asked.append(
            {"call": call, "text": text, "voice": voice, "speed": speed, "lang": lang}
        )
        spans = list(self.spans) if self.has_timings else []
        return numpy.zeros(self.samples, dtype=numpy.float32), RATE, spans


def _speaking(session: _Session, *keys: str) -> kokoro.KokoroEngine:
    """`session` as the engine a deployment naming `keys` would be serving.

    Built the way `_Prepared.open` builds it — the voices described from their
    ids, then put through `_declaring` so they carry what this session can do —
    because an engine assembled any other way could serve a voice claiming a
    capability its session does not have, which is the state this module is
    arranged to make unreachable.
    """
    installed = {key: kokoro._describe(key, SERVES) for key in keys}
    return kokoro.KokoroEngine(
        session, kokoro._declaring(session, installed), sample_rate=RATE
    )


def _pack(models_dir, *keys: str):
    """A style pack holding exactly `keys`, and an export file beside it.

    Real npz rather than a monkeypatched `numpy.load`: `_install` opens this file,
    reads its member names and reports what it offers, so writing one keeps that
    whole path under test at the cost of a few hundred bytes. The vectors are a
    single zero — nothing here speaks, and the pack's only job in this engine is
    to answer whether a configured id is in it.

    The export is a placeholder for the same reason `_fetch` is satisfied by any
    file at the name: nothing in this file opens it. What proves a real export
    loads is `speaks.py`, against the image that baked it.
    """
    import numpy

    models_dir.mkdir(parents=True, exist_ok=True)
    voices = models_dir / kokoro._VOICES_FILE
    with voices.open("wb") as handle:
        numpy.savez(handle, **{key: numpy.zeros(1, dtype=numpy.float32) for key in keys})
    (models_dir / kokoro.DEFAULT_MODEL).write_bytes(b"not opened here")
    return models_dir


@pytest.fixture
def opens(monkeypatch):
    """`kokoro_onnx.Kokoro` answered by [`_Session`], and every session it opened.

    The lifecycle methods reach the library by name at call time — `_open` does
    its own `from kokoro_onnx import ...` — so replacing the attribute is enough,
    and everything above it runs for real: the assets are fetched, the pack is
    parsed, the voices are found, `_declaring` reads the session. What is skipped
    is the ~145 MB parse of a graph nothing here asks a question of.

    Yields the list rather than one session because `acquire` and `open` each open
    their own, and "how many were opened" is a property one of the tests below is
    about.
    """
    import kokoro_onnx

    opened: list[_Session] = []

    def _session(_model_path, _voices_path):
        opened.append(_Session())
        return opened[-1]

    monkeypatch.setattr(kokoro_onnx, "Kokoro", _session)
    return opened


# ---------------------------------------------- the stand-in, held to the library


def test_the_stand_in_answers_the_calls_this_engine_makes():
    """[LAW:one-source-of-truth] `_Session` is a map of `Kokoro`; this redraws it.

    Every other test in this file speaks to the stand-in, so the library is free
    to rename a method or reorder a return without a single one of them going
    red — the engine would keep passing here and fail on the first request an
    image served. This is the only thing standing in that gap, which is why it
    imports the real class rather than reading the stand-in back.

    Names and parameters rather than behaviour: what `create_timed` *does* is the
    library's business and is proved against the artifact by `speaks.py`. What
    this engine depends on is that it exists, that it accepts these four
    arguments by these names, and that a session carries `has_timings` — every
    one of which is a promise the stand-in makes on the library's behalf.
    """
    import inspect

    import kokoro_onnx

    for call in ("create", "create_timed"):
        real = inspect.signature(getattr(kokoro_onnx.Kokoro, call)).parameters
        # By name, because the engine passes them by name. A library that made
        # `lang` positional-only would leave this file green and every synthesis
        # a TypeError.
        assert {"text", "voice", "speed", "lang"} <= set(real), call
        stood_in = inspect.signature(getattr(_Session, call)).parameters
        assert set(stood_in) - {"self"} == {"text", "voice", "speed", "lang"}, call

    # Read out of the source because `has_timings` is settled per instance, from
    # the graph's outputs, and there is no instance here to ask — which is the
    # whole point of the file. Textual and therefore weak, and still the only
    # thing between a renamed attribute and an `AttributeError` in `_declaring`
    # on the first boot of an image.
    assert "self.has_timings" in inspect.getsource(kokoro_onnx.Kokoro)

    assert [f.name for f in fields(kokoro_onnx.Timing)] == ["phoneme", "start", "end"]


# ------------------------------------------------- the capability and its source


def test_the_export_that_reports_durations_declares_it():
    """The default deployment, whose export carries a `duration` output."""
    speaking = _speaking(_Session(has_timings=True), *KOKORO_VOICES)

    # Per voice, not merely somewhere in the union: `open()` claims every voice
    # this export speaks carries the same set, and a stamp applied to only one of
    # the voices a deployment loads would satisfy `declared()` while leaving the
    # rest silently incapable.
    assert all(Capability.TIMESTAMPS in v.capabilities for v in speaking.voices())
    assert all(Capability.SPEED in v.capabilities for v in speaking.voices())


def test_the_export_without_durations_does_not_declare_the_capability():
    """[LAW:one-source-of-truth] The capability follows the session, not the name.

    The same engine, the same voices, the same code — one older export — and it
    stops claiming to measure. Had the answer been a constant in the engine
    module it would have been right for whichever export was current when it was
    written and quietly wrong for the other, and wrong in the expensive
    direction: a server reporting character timings it never measured.

    Asked of a session rather than of the two published exports, which is what
    `_declaring` actually reads: the file name never reached it, and downloading
    145 MB of `model-files-v1.0` to set one boolean was the long way round to
    setting it.
    """
    speaking = _speaking(_Session(has_timings=False), *KOKORO_VOICES)

    assert Capability.TIMESTAMPS not in declared(speaking)
    # Still a working engine, so this is a capability absent rather than a
    # deployment broken — the distinction the whole negotiation rests on.
    assert Capability.SPEED in declared(speaking)
    assert speaking.voices()


def test_an_engine_that_cannot_measure_says_so_rather_than_inventing_a_timeline():
    """The claim `speak_timed`'s `measured` makes, checked where it is hardest.

    The server never reaches this: the 501 gate refuses first, and that gate is
    the single enforcer. But the engine's answer is written to be true on its own
    — `measured` is read off whether timings really came back, not off the
    capability that gated the call — and a `measured=True` written there by habit
    would be invisible to every other test in this suite, since nothing else ever
    asks a non-measuring engine to measure.

    So this is the one place that asks. The timeline still spans the whole
    utterance, because `TimedSpeech` requires that of everyone; what changes is
    that it stops claiming the boundaries were measurements, which is what
    downstream reads to choose `Fidelity.INTERPOLATED`.

    One stretch and not merely "every stretch separates words": with nothing
    attributed to anything there is nothing to divide the utterance at, so an
    answer of several separators would be boundaries invented and then marked as
    gaps — the same fabrication in a costume the weaker assertion accepts.
    """
    speaking = _speaking(_Session(samples=RATE, has_timings=False), *KOKORO_VOICES)

    spoken = speaking.speak_timed(speaking.voices()[0], TEXT, Prosody())

    assert spoken.measured is False
    assert spoken.timings == (Timing(samples=RATE, separates_words=True),)
    assert sum(timing.samples for timing in spoken.timings) * 2 == len(spoken.pcm)


# ------------------------------------------------------------ what it speaks in


def test_the_offered_order_is_the_configured_order(tmp_path, opens):
    """[LAW:one-source-of-truth] The first voice offered is the default voice.

    `Engine.voices` makes this order load-bearing: a deployment naming no
    fallback answers unknown ids in whichever voice comes first. Kokoro's own
    `get_voices()` is alphabetical, so an engine that passed it straight through
    — or sorted for tidiness — would hand every such deployment `af_alloy`
    instead of the voice its operator listed first. Piper shipped exactly that
    bug and it was caught in review rather than by a test, so this is the test.

    Driven through `open` rather than by constructing the engine, because the
    order is set where the voices are described from the configured keys and an
    engine handed a dict already in the right order would agree with any
    arrangement of that code at all.
    """
    _pack(tmp_path, *KOKORO_VOICES)

    ordered = kokoro_prepared(tmp_path).open()
    assert [voice.id for voice in ordered.voices()] == list(KOKORO_VOICES)

    reversed_order = tuple(reversed(KOKORO_VOICES))
    reversed_engine = kokoro_prepared(tmp_path, voices=reversed_order).open()
    assert [voice.id for voice in reversed_engine.voices()] == list(reversed_order)


@pytest.mark.parametrize(
    "key,name,language,iso,gender",
    [
        ("af_heart", "Heart", "en-us", "en", "female"),
        ("am_michael", "Michael", "en-us", "en", "male"),
        ("bf_emma", "Emma", "en-gb", "en", "female"),
        ("bm_george", "George", "en-gb", "en", "male"),
        ("ef_dora", "Dora", "es", "es", "female"),
        ("jm_kumo", "Kumo", "ja", "ja", "male"),
        # espeak calls Mandarin `cmn` and refuses `zh`, so a voice whose id
        # begins `z` is the one that catches a language map transcribed from the
        # id prefixes instead of from what the phonemizer accepts. It is also the
        # one row that exercises `_ISO_SPELLINGS`: a naive `.split("-")[0]`
        # derives the right ISO code for every other row and `cmn` for this one.
        ("zf_xiaoni", "Xiaoni", "cmn", "zh", "female"),
    ],
)
def test_a_voice_is_described_from_its_id(key, name, language, iso, gender):
    """The id is the only source of metadata: the style pack carries no words.

    Several ids rather than one, and deliberately across languages. The language
    is not decoration — it selects the phonemizer that `speak` runs the text
    through, so a description that collapses every voice to `en-us` is also an
    engine reading Spanish with English phonemes. One English example would have
    agreed with that mutation perfectly.

    Both spellings are asserted because both are load-bearing and they are not
    interchangeable: espeak's is what the phonemizer is handed, ISO 639-1 is what
    a caller's `language_code` is matched against. The `en-us`/`en` rows catch a
    region suffix leaking into the match, and `cmn`/`zh` catches the one language
    espeak and ISO name differently.
    """
    voice = kokoro._describe(key, SERVES)

    assert voice.id == key
    assert voice.name == name
    assert voice.language == iso
    assert kokoro._language(key) == language
    # The language is a field now, not a label. It was published here while
    # nothing read it; a reader made it a field, and leaving the label behind
    # would be the same fact in two places, free to disagree.
    assert dict(voice.labels) == {"gender": gender, "engine": "kokoro"}


def test_every_language_the_map_names_is_one_espeak_accepts():
    """[LAW:verifiable-goals] The map is checked against the backend, not read back.

    `_describe` only copies these values into a label, so every other test here
    would pass just as well on a table of invented codes. Only the backend knows
    which ones it accepts.
    """
    from phonemizer.backend import EspeakBackend

    supported = EspeakBackend.supported_languages()

    assert set(kokoro._VOICE_LANGUAGES.values()) <= set(supported)
    assert "zh" not in supported


def test_every_measurement_accounts_for_every_sample():
    """The whole timeline, to the sample, from spans chosen to produce all of it.

    Kokoro reports floating-point seconds and covers only the phonemes — the
    lead-in before the first and the run-out after the last belong to none — so
    three separate things have to come out of one derivation: the run-up is a gap
    rather than the start of the first word, a space between phonemes is a gap
    too, and every sample is accounted for exactly once.

    Asserted as the tuple rather than as three properties of it, which is what
    the stand-in bought. Against a real export the spans are whatever the model
    said, so the only checkable claim was the sum — and the sum holds just as
    well when every boundary inside it is in the wrong place. Here the spans are
    chosen, so each of the five stretches below is a number with a reason:

        1024   the lead-in, before the first phoneme, marked as a gap
        1024   the phoneme itself, which is not a gap
        1024   the space between the two, which is
        2048   the second phoneme
        3072   the run-out, after the last phoneme, marked as a gap

    A different implementation of the same contract has no freedom here; there is
    exactly one timeline that describes these spans over this many samples.
    """
    session = _Session(
        samples=RATE,
        spans=(
            _span("h", 0.125, 0.25),
            _span(" ", 0.25, 0.375),
            _span("ə", 0.375, 0.625),
        ),
    )
    speaking = _speaking(session, *KOKORO_VOICES)

    spoken = speaking.speak_timed(speaking.voices()[0], TEXT, Prosody())

    assert spoken.measured is True
    assert spoken.timings == (
        Timing(samples=1024, separates_words=True),
        Timing(samples=1024, separates_words=False),
        Timing(samples=1024, separates_words=True),
        Timing(samples=2048, separates_words=False),
        Timing(samples=3072, separates_words=True),
    )
    assert sum(timing.samples for timing in spoken.timings) * 2 == len(spoken.pcm)


def test_the_rounding_telescopes_rather_than_drifting_a_sample_per_phoneme():
    """The reason boundaries are rounded once and durations are differences.

    Every span here is one and a half samples wide, which is the shape the two
    derivations disagree about. Rounding each *duration* on its own gives four
    stretches of 2 and a timeline 8 samples long describing 6 samples of audio —
    green under every sum-free assertion, and a caption drifting further behind
    the speech with each phoneme. Rounding each *boundary* once and subtracting
    gives 2, 1, 1, 2: the error cancels at the next boundary instead of
    accumulating, and the total is exact by construction rather than by luck.

    Six samples rather than a realistic utterance because the drift is one sample
    per phoneme either way — at this size it is the whole answer, and at a
    realistic size it is a rounding error nobody would read off a failure.
    """
    step = 3 / (2 * RATE)
    session = _Session(
        samples=6,
        spans=tuple(
            _span(chr(ord("a") + n), n * step, (n + 1) * step) for n in range(4)
        ),
    )
    speaking = _speaking(session, *KOKORO_VOICES)

    spoken = speaking.speak_timed(speaking.voices()[0], TEXT, Prosody())

    assert spoken.timings == (
        Timing(samples=2, separates_words=False),
        Timing(samples=1, separates_words=False),
        Timing(samples=1, separates_words=False),
        Timing(samples=2, separates_words=False),
    )


@pytest.mark.parametrize(
    "method,reaches,drain",
    [
        # Which library call each method reaches, and what it answers with — the
        # two things the call sites differ in, passed as values rather than
        # branched on. `speak` hands back an unstarted generator; `speak_timed`
        # hands back finished bytes.
        ("speak", "create", lambda spoken: list(spoken.audio)),
        ("speak_timed", "create_timed", lambda spoken: spoken.pcm),
    ],
)
def test_both_call_sites_hand_the_session_the_speed_and_the_voices_language(
    method, reaches, drain
):
    """The two arguments an engine can drop without producing a single error.

    `speak_timed` is a second call site into the same library and forwards the
    same prosody, so a `speed` dropped only there returns a correctly-summing
    timeline of an utterance spoken at the wrong rate — with the response header
    reporting the speed as honoured. It was checked by comparing the lengths of
    two utterances, which a `create_timed` that honoured speed while ignoring
    `lang` also passes.

    `lang` is the other half and the more expensive one: it selects the
    phonemizer, so a voice whose language never reaches the library is read aloud
    in English phonemes — fluent, confident, and wrong, with no error anywhere.
    `bf_emma` is `en-gb` and ships in `DEFAULT_VOICES`, so a default deployment
    speaks it; that this is a language espeak really accepts is the separate
    claim `test_every_language_the_map_names_is_one_espeak_accepts` makes against
    the backend.

    Read off the call the session received rather than off the audio it returned,
    because the audio is the library's answer and this is a question about what
    the library was asked.
    """
    session = _Session(samples=RATE, spans=(_span("h", 0.125, 0.25),))
    speaking = _speaking(session, "bf_emma")

    # Drained rather than merely called: `speak` synthesizes nothing until its
    # samples are pulled, which is the property `_stream` exists for, so an
    # undrained answer reaches the library not at all.
    drain(getattr(speaking, method)(speaking.voices()[0], TEXT, Prosody(speed=2.0)))

    assert session.asked == [
        {
            "call": reaches,
            "text": TEXT,
            "voice": "bf_emma",
            "speed": 2.0,
            "lang": "en-gb",
        }
    ]


# --------------------------------------------------------- its own environment


def test_the_defaults_name_a_real_export_and_real_voices():
    prepared = kokoro.configure({}, frozenset(), SERVES)

    assert prepared.model == kokoro.DEFAULT_MODEL
    assert prepared.keys == kokoro.DEFAULT_VOICES


@pytest.mark.parametrize(
    "key", ["heart", "af-heart", "xf_heart", "ax_heart", "af_", "a_heart"]
)
def test_a_voice_id_it_cannot_read_a_language_out_of_is_refused(key):
    """[LAW:no-silent-failure] A guessed language is fluent-sounding nonsense.

    The first character selects the phonemizer. An id this module cannot read
    would have to be given a default language, and a Spanish voice reading
    Spanish text through the English phonemizer produces audio that plays
    perfectly and is wrong — the silent wrong answer this service exists to stop
    making. Refused at the parse rather than discovered at synthesis.
    """
    with pytest.raises(ConfigError, match="KOKORO_VOICES"):
        kokoro.configure({"KOKORO_VOICES": key}, frozenset(), SERVES)


@pytest.mark.parametrize("typo", ["tru", "yess", "0.0", "maybe"])
def test_a_boolean_that_is_not_one_is_reported_rather_than_read_as_off(typo):
    """The shared rule in `provisioning.flag`, reached through this engine.

    It was private to the Piper module until this one wanted it too. Two copies
    would have been free to drift, and a deployment learning that one engine
    rejects `tru` while another quietly reads it as "off" would be learning it
    from the audio.
    """
    with pytest.raises(ConfigError, match="KOKORO_ALLOW_DOWNLOAD"):
        kokoro.configure({"KOKORO_ALLOW_DOWNLOAD": typo}, frozenset(), SERVES)


@pytest.mark.parametrize(
    "name,value",
    [("KOKORO_MODELS_DIR", "   "), ("KOKORO_MODEL", "  "), ("KOKORO_VOICES", " , ")],
)
def test_a_present_but_blank_setting_is_not_an_absent_one(name, value):
    """An unset variable interpolated into a compose file is how you get here.

    `KOKORO_MODELS_DIR=` is a present key, so `get` returns "" rather than the
    default and `Path("")` is the working directory — the server would then read
    and write ~140 MB of assets wherever it happened to be launched from, having
    reported nothing.
    """
    with pytest.raises(ConfigError, match=name):
        kokoro.configure({name: value}, frozenset(), SERVES)


def test_every_problem_in_this_engine_s_configuration_is_reported_together():
    """One restart, the whole list — and the list an engine contributes to it."""
    with pytest.raises(ConfigError) as raised:
        kokoro.configure(
            {
                "KOKORO_VOICES": "af_heart,nonsense",
                "KOKORO_MODELS_DIR": " ",
                "KOKORO_ALLOW_DOWNLOAD": "maybe",
            },
            frozenset(),
            SERVES,
        )

    joined = " ".join(raised.value.problems)
    assert len(raised.value.problems) == 3
    for expected in ("KOKORO_VOICES", "KOKORO_MODELS_DIR", "KOKORO_ALLOW_DOWNLOAD"):
        assert expected in joined


# ------------------------------------------------------------- getting it ready


def test_a_missing_asset_with_downloading_off_fails_loudly(tmp_path):
    """A deployment problem is a refusal to boot, not a fetch nobody asked for.

    The image bakes its assets and turns downloading off, so a missing file at
    boot means the image is wrong — and re-downloading it would paper over that
    while making the container depend on GitHub being reachable to serve.
    """
    with pytest.raises(FileNotFoundError, match="downloading is off"):
        kokoro_prepared(tmp_path, allow_download=False).open()


def test_an_unnamed_espeak_is_refused_before_the_export_is_opened(tmp_path):
    """The abort this engine used to take on its first request, moved to boot.

    `espeakng-loader` aborts the process rather than failing to load, so an
    unnamed library was not a 500 anyone could read: the server loaded its
    voices, answered `/health` with 200, and died without a traceback on the
    first real synthesis.

    `tmp_path` holds no assets, and the test above shows what opening it costs
    without this refusal — a `FileNotFoundError` from the asset install. Getting
    `ConfigError` here is therefore the ordering claim as well as the refusal
    one: nothing is fetched or opened before the library is named.
    """
    prepared = kokoro.configure(
        {"KOKORO_MODELS_DIR": str(tmp_path), "KOKORO_ALLOW_DOWNLOAD": "0"},
        frozenset(),
        SERVES,
    )

    with pytest.raises(ConfigError) as raised:
        prepared.open()

    assert kokoro.ESPEAK_LIBRARY in str(raised.value)
    # Every one of them, because the refusal's job is to tell an operator on any
    # platform where to look — and because naming them is the whole of what this
    # engine is allowed to do with the list.
    for candidate in kokoro.ESPEAK_LIBRARY_CANDIDATES:
        assert candidate in str(raised.value)


def test_the_espeak_library_is_read_at_the_parse_and_refused_only_at_open():
    """The bake needs no phonemes, so the shared parse must not refuse for them.

    [LAW:one-source-of-truth] `configure` is the one place this engine holds a
    string out of the environment, so it reads the variable — but `python -m
    elvenspeak.bake` goes through the same parse and synthesizes nothing, and
    refusing it for a library it will never call would make an espeak-less
    machine unable to bake assets it could serve from elsewhere. Same split
    `main._app` makes for `unsized`.
    """
    assert kokoro.configure({}, frozenset(), SERVES).espeak_library == ""
    assert (
        kokoro.configure(
            {kokoro.ESPEAK_LIBRARY: "  /usr/lib/libespeak-ng.so.1  "},
            frozenset(),
            SERVES,
        ).espeak_library
        == "/usr/lib/libespeak-ng.so.1"
    )


def test_a_download_that_produces_nothing_is_a_failure_not_an_install(
    tmp_path, monkeypatch
):
    """[LAW:no-silent-failure] `urlopen` reports success by returning.

    A proxy error page, a full disk, or a release asset that has been withdrawn
    all arrive as a clean response with a short or empty body. Without this check
    the empty file is renamed into place and every later run treats it as
    installed — the failure resurfaces as an opaque ONNX parse error naming
    neither the download nor the asset, on a machine where re-running the build
    fixes nothing because the file already exists.
    """
    import urllib.request

    class _EmptyResponse:
        def read(self, _size):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda _url: _EmptyResponse())

    with pytest.raises(OSError, match="empty"):
        kokoro_prepared(tmp_path, allow_download=True).acquire()

    # And it left nothing behind that a later run would mistake for an install.
    assert not list(tmp_path.glob("*.onnx"))


@pytest.mark.parametrize(
    "sample,expected",
    [(0.0, 0), (1.0, 32767), (-1.0, -32767), (2.5, 32767), (-2.5, -32767)],
)
def test_samples_outside_the_nominal_range_clip_rather_than_wrap(sample, expected):
    """A model output above 1.0 wraps to a maximally negative sample uncled.

    That is an audible click, and it is invisible to every test of lengths,
    rates and sums — the audio is exactly as long as it should be and exactly as
    wrong. Kokoro's outputs sat well inside the range in every sentence tried,
    which is precisely why this is asserted rather than assumed: a quiet
    invariant that happens to hold is one nobody notices breaking.
    """
    import numpy

    pcm = kokoro._pcm(numpy.array([sample], dtype=numpy.float32))

    assert int.from_bytes(pcm, "little", signed=True) == expected


def test_an_export_that_downloaded_whole_and_is_not_a_model_fails_the_bake(tmp_path):
    """Presence is not readability, and the build is where that has to be caught.

    A `.onnx` can arrive complete, non-empty, and still not be a loadable graph:
    a release asset replaced in place, a corrupted cache, a bind mount that
    received a partial copy from somewhere the size check never saw. Every file
    check there is passes on that, so the only thing that finds it is opening it.

    This is the same lesson as the voices bake: an earlier version of that
    stopped at "both files exist", which let a truncated sidecar produce a green
    image that failed at container startup instead. `acquire` opens the session
    and discards it precisely so this fails here, where it is cheap, rather than
    on every container start forever.

    The one test in this file that reaches the real `kokoro_onnx.Kokoro`, and it
    needs no published export to do it — the property is that a *bad* file is
    refused, and a bad file is thirty-six bytes. `_pack` supplies the sidecar so
    the failure has to come from the graph and not from the file beside it.
    """
    _pack(tmp_path, *KOKORO_VOICES)
    (tmp_path / kokoro.DEFAULT_MODEL).write_bytes(b"complete, non-empty, and not a model")

    with pytest.raises(Exception) as raised:
        kokoro_prepared(tmp_path).acquire()

    # Not the empty-file guard and not a missing-file error: those would mean the
    # bake had rejected it for a reason that says nothing about the graph.
    assert "empty" not in str(raised.value)
    assert not isinstance(raised.value, FileNotFoundError)


def test_acquire_describes_what_it_installed(tmp_path, opens):
    """[LAW:parse-dont-validate] The bake's guarantee is the voices it returns.

    An engine that cannot describe what it installed has not installed it, and
    the build is the last moment that failure is cheap.
    """
    _pack(tmp_path, *KOKORO_VOICES)
    prepared = kokoro_prepared(tmp_path)

    voices = prepared.acquire()
    served = prepared.open().voices()

    assert [voice.id for voice in voices] == list(KOKORO_VOICES)

    # [FRAMING:representation] And it describes them the way they will boot. A
    # `Voice` states what speaking in it really does, so a build that reported
    # them capability-less while `open` serves them able to measure would be two
    # descriptions of one voice, disagreeing.
    assert {voice.id: voice.capabilities for voice in voices} == {
        voice.id: voice.capabilities for voice in served
    }
    assert all(Capability.TIMESTAMPS in voice.capabilities for voice in voices)

    # The other thing a `Voice` states about its speaker, held to the same
    # standard: `_declaring` runs on both paths, so a `serves` dropped at one of
    # them describes a voice the build can reach by engine name and the boot
    # cannot. Compared against the real declaration too, since agreeing on the
    # wrong set is what a dropped argument would also look like.
    assert {voice.id: voice.models for voice in voices} == {
        voice.id: voice.models for voice in served
    }
    assert all(voice.models == serves("kokoro") for voice in voices)

    # One session each, and two in all: both lifecycle methods really opened one,
    # which is what makes the two descriptions above answers from the session
    # rather than from the filename it was opened by.
    assert len(opens) == 2


def test_a_voice_that_is_not_in_the_pack_is_caught_at_install(tmp_path, opens):
    """Caught against the file, not against a list this module keeps.

    A hard-coded roster of the 54 published voices would be a second map of the
    pack's contents, free to drift the day a pack ships a 55th — and drifting
    towards refusing voices that exist.

    Which is why the pack here holds exactly one voice and the deployment names
    two. A synthetic pack cannot prove a name the published one really carries —
    that is `speaks.py`'s question, against an image whose bake really ran — but
    it proves the only thing this code does with a pack: read what is in it, and
    refuse what is not.
    """
    _pack(tmp_path, "af_heart")

    with pytest.raises(ValueError, match="no voice named 'af_nonexistent'"):
        kokoro_prepared(tmp_path, voices=("af_heart", "af_nonexistent")).acquire()

    assert not opens, "the session was opened for a deployment already known bad"
