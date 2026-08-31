"""Kokoro's own configuration, and the capability it is entitled to withhold.

`tests/test_conformance.py` already asks this engine every question the seam
asks of all of them. What is left here is what only this engine can be asked:
the environment it parses for itself, the voice ids it reads a language out of,
and — the reason this engine was chosen — that whether it can place phonemes in
time follows the export it was given rather than its name.

That last one is the ticket's subject and is worth stating plainly, because it
is not what the ticket assumed. Kokoro was picked as an engine with no phoneme
alignment at all. It turned out to have one in some of its published exports and
not in others: `kokoro_onnx` decides it as `"duration" in session.get_outputs()`,
the `model-files-v1.0` export emits only `audio`, and every `model-files-v1.1`
export emits `waveform` and `duration`. So the honest engine derives the
capability from the session, and both halves of that are checked below.
"""

from __future__ import annotations

import pytest
from conftest import KOKORO_TIMELESS_MODEL, KOKORO_VOICES, kokoro_prepared

from elvenspeak import kokoro
from elvenspeak.engine import Capability, Prosody
from elvenspeak.provisioning import ConfigError

pytestmark = pytest.mark.usefixtures("installed_assets")

TEXT = "Compatibility is measurable, and this sentence is long enough to measure."


@pytest.fixture(scope="module")
def engine(installed_assets):
    """The engine as a default deployment opens it, built once for the module."""
    return kokoro_prepared().open()


# ------------------------------------------------- the capability and its source


def test_the_export_that_reports_durations_declares_it(engine):
    """The default deployment, whose export carries a `duration` output."""
    assert Capability.TIMESTAMPS in engine.capabilities()
    assert Capability.SPEED in engine.capabilities()


def test_the_export_without_durations_withholds_the_capability(installed_assets):
    """[LAW:one-source-of-truth] The capability follows the session, not the name.

    The same engine, the same voices, the same code — one older export — and it
    stops claiming to measure. Had the answer been a constant in the engine
    module it would have been right for whichever export was current when it was
    written and quietly wrong for the other, and wrong in the expensive
    direction: a server reporting character timings it never measured.
    """
    engine = kokoro_prepared(model=KOKORO_TIMELESS_MODEL).open()

    assert Capability.TIMESTAMPS not in engine.capabilities()
    # Still a working engine, so this is a capability withheld rather than a
    # deployment broken — the distinction the whole negotiation rests on.
    assert Capability.SPEED in engine.capabilities()
    assert engine.voices()


def test_an_engine_that_cannot_measure_says_so_rather_than_inventing_a_timeline(
    installed_assets,
):
    """The claim `speak_timed`'s `measured` makes, checked where it is hardest.

    The server never reaches this: the 501 gate above refuses first, and that
    gate is the single enforcer. But the engine's answer is written to be true
    on its own — `measured` is read off whether timings really came back, not
    off the capability that gated the call — and a `measured=True` written there
    by habit would be invisible to every other test in this suite, since nothing
    else ever asks a non-measuring engine to measure.

    So this is the one place that asks. The timeline still spans the whole
    utterance, because `TimedSpeech` requires that of everyone; what changes is
    that it stops claiming the boundaries were measurements, which is what
    downstream reads to choose `Fidelity.INTERPOLATED`.
    """
    engine = kokoro_prepared(model=KOKORO_TIMELESS_MODEL).open()
    voice = engine.voices()[0]

    spoken = engine.speak_timed(voice, TEXT, Prosody())

    assert spoken.measured is False
    assert sum(timing.samples for timing in spoken.timings) * 2 == len(spoken.pcm)
    assert all(timing.separates_words for timing in spoken.timings)


def test_a_timestamps_request_is_refused_rather_than_answered_with_invented_numbers(
    installed_assets,
):
    """The property this engine was added to prove, end to end through the API.

    An engine that cannot measure must produce a refusal a caller can read, not
    an alignment derived from nothing. The refusal is assembled from
    [`Capability`]'s own sentence, so the endpoint names no engine and no
    environment variable — this is the whole 501 gate, exercised for the first
    time by an engine that really cannot do the thing.
    """
    from fastapi.testclient import TestClient

    from elvenspeak import create_app
    from elvenspeak.settings import Settings

    prepared = kokoro_prepared(model=KOKORO_TIMELESS_MODEL)
    settings = Settings(
        engine=prepared,
        fallback=KOKORO_VOICES[0],
        api_key=None,
        host="127.0.0.1",
        port=0,
    )
    client = TestClient(create_app(settings, prepared.open()))

    response = client.post(
        f"/v1/text-to-speech/{KOKORO_VOICES[0]}/with-timestamps",
        json={"text": "Hello there."},
    )

    assert response.status_code == 501
    detail = response.json()["detail"]
    assert "how long each part of an utterance took" in detail
    # No alignment anywhere in the body: a 501 that still carried a character
    # timeline would be the invented numbers under a different status code.
    assert "alignment" not in response.text


# ------------------------------------------------------------ what it speaks in


def test_the_offered_order_is_the_configured_order(engine):
    """[LAW:one-source-of-truth] The first voice offered is the default voice.

    `Engine.voices` makes this order load-bearing: a deployment naming no
    fallback answers unknown ids in whichever voice comes first. Kokoro's own
    `get_voices()` is alphabetical, so an engine that passed it straight through
    — or sorted for tidiness — would hand every such deployment `af_alloy`
    instead of the voice its operator listed first. Piper shipped exactly that
    bug and it was caught in review rather than by a test, so this is the test.
    """
    assert [voice.id for voice in engine.voices()] == list(KOKORO_VOICES)

    reversed_order = tuple(reversed(KOKORO_VOICES))
    assert [
        voice.id for voice in kokoro_prepared(voices=reversed_order).open().voices()
    ] == list(reversed_order)


@pytest.mark.parametrize(
    "key,name,language,gender",
    [
        ("af_heart", "Heart", "en-us", "female"),
        ("am_michael", "Michael", "en-us", "male"),
        ("bf_emma", "Emma", "en-gb", "female"),
        ("bm_george", "George", "en-gb", "male"),
        ("ef_dora", "Dora", "es", "female"),
        ("jm_kumo", "Kumo", "ja", "male"),
        # espeak calls Mandarin `cmn` and refuses `zh`, so a voice whose id
        # begins `z` is the one that catches a language map transcribed from the
        # id prefixes instead of from what the phonemizer accepts.
        ("zf_xiaoni", "Xiaoni", "cmn", "female"),
    ],
)
def test_a_voice_is_described_from_its_id(key, name, language, gender):
    """The id is the only source of metadata: the style pack carries no words.

    Several ids rather than one, and deliberately across languages. The language
    is not decoration — it selects the phonemizer that `speak` runs the text
    through, so a description that collapses every voice to `en-us` is also an
    engine reading Spanish with English phonemes. One English example would have
    agreed with that mutation perfectly.
    """
    voice = kokoro._describe(key)

    assert voice.id == key
    assert voice.name == name
    assert dict(voice.labels) == {
        "language": language,
        "gender": gender,
        "engine": "kokoro",
    }


def test_every_measurement_accounts_for_every_sample(engine):
    """[`TimedSpeech`]'s invariant, on the arithmetic most likely to break it.

    Kokoro reports floating-point seconds and covers only the phonemes — the
    lead-in before the first and the run-out after the last belong to none. The
    derivation rounds each boundary once and takes differences, so the rounding
    telescopes; rounding each duration on its own drifts by a sample per phoneme
    and leaves a timeline slowly parting company with the audio it describes.
    """
    spoken = engine.speak_timed(engine.voices()[0], TEXT, Prosody())

    assert spoken.measured is True
    assert spoken.timings
    assert sum(timing.samples for timing in spoken.timings) * 2 == len(spoken.pcm)


def test_the_lead_in_is_reported_as_a_gap_and_not_as_the_first_sound(engine):
    """The audio before the first phoneme belongs to no word, and says so.

    Kokoro's spans start a tenth of a second in; the audio before that is the
    model's run-up. Folded into the first phoneme it would make the first word of
    every utterance start early and last longer than it was measured to — a
    caption that leads the speech, which is the failure that looks like taste
    rather than a bug. The alternative that also keeps the samples adding up is
    to absorb the lead-in silently, so the sum is not the property that catches
    this; the first stretch's own answer is.
    """
    spoken = engine.speak_timed(engine.voices()[0], TEXT, Prosody())

    assert spoken.timings[0].separates_words is True
    assert spoken.timings[0].samples > 0


def test_word_gaps_inside_the_utterance_are_marked_too(engine):
    """[LAW:one-source-of-truth] The one place Kokoro's alphabet is interpreted.

    More than two, which is the number that matters: the lead-in and the run-out
    are separators whatever `_separates_words` answers, so a test asking only
    whether *any* stretch separates words passes for an engine that has stopped
    recognising spaces entirely. `alignment` divides each word's span across its
    characters, so with no interior gaps every word of a sentence becomes one
    word and the timeline stops being word-exact while still summing correctly.
    """
    spoken = engine.speak_timed(engine.voices()[0], TEXT, Prosody())

    gaps = [timing for timing in spoken.timings if timing.separates_words]
    assert len(gaps) > 2
    assert any(not timing.separates_words for timing in spoken.timings)


def test_a_measured_utterance_honours_speed_too(engine):
    """`speak_timed` is a second call site, and it forwards the same prosody.

    The pace test in the conformance suite drives `speak`, so a `speed` dropped
    only on this path is invisible to it — and the timestamp endpoints would
    return a correctly-summing timeline of an utterance spoken at the wrong rate,
    with the ignored header reporting the speed as honoured.
    """
    voice = engine.voices()[0]

    fast = engine.speak_timed(voice, TEXT, Prosody(speed=2.0))
    slow = engine.speak_timed(voice, TEXT, Prosody(speed=0.5))

    assert len(fast.pcm) < len(slow.pcm)
    for spoken in (fast, slow):
        assert sum(t.samples for t in spoken.timings) * 2 == len(spoken.pcm)


# --------------------------------------------------------- its own environment


def test_the_defaults_name_a_real_export_and_real_voices():
    prepared = kokoro.configure({})

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
        kokoro.configure({"KOKORO_VOICES": key})


@pytest.mark.parametrize("typo", ["tru", "yess", "0.0", "maybe"])
def test_a_boolean_that_is_not_one_is_reported_rather_than_read_as_off(typo):
    """The shared rule in `provisioning.flag`, reached through this engine.

    It was private to the Piper module until this one wanted it too. Two copies
    would have been free to drift, and a deployment learning that one engine
    rejects `tru` while another quietly reads it as "off" would be learning it
    from the audio.
    """
    with pytest.raises(ConfigError, match="KOKORO_ALLOW_DOWNLOAD"):
        kokoro.configure({"KOKORO_ALLOW_DOWNLOAD": typo})


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
        kokoro.configure({name: value})


def test_every_problem_in_this_engine_s_configuration_is_reported_together():
    """One restart, the whole list — and the list an engine contributes to it."""
    with pytest.raises(ConfigError) as raised:
        kokoro.configure(
            {
                "KOKORO_VOICES": "af_heart,nonsense",
                "KOKORO_MODELS_DIR": " ",
                "KOKORO_ALLOW_DOWNLOAD": "maybe",
            }
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


def test_acquire_describes_what_it_installed(installed_assets):
    """[LAW:parse-dont-validate] The bake's guarantee is the voices it returns.

    An engine that cannot describe what it installed has not installed it, and
    the build is the last moment that failure is cheap.
    """
    voices = kokoro_prepared().acquire()

    assert [voice.id for voice in voices] == list(KOKORO_VOICES)


def test_a_voice_that_is_not_in_the_pack_is_caught_at_install(installed_assets):
    """Caught against the file, not against a list this module keeps.

    A hard-coded roster of the 54 published voices would be a second map of the
    pack's contents, free to drift the day a pack ships a 55th — and drifting
    towards refusing voices that exist.
    """
    with pytest.raises(ValueError, match="no voice named"):
        kokoro_prepared(voices=("af_heart", "af_nonexistent")).acquire()
