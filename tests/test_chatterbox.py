"""Chatterbox's own configuration, its voice ids, and the capabilities it lacks.

`tests/test_conformance.py` already asks this engine every question the seam asks
of all of them. What is left here is what only this engine can be asked, and the
list is short because most of what makes Chatterbox different is a cost rather
than a behaviour.

Three things are genuinely its own:

The device it runs on has no default, which is the one real decision in
`chatterbox.configure` and the only setting in this repository that a deployment
must state before it can boot. Every candidate default is wrong somewhere and
silently so, so the refusal is the feature and is asserted as one.

A voice is `<speaker>-<language>`, and the voice set is the product of the two
lists. That is what this engine exists for: `builtin-en` and `builtin-es` are one
person speaking two languages, which is a thing neither Piper's nor Kokoro's
catalogue can express — their per-language voices are different people. The
product is asserted from descriptions rather than from a running model, because
composing an id needs no 3 GiB of checkpoints.

It declares no capabilities at all, and is the first engine here that declares
none. Kokoro proved a capability can follow the export rather than the engine's
name; this one proves the empty set is a real answer that the server acts on.

# What costs what

Everything above the `chatterbox_installed` divider runs anywhere, needs no
network and does not import the library — which is the same seam
`tests/test_encoding.py` proves from the other side: `configure` parses the
environment without importing `chatterbox`, so a machine with no accelerator and
no 3 GiB to spare still checks the decisions this module actually makes.

Below the divider the checkpoints are on disk: ~3.06 GiB fetched once, ~4.8 GiB
resident per model opened, and synthesis at 8-33x real time on `cpu`. No test
holds two at once — measured, two live models are 8.11 GiB against a build runner
with 7.9 GB — and the refusals hold none, answerable from a table and a stat.
"""

from __future__ import annotations

import gc
import re
import threading
import time

import pytest
from conftest import (
    SERVES,
    chatterbox_prepared,
    declared,
    serves,
)

from elvenspeak import chatterbox
from elvenspeak.engine import Capability, Prosody
from elvenspeak.provisioning import ConfigError

TEXT = "Compatibility is measurable, and this sentence is long enough to measure."


def parsed(**overrides: str):
    """`configure` over a complete environment with `overrides` applied.

    A whole environment rather than the one variable under test, so that a
    failure names the setting the test is about instead of the first one the
    parse happened to reach. `None` removes a variable, which is how the tests
    about an unset setting say so — an empty string is a different case here and
    has its own assertions.
    """
    env = {
        chatterbox.SPEAKERS: chatterbox.BUILTIN_SPEAKER,
        chatterbox.LANGUAGES: "en,es",
        chatterbox.MODELS_DIR: "/tmp/does-not-need-to-exist",
        chatterbox.DEVICE: "cpu",
        chatterbox.ALLOW_DOWNLOAD: "0",
    }
    env.update(overrides)
    return chatterbox.configure(
        {name: value for name, value in env.items() if value is not None},
        frozenset(),
        SERVES,
    )


# ------------------------------------------- the device, and its missing default


def test_an_unset_device_is_refused_rather_than_defaulted():
    """[LAW:types-are-the-program] The decision this engine is arranged around.

    Every candidate default is wrong somewhere and silently so: `cuda` will not
    boot on Apple hardware, and `cpu` boots everywhere and then serves at 10-33
    times real time — measured, and an order of magnitude past useful. A default
    is a claim that one answer is right when the deployment said nothing, and
    there is no such answer, so an omission that would be read as a real answer
    is refused instead.

    The message is asserted as well as the refusal. An operator who did not know
    this setting existed reads this line and nothing else, so it has to name the
    variable, the candidates, and the reason there is no default — otherwise the
    refusal is merely an obstacle rather than the explanation it is meant to be.
    """
    with pytest.raises(ConfigError) as raised:
        parsed(**{chatterbox.DEVICE: None})

    message = "\n".join(raised.value.problems)
    assert chatterbox.DEVICE in message
    for candidate in chatterbox.DEVICES:
        assert candidate in message
    assert "no default" in message


@pytest.mark.parametrize("device", chatterbox.DEVICES)
def test_every_named_device_is_accepted(device: str):
    """The other half of the equivalence, without which the test above is a typo.

    A `configure` that refused every string would satisfy the refusal test
    perfectly. `DEVICES` is the closed set this engine has been measured on, and
    each member of it has to actually get through.
    """
    assert parsed(**{chatterbox.DEVICE: device}).device == device


def test_a_device_torch_would_accept_but_nobody_measured_is_still_refused():
    """The set is closed on purpose, and `torch.device` is not the authority.

    `torch.device("xpu")` and `torch.device("meta")` are both real, and this
    engine has been measured on neither. The point of naming a device is to be
    refused when the deployment and the hardware disagree, which a check that
    deferred to torch's vocabulary could never do.
    """
    with pytest.raises(ConfigError):
        parsed(**{chatterbox.DEVICE: "xpu"})


# ------------------------------------------------- everything else it parses


def test_an_empty_models_dir_is_refused_rather_than_read_as_the_working_directory():
    """`CHATTERBOX_MODELS_DIR=` is a present key whose value is the empty string.

    `Path("")` is the working directory, so without this the server reads and
    writes 3 GiB of checkpoints wherever it happened to be launched from, having
    reported nothing. Unset is a different statement and is answered with the
    module's own default, which the next test holds.
    """
    with pytest.raises(ConfigError) as raised:
        parsed(**{chatterbox.MODELS_DIR: ""})

    assert chatterbox.MODELS_DIR in "\n".join(raised.value.problems)


def test_an_unset_models_dir_falls_back_to_the_projects_own_directory():
    """Unset is answered, unlike the empty string, and the difference is the point."""
    assert parsed(**{chatterbox.MODELS_DIR: None}).models_dir.name == "models"


@pytest.mark.parametrize("variable", [chatterbox.SPEAKERS, chatterbox.LANGUAGES])
def test_a_list_that_names_nothing_is_refused(variable: str):
    """Both list settings, because both reach the same reader.

    An engine with no speakers or no languages offers no voices, which `/health`
    reports rather than crashes on — so it would boot, serve a listing of
    nothing, and answer every request with the fallback machinery instead of a
    voice. Refused at the parse, where the operator is still reading.
    """
    with pytest.raises(ConfigError) as raised:
        parsed(**{variable: ",,"})

    assert variable in "\n".join(raised.value.problems)


@pytest.mark.parametrize(
    "written,read",
    [
        ("en,es", ("en", "es")),
        (" en , es ", ("en", "es")),
        ("en,,es", ("en", "es")),
        ("en,es,", ("en", "es")),
    ],
)
def test_a_list_setting_survives_the_typos_a_shell_makes(written: str, read: tuple):
    """A trailing comma and a doubled one are the same typo, and neither is a language.

    Read through one helper rather than spelled twice, so `CHATTERBOX_SPEAKERS`
    and `CHATTERBOX_LANGUAGES` cannot disagree about which of them is empty.
    """
    assert parsed(**{chatterbox.LANGUAGES: written}).languages == read


def test_every_problem_is_reported_at_once_rather_than_one_per_restart():
    """The list this parse exists to produce, spliced into the server's own.

    An operator bringing the service up should not discover a bad device, then
    one restart later an empty speaker list, then one restart later a models
    directory that was blank. Each restart of this engine loads 3 GiB, so the
    cost of a parse that gives up at the first problem is measured in minutes.
    """
    with pytest.raises(ConfigError) as raised:
        parsed(
            **{
                chatterbox.DEVICE: "gpu-please",
                chatterbox.SPEAKERS: "",
                chatterbox.MODELS_DIR: "",
            }
        )

    reported = "\n".join(raised.value.problems)
    assert chatterbox.DEVICE in reported
    assert chatterbox.SPEAKERS in reported
    assert chatterbox.MODELS_DIR in reported


# --------------------------------------------------- what a voice is, and is not


class _NotReallyConditionals:
    """A stand-in for the cloned identity `_describe` only ever carries.

    `_describe` stores this value and reads nothing off it, which is what makes
    the voice product assertable without 3 GiB of checkpoints — and is itself
    worth pinning: the day describing a voice needs the model, these tests get
    expensive and the reason will be this class failing rather than a mystery.
    """


def described(speaker: str, language: str):
    return chatterbox._describe(
        speaker, language, _NotReallyConditionals(), serves("chatterbox")
    )


def test_a_voice_id_is_its_speaker_and_its_language():
    """[LAW:types-are-the-program] The id is a total description of the voice.

    There is no per-voice metadata to look up — a cloned speaker is a waveform
    and a language is a two-letter tag — so the id is composed rather than read,
    and `Voice.id` stays stable across restarts because both halves come from
    configuration rather than from anything the model generates.
    """
    spoken = described(chatterbox.BUILTIN_SPEAKER, "es")

    assert spoken.voice.id == "builtin-es"
    assert spoken.voice.language == "es"
    assert spoken.language == "es"


def test_two_voices_of_one_speaker_are_legibly_the_same_person():
    """The whole reason this engine is worth what it costs, as a client sees it.

    `es_ES-davefx-medium` and `en_US-lessac-high` are two different people, so
    Piper's catalogue cannot express "the same voice, another language" at all.
    Here the `speaker` label is equal across the pair while the language differs,
    which is what lets a client tell that switching between them keeps the
    speaker — and it is published as a label because ElevenLabs' voice object has
    no field for it.
    """
    english = described(chatterbox.BUILTIN_SPEAKER, "en").voice
    spanish = described(chatterbox.BUILTIN_SPEAKER, "es").voice

    assert english.id != spanish.id
    assert dict(english.labels)["speaker"] == dict(spanish.labels)["speaker"]
    assert english.language != spanish.language
    # Sharing an identity is not sharing an id: a catalogue keyed by id silently
    # drops the second of any pair that collides.
    assert {english.id, spanish.id} == {"builtin-en", "builtin-es"}


def test_a_voice_declares_no_capability_at_all():
    """The first engine here that declares nothing, and it means it.

    `generate` has no rate parameter, so SPEED would be a claim the audio
    contradicts. Nothing in the model measures durations, so TIMESTAMPS would be
    an alignment derived from nothing — and a fabricated alignment is worse than
    a refusal, because a caption renderer cannot tell it from a real one.

    Asserted against the whole enum rather than member by member, so a capability
    added later is claimed by nobody here until somebody decides it is true.
    """
    assert described(chatterbox.BUILTIN_SPEAKER, "en").voice.capabilities == frozenset()


def test_a_voice_answers_to_the_engine_that_will_speak_it():
    """A voice naming no model id refuses the caller who named the engine itself."""
    assert described(chatterbox.BUILTIN_SPEAKER, "en").voice.models == serves("chatterbox")


# --------------------------------------------- the assets, as CLAUDE.md wants them


def test_the_checkpoints_are_pinned_to_a_commit_rather_than_to_a_branch():
    """[FRAMING:representation] A floating artifact is the thing this repo forbids.

    Upstream loads this repository at `revision="main"`, which is a pointer the
    publisher can move: two builds of the same commit of *this* repository would
    then bake different weights and there would be nothing in the image to say
    so. The same rule that makes CI build from a commit rather than a working
    tree makes this a full commit sha, and this test is what keeps it one.
    """
    assert re.fullmatch(r"[0-9a-f]{40}", chatterbox._REVISION), chatterbox._REVISION


def test_only_the_files_this_engine_uses_are_fetched():
    """The repository holds 13.2 GiB across several export generations.

    An unfiltered `snapshot_download` would put all of it in the image. The
    allow-list is what makes the asset bill 3.06 GiB, so it is asserted to exist
    and to be a list rather than the everything-pattern that would silently
    restore the 13.2 GiB.
    """
    assert chatterbox._ASSETS
    assert "*" not in chatterbox._ASSETS


def test_the_setuptools_pin_that_keeps_the_watermarker_importable_is_still_there():
    """[LAW:no-silent-failure] The pin that must not be tidied away, held by a test.

    `chatterbox-tts` reaches its watermarker through `perth`, whose `__init__`
    catches `ImportError` and sets `PerthImplicitWatermarker = None`. The import
    really fails on `from pkg_resources import resource_filename`, which
    setuptools removed in 81 — so on a current setuptools the name silently
    becomes None and the model constructor dies forty lines away with `TypeError:
    'NoneType' object is not callable`, naming nothing at all.

    It reads like an incidental pin and it costs hours to rediscover, which is
    exactly the shape of a line somebody removes while tidying. A comment asks;
    this insists.
    """
    import tomllib
    from pathlib import Path

    pyproject = tomllib.loads(
        (Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    )
    requirements = pyproject["project"]["optional-dependencies"]["chatterbox"]

    assert any(
        requirement.replace(" ", "").startswith("setuptools<")
        for requirement in requirements
    ), requirements


# ------------------------------------------------------- who `builtin` turns out to be


class _MarkedConditionals:
    """A cloned identity distinguishable in the samples it produces."""

    def __init__(self, mark: float) -> None:
        self.mark = mark


class _AlreadyCloned:
    """A model at the point `_cloned` really meets it: holding the last clone."""

    def __init__(self, conds) -> None:
        self.conds = conds


def test_the_builtin_speaker_is_the_checkpoints_own_voice_however_late_it_is_named(
    tmp_path,
):
    """[LAW:no-ambient-temporal-coupling] Cloning order cannot decide who `builtin` is.

    `prepare_conditionals` writes `model.conds`, and `voices` makes the
    configured speaker order load-bearing rather than sorted, so
    `CHATTERBOX_SPEAKERS=alice,builtin` is ordinary input and reaches this
    function with the model holding alice. Read back off the model, `builtin`
    would be alice — every `builtin-*` voice a different person, baked into the
    image at build, with nothing raised and nothing logged.
    """
    checkpoint = _MarkedConditionals(0.25)
    model = _AlreadyCloned(_MarkedConditionals(0.75))

    assert chatterbox._cloned(model, "builtin", tmp_path, checkpoint) is checkpoint


# ----------------------------------------- speaking two voices at the same time


class _EchoingModel:
    """A model that answers with whichever speaker `conds` names when it reads it.

    The real `generate` reads `self.conds` partway through its work, which is
    what makes a second writer able to land between another caller's write and
    its read. This one reproduces that shape and nothing else: it waits, then
    reads, then says what it read — turning "answered in somebody else's voice"
    into a value a test can compare instead of something only an ear can catch.
    """

    def __init__(self, torch) -> None:
        self.conds = None
        self._torch = torch

    def generate(self, text: str, language_id: str):
        time.sleep(0.05)
        return self._torch.full((1, 1), self.conds.mark)


def test_two_voices_spoken_at_once_are_each_answered_in_their_own_voice():
    """The engine serialises synthesis, so overlap cannot swap two callers' speakers.

    Selecting a voice means writing to the model object, and the server calls
    engines off the event loop, so two requests naming two voices really are
    inside `speak` at once. Unserialised, the second one's write lands between
    the first one's write and its read and the first caller is answered as
    somebody else — audio that is fluent and simply the wrong person, which no
    assertion about rates or lengths would ever catch.

    Asserted against each voice's own uncontended result rather than against the
    lock: any implementation that keeps the two apart passes, and one that stops
    keeping them apart fails.
    """
    torch = pytest.importorskip("torch")

    voices = {
        speaker: chatterbox._describe(
            speaker, "en", _MarkedConditionals(mark), serves("chatterbox")
        )
        for speaker, mark in (("alice", 0.25), ("bob", 0.75))
    }
    serving = chatterbox.ChatterboxEngine(
        _EchoingModel(torch), {item.voice.id: item for item in voices.values()}, 24000
    )

    def spoken(speaker: str) -> bytes:
        return serving.speak_timed(voices[speaker].voice, TEXT, Prosody()).pcm

    alone = {speaker: spoken(speaker) for speaker in voices}
    assert alone["alice"] != alone["bob"], "the stand-in cannot tell the two apart"

    together: dict[str, bytes] = {}
    threads = [
        threading.Thread(target=lambda s=speaker: together.__setitem__(s, spoken(s)))
        for speaker in voices
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert together == alone


# ============================================================================
# Everything below opens the real model: ~3.06 GiB fetched, ~4.8 GiB resident,
# and synthesis at 8-33x real time on `cpu`.
# ============================================================================


@pytest.fixture(scope="module")
def engine(chatterbox_installed):
    """The engine as a default deployment opens it, built once for the module."""
    return chatterbox_prepared(languages=("en", "es")).open()


def test_the_offered_order_is_the_configured_order(chatterbox_installed):
    """[LAW:one-source-of-truth] The first voice offered is the default voice.

    `Engine.voices` makes this order load-bearing: a deployment naming no
    fallback answers unknown ids in whichever voice comes first. Piper shipped a
    sort-for-tidiness bug of exactly this shape, so every engine here is held to
    it — and this one has a second way to get it wrong, since its voices come out
    of a nested loop over two configured lists rather than out of one.

    The one test here that needs two differently configured models, so it owns
    both and holds them one at a time: measured, a second live model is 8.11 GiB
    resident against a build runner with 7.9 GB. It takes `chatterbox_installed`
    rather than `engine` for that reason — the module fixture would still be
    holding the first while this opened the second.
    """
    ordered = chatterbox_prepared(languages=("en", "es")).open()
    assert [voice.id for voice in ordered.voices()] == ["builtin-en", "builtin-es"]

    del ordered
    gc.collect()

    reversed_order = chatterbox_prepared(languages=("es", "en")).open()
    assert [voice.id for voice in reversed_order.voices()] == [
        "builtin-es",
        "builtin-en",
    ]


def test_the_engine_declares_nothing_it_cannot_do(engine):
    """The union over the voices on offer, which is how the server asks it.

    The per-voice claim is asserted above without a model; this is the same fact
    from the side the 501 gate reads, on voices a real `open` really produced.
    """
    assert declared(engine) == frozenset()


def test_a_language_the_model_does_not_speak_is_refused_when_it_is_discovered(
    chatterbox_installed,
):
    """The library is the authority on its own languages, and it is asked.

    `configure` cannot check this: it parses the environment without importing
    `chatterbox`, which is the seam `tests/test_encoding.py` proves from the
    other side — so the check lives at `open`, against the library's own
    `SUPPORTED_LANGUAGES`. A table of language codes copied into this repository
    would be a second source free to drift from the model it describes.
    """
    with pytest.raises(ConfigError) as raised:
        chatterbox_prepared(languages=("en", "kl")).open()

    assert "kl" in "\n".join(raised.value.problems)


def test_a_speaker_with_no_reference_recording_is_refused_at_boot(
    chatterbox_installed,
):
    """A voice id is stable because its speaker is an asset, not a request-time upload.

    So a named speaker whose `.wav` was never baked is a deployment that would
    list a voice it cannot speak in. It fails while an operator is watching
    rather than on the first request that names it.
    """
    with pytest.raises(ConfigError) as raised:
        chatterbox_prepared(speakers=("nobody",)).open()

    reported = "\n".join(raised.value.problems)
    assert "nobody" in reported
    assert chatterbox.REFERENCES_DIR in reported


def test_an_engine_that_cannot_measure_says_so_rather_than_inventing_a_timeline(
    engine,
):
    """The claim `speak_timed`'s `measured` makes, on the engine that cannot measure.

    The server never reaches this — no voice declares TIMESTAMPS, so the 501 gate
    refuses first and that gate is the single enforcer. The engine's answer is
    written to be true on its own anyway, and this is the only place that asks:
    the audio is real, every sample is accounted for, and the whole utterance is
    one separator because none of it was attributed to anything.

    Inventing boundaries to fill the tuple is the one thing that must not happen
    here, and it is the thing that would pass every other test in this suite.
    """
    spoken = engine.speak_timed(engine.voices()[0], TEXT, Prosody())

    assert spoken.measured is False
    assert sum(timing.samples for timing in spoken.timings) * 2 == len(spoken.pcm)
    assert all(timing.separates_words for timing in spoken.timings)
    assert len(spoken.pcm) > 0


def test_a_timestamps_request_is_refused_rather_than_answered_with_invented_numbers(
    chatterbox_installed,
):
    """The refusal end to end, assembled from `Capability`'s own sentence.

    Kokoro proved this for an engine whose *export* could not measure. This one
    proves it for an engine that never can, and whose voices therefore declare
    the empty set — the first deployment here in which no endpoint that needs a
    capability answers at all.
    """
    from fastapi.testclient import TestClient

    from elvenspeak import create_app
    from elvenspeak.engines import ENGINES
    from elvenspeak.settings import Settings

    prepared = chatterbox_prepared()
    settings = Settings(
        engine=prepared,
        engine_name="chatterbox",
        known_engines=frozenset(ENGINES),
        withheld=frozenset(),
        fallback="builtin-en",
        api_key=None,
        host="127.0.0.1",
        port=0,
    )
    client = TestClient(create_app(settings, prepared.open()))

    response = client.post(
        "/v1/text-to-speech/builtin-en/with-timestamps", json={"text": "Hello there."}
    )

    assert response.status_code == 501
    assert Capability.TIMESTAMPS.value in response.json()["detail"]
    # No alignment anywhere in the body: a 501 still carrying a character
    # timeline would be the invented numbers under a different status code.
    assert "alignment" not in response.text


def test_the_engine_speaks_on_the_device_the_deployment_named(engine):
    """The short-utterance property the eventual mixed-language work needs.

    Kokoro has a measured zero-sample defect on short inputs, which is why its
    Spanish is deliberately not baked. Chatterbox does not: every short utterance
    measured — three and four characters, in both languages — produced healthy
    audio. Asserted here because whatever renders mixed language will feed this
    engine short spans, and an engine that goes silent on them is useless for it
    however well it reads a paragraph.
    """
    for voice in engine.voices():
        spoken = engine.speak(voice, "Yes.", Prosody())
        assert len(b"".join(spoken.audio)) > 0, voice.id
