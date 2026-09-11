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

Nothing here fetches the checkpoints or loads the model. That used to be a
divider halfway down this file, with ~3.06 GiB fetched once and ~4.69 GiB
resident below it, and the whole of that is now gone: the questions those tests
asked are answered either from configuration, or from [`_Model`] — the two things
this engine reads off `ChatterboxMultilingualTTS`, which are a `conds` slot it
writes and a `generate` it calls.

Two of them are not, and they went to `speaks.py` rather than being weakened
here. That a real utterance comes back as audio and not as silence is a claim
about the model, and a stand-in agrees with it by construction — which is the
same as not asking — so it is asked of every voice the published image offers,
where the answer means something.

The second was split rather than moved, and the split is the honest part.
`speaks.py` puts two callers inside the real engine at once and establishes that
each is answered in full — no refusal, no deadlock, no truncated or crossed body.
It cannot establish whose voice answered: a swapped identity comes back fluent
and the right length, and separating it from a correct answer over PCM would take
a speaker embedding. So that half stays here, where the conditionals are objects
a stand-in can tell apart and each result is compared against the same voice's
uncontended one. Neither file holds the whole claim; between them none of it was
dropped.

One import survives, and it is stated rather than skipped past: the concurrency
test imports torch, because the code it exercises builds its samples through
`torch.int16`. On an install without the chatterbox extra it errors. That is the
intended outcome — it is the only test proving the lock that keeps two callers
from being answered in each other's voice at the level of this engine's own code,
and a skip is indistinguishable from a pass in a summary.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field

import pytest
from conftest import (
    SERVES,
    chatterbox_prepared,
    declared,
    serves,
)

from elvenspeak import chatterbox
from elvenspeak.engine import Prosody, Timing
from elvenspeak.provisioning import ConfigError

TEXT = "Compatibility is measurable, and this sentence is long enough to measure."

#: The rate [`_Model`] speaks at, and the one every sample count below is
#: arithmetic against. Not the library's `S3GEN_SR`: a stand-in answering with
#: the real constant would let a `sample_rate` read off the library instead of
#: off the opened model agree with these assertions by coincidence.
RATE = 8192


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
    boot on Apple hardware, and `cpu` boots everywhere and then serves at 8-33
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


def test_the_upper_bound_that_pins_chatterbox_to_the_internals_we_read_is_there():
    """[LAW:one-source-of-truth] A bound on internals, held where it can be checked.

    This module reads `chatterbox-tts` internals no release note covers:
    `_cloned`'s ordering fix holds only because `prepare_conditionals` rebinds
    `self.conds` rather than writing into it, `_synthesized`'s guard depends on
    `generate` mutating `conds.t3` only when `exaggeration` is passed, and
    `_ASSETS` names this release's exact checkpoint filenames.

    A release that changed any of them would import cleanly and pass every test
    in this file — the only symptom is each builtin-* voice becoming the wrong
    person, which no test here can observe. So the lock is not allowed to walk
    across a minor on its own: `uv lock --upgrade` has to stop and ask.
    """
    import tomllib
    from pathlib import Path

    pyproject = tomllib.loads(
        (Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    )
    requirements = pyproject["project"]["optional-dependencies"]["chatterbox"]

    assert any(
        "chatterbox-tts" in requirement.replace(" ", "")
        and "<" in requirement.replace(" ", "")
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

    assert (
        chatterbox._cloned(model, chatterbox.BUILTIN_SPEAKER, tmp_path, checkpoint)
        is checkpoint
    )


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
    import torch

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


# ------------------------------------------- not holding the weights twice


class _Loading:
    """A class with a `load_state_dict`, which is all the release wraps.

    Standing in for `T3` because that is the entire contract: something copies a
    dict and something else has to be able to say when the copy is finished.
    Recording what arrived is how the test can tell "released after the copy"
    from "released instead of it" — a wrapper that emptied the dict first would
    leave the model silently unweighted and every assertion about memory green.
    """

    def load_state_dict(self, state_dict):
        self.copied = dict(state_dict)
        return "what the original returned"


def test_the_weights_are_released_once_they_have_been_copied():
    """The 2 GiB the load used to carry past its last use is dropped at the copy.

    Asserted on the caller's own dict, because that is what upstream holds: the
    name `from_local` binds stays in scope until the function returns, so
    emptying the object it names is the only thing that frees the tensors from
    outside. Both halves are the point — the copy is complete, and the original
    is gone.
    """
    weights = {"speech_head.weight": ["2044.7 MiB of it"]}
    loading = _Loading()

    with chatterbox._releasing_state_dict(_Loading):
        returned = loading.load_state_dict(weights)

    assert returned == "what the original returned"
    assert loading.copied == {"speech_head.weight": ["2044.7 MiB of it"]}
    assert weights == {}


def test_the_class_is_left_as_it_was_found_even_when_the_load_raises():
    """[LAW:no-shared-mutable-globals] The window closes on the path nobody plans for.

    `T3` is shared by everything that imports it, and a load that raises is the
    likeliest way this repository ever exercises the failure path — a truncated
    checkpoint, a revision that moved. Leaking the wrapper past that would leave
    every later `load_state_dict` in the process silently dropping its argument.
    """
    original = _Loading.load_state_dict

    with pytest.raises(ZeroDivisionError):
        with chatterbox._releasing_state_dict(_Loading):
            1 / 0

    assert _Loading.load_state_dict is original


# ============================================================================
# Where the divider used to be. Below it the checkpoints were on disk and the
# model was loaded; what that cost, and where the two claims that really needed
# it went instead, is stated in the module docstring rather than copied here — a
# second copy of a measured figure is free to drift from the table it came from,
# and this one had.
# ============================================================================


@dataclass
class _Model:
    """`ChatterboxMultilingualTTS`, reduced to the two things this engine uses.

    `conds` is a slot rather than a value, and that is the whole reason this
    engine has a lock: `generate` reads the speaker off it, so selecting a voice
    means writing to the model object. A stand-in that took the speaker as an
    argument would be a friendlier library than the real one and would make the
    lock look like a precaution.

    [LAW:no-shared-mutable-globals] `asked` records each call with the `conds`
    that were in the slot when it ran, which is what lets a test ask whether a
    voice was answered in its own speaker rather than in whoever was written last.
    """

    #: How many samples every synthesis answers with, as silence: nothing here
    #: listens, and `test_encoding.py` is where sample values mean anything.
    samples: int = 0
    conds: object = "the checkpoints' own identity"
    asked: list[dict] = field(default_factory=list)

    def generate(self, text, language_id):
        import torch

        self.asked.append(
            {"text": text, "language_id": language_id, "conds": self.conds}
        )
        return torch.zeros(1, self.samples)


def speaking(model: _Model, *voices: tuple[str, str]) -> chatterbox.ChatterboxEngine:
    """`model` as the engine a deployment naming these `(speaker, language)` pairs
    would be serving.

    Every voice of one speaker shares that speaker's identity, which is the whole
    economy of offering 23 languages for the price of one model — so the
    conditionals here are per speaker too, and a test can tell "answered in the
    wrong language" apart from "answered in the wrong person".
    """
    conditionals = {speaker: f"the identity of {speaker}" for speaker, _ in voices}
    spoken = {}
    for speaker, language in voices:
        item = chatterbox._describe(speaker, language, conditionals[speaker], SERVES)
        spoken[item.voice.id] = item
    return chatterbox.ChatterboxEngine(model, spoken, sample_rate=RATE)


@pytest.fixture
def loads(tmp_path, monkeypatch):
    """`open` with the fetch answered by a snapshot and the load by [`_Model`].

    Everything between those two runs for real — the language table is consulted,
    the reference recordings are stat'd, `conds.pt` is looked for, the speaker and
    language loops build the catalogue — which is where every property below
    lives. What is skipped is the ~3.06 GiB download and the ~4.69 GiB load of
    weights that no assertion here reads a single number out of.

    Yields the models it loaded, because "how many were loaded" is the shape of
    the memory ceiling this suite used to be killed by: two live models measured
    8.11 GiB against a build runner with 7.9 GB.
    """
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    (tmp_path / chatterbox._BUILTIN_CONDITIONALS).write_bytes(b"")
    loaded: list[_Model] = []

    def _from_local(_checkpoints, _device):
        loaded.append(_Model())
        return loaded[-1]

    monkeypatch.setattr(chatterbox, "_fetch", lambda models_dir, allow: tmp_path)
    monkeypatch.setattr(ChatterboxMultilingualTTS, "from_local", _from_local)
    return loaded


def test_the_offered_order_is_the_configured_order(loads):
    """[LAW:one-source-of-truth] The first voice offered is the default voice.

    `Engine.voices` makes this order load-bearing: a deployment naming no
    fallback answers unknown ids in whichever voice comes first. Piper shipped a
    sort-for-tidiness bug of exactly this shape, so every engine here is held to
    it — and this one has a second way to get it wrong, since its voices come out
    of a nested loop over two configured lists rather than out of one.

    Two differently configured deployments, which used to be the most expensive
    test in this repository: it held two live models one at a time and set the
    suite's high-water mark near 6.6 GiB, and it is the test every killed run
    died inside. The order comes out of the loops in `_open` and never out of the
    weights, so both of those models were paid for to read a dictionary back.
    """
    ordered = chatterbox_prepared(languages=("en", "es")).open()
    assert [voice.id for voice in ordered.voices()] == ["builtin-en", "builtin-es"]

    reversed_order = chatterbox_prepared(languages=("es", "en")).open()
    assert [voice.id for voice in reversed_order.voices()] == [
        "builtin-es",
        "builtin-en",
    ]


def test_the_engine_declares_nothing_it_cannot_do(loads):
    """The union over the voices on offer, which is how the server asks it.

    The per-voice claim is asserted above from a description alone; this is the
    same fact from the side the 501 gate reads, on voices a real `open` really
    assembled — which is the half that would survive `_describe` staying honest
    while `_open` stamped something onto the catalogue on its way past.
    """
    assert declared(chatterbox_prepared(languages=("en", "es")).open()) == frozenset()


@pytest.fixture
def unfetched(monkeypatch):
    """`_fetch` rigged to fail the test that reaches it.

    Everything a deployment can get wrong about its languages or its speakers is
    answerable from a static table and a stat on `models_dir`, so `open` answers
    all of it before it downloads anything. Ordered the other way the same typo
    costs ~3.06 GiB and several minutes before it is reported — and the two tests
    below would not notice, because a refusal that arrives late is still a
    refusal. This is what makes the ordering a property rather than an accident.

    It is also what lets those two tests need no checkpoints: with the fetch
    refused, they need the library and nothing it would have downloaded.
    """

    def unreached(models_dir, allow_download):
        pytest.fail("_fetch ran: a refusal answerable from configuration paid for it")

    monkeypatch.setattr(chatterbox, "_fetch", unreached)


def test_a_language_the_model_does_not_speak_is_refused_when_it_is_discovered(
    unfetched,
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


def test_a_speaker_with_no_reference_recording_is_refused_at_boot(unfetched):
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


def test_a_device_the_host_cannot_provide_is_refused_before_anything_is_fetched(
    unfetched, monkeypatch
):
    """The device is answerable from the host, so it is answered with the typos.

    `from_local` reaches the same verdict on its own, and reaches it in the two
    worst places: after ~3.06 GiB of download, and inside torch, where the words
    are `Torch not compiled with CUDA enabled` — true, and naming neither the
    variable that chose cuda nor the fact that a deployment chose it. So the
    probe is a round trip of one number, it happens beside the language and
    speaker checks, and it keeps torch's sentence while adding the two facts
    torch cannot know.

    Which device is missing is STATED here rather than inherited from the
    machine. A test that asked for the accelerator this host genuinely lacks
    would assert the opposite thing on a CUDA box and nothing at all on some
    third one — the shape that cost `kokoro_prepared` a review round on #53.

    `probed` pins the device as well as the refusal. A probe that always asked
    about `cpu` would satisfy every other assertion here while proving that a
    deployment's `cuda` was never looked at.
    """
    refused = "cuda"
    probed: list[str] = []

    def allocate(device: str) -> None:
        probed.append(device)
        if device == refused:
            raise AssertionError("Torch not compiled with CUDA enabled")

    monkeypatch.setattr(chatterbox, "_allocate", allocate)

    with pytest.raises(ConfigError) as raised:
        chatterbox_prepared(device=refused).open()

    reported = "\n".join(raised.value.problems)
    assert chatterbox.DEVICE in reported
    assert refused in reported
    assert "Torch not compiled with CUDA enabled" in reported
    assert probed == [refused]


def test_the_bake_is_refused_the_device_too_because_it_loads_the_model_onto_it(
    unfetched, monkeypatch
):
    """Where this engine parts company with kokoro's espeak refusal, deliberately.

    `kokoro._Prepared.open` refuses an unnamed espeak library and `acquire` does
    not, because the bake synthesizes nothing and must not be refused for a
    library it will never call. The device is not that: `acquire` loads the model
    through this same `_open`, on this same device, to prove 3.06 GiB of
    checkpoints are readable — so a bake naming a device the builder lacks is a
    bake that was going to fail at `from_local` regardless.

    Refusing it here is therefore no new requirement, only the same one made
    legible and made early. This test exists because the reverse edit reads like
    consistency: moving the probe into `open` alone would look like following the
    kokoro precedent, and would quietly hand the bake back its 3.06 GiB traceback.
    """

    def allocate(device: str) -> None:
        raise RuntimeError("MPS backend is not available")

    monkeypatch.setattr(chatterbox, "_allocate", allocate)

    with pytest.raises(ConfigError) as raised:
        chatterbox_prepared(device="mps").acquire()

    assert "MPS backend is not available" in "\n".join(raised.value.problems)


def test_checkpoints_with_no_builtin_voice_are_refused_before_the_model_loads(
    tmp_path, monkeypatch
):
    """[LAW:parse-dont-validate] The refusal costs a stat rather than 4.69 GiB.

    A snapshot without `conds.pt` carries no `builtin` identity, and this asks
    two things of that: that it is refused at all, and that it is refused at the
    same moment as a typo'd language or a missing reference `.wav` — before
    `from_local`. Read back off the loaded model instead, the same deployment
    would spend ~4.69 GiB and a minute of an operator's restart to be told what a
    directory listing already said.

    Needs no checkpoints: `_fetch` is answered with an empty directory
    and `from_local` is the thing that must not run, so the property is provable
    without the 3.06 GiB the fixture downloads. The library itself is needed, for
    the `SUPPORTED_LANGUAGES` the sibling checks read.
    """
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    def unreached(*_args, **_kwargs):
        pytest.fail("from_local ran: the refusal moved back behind the model load")

    monkeypatch.setattr(chatterbox, "_fetch", lambda models_dir, allow: tmp_path)
    monkeypatch.setattr(ChatterboxMultilingualTTS, "from_local", unreached)

    with pytest.raises(ConfigError) as raised:
        chatterbox_prepared(speakers=(chatterbox.BUILTIN_SPEAKER,)).open()

    reported = "\n".join(raised.value.problems)
    assert chatterbox._BUILTIN_CONDITIONALS in reported
    assert chatterbox.BUILTIN_SPEAKER in reported


def test_an_engine_that_cannot_measure_says_so_rather_than_inventing_a_timeline():
    """The claim `speak_timed`'s `measured` makes, on the engine that cannot measure.

    The server never reaches this — no voice declares TIMESTAMPS, so the 501 gate
    refuses first and that gate is the single enforcer. The engine's answer is
    written to be true on its own anyway, and this is the only place that asks:
    every sample is accounted for, and the whole utterance is one separator
    because none of it was attributed to anything.

    Inventing boundaries to fill the tuple is the one thing that must not happen
    here, and it is the thing that would pass every other test in this suite —
    which is why the tuple is asserted whole rather than summed. A timeline of
    four invented gaps sums exactly as well as one honest one.
    """
    engine = speaking(_Model(samples=RATE), ("builtin", "en"))

    spoken = engine.speak_timed(engine.voices()[0], TEXT, Prosody())

    assert spoken.measured is False
    assert spoken.timings == (Timing(samples=RATE, separates_words=True),)
    assert sum(timing.samples for timing in spoken.timings) * 2 == len(spoken.pcm)


def test_an_utterance_that_came_back_empty_is_a_timeline_of_nothing():
    """The other half of the same arithmetic, and the one with a division in it.

    `TimedSpeech` requires the durations to sum to the sample count, and a
    zero-sample answer meets that with no stretches at all — where a tuple
    carrying one stretch of zero samples would be a timeline claiming a gap that
    no audio is under. It is reachable: an engine that produced nothing is a state
    every other engine here has a named path for, and this one answers it in
    arithmetic rather than in a branch anybody wrote on purpose.
    """
    engine = speaking(_Model(samples=0), ("builtin", "en"))

    spoken = engine.speak_timed(engine.voices()[0], TEXT, Prosody())

    assert spoken.pcm == b""
    assert spoken.timings == ()
    assert spoken.measured is False


def test_each_voice_is_answered_in_its_own_speaker_and_its_own_language():
    """[LAW:single-enforcer] Both of the model's mutable slots, read at the call.

    `generate` reads the speaker off `model.conds`, so selecting a voice means
    writing to the model — and the language is a plain argument beside it. A voice
    reaching the library with the wrong one of either is answered fluently, in the
    wrong person or the wrong accent, with nothing raised anywhere. That is the
    failure `test_two_voices_spoken_at_once_are_each_answered_in_their_own_voice`
    proves cannot happen under concurrency; this is the same claim for one caller,
    which is the version that catches a `conds` never written at all.

    Two speakers and two languages rather than one of each, because the catalogue
    is their product: with a single speaker, an engine that wrote whichever
    identity it cloned last passes.
    """
    model = _Model(samples=RATE)
    engine = speaking(
        model, ("builtin", "en"), ("builtin", "es"), ("nobody", "en"), ("nobody", "es")
    )

    for voice in engine.voices():
        b"".join(engine.speak(voice, TEXT, Prosody()).audio)

    assert [(call["conds"], call["language_id"]) for call in model.asked] == [
        ("the identity of builtin", "en"),
        ("the identity of builtin", "es"),
        ("the identity of nobody", "en"),
        ("the identity of nobody", "es"),
    ]
