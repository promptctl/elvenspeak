"""Startup, and the failures it is supposed to turn into a refusal to boot.

`Prepared.open()` exists to make a broken deployment one clean failure before the
first request, rather than an unbounded delay or a 500 inside somebody's
synthesis. Its failure paths had no coverage at all, which is the wrong way
round: they are the whole reason opening the engine is separate from serving.

Also this engine's own configuration, which moved here from
`tests/test_settings.py` with the parsing it covers. `PIPER_MODELS_DIR` describes
a directory of ONNX files; the module that understands what that means is the one
that should be answering for what a bad one does.

No real Piper model is needed. `_describe` reads only the `.onnx.json` sidecar
beside the weights and never opens the ONNX file itself, and the session that
opening builds from the weights is stubbed here — the ONNX runtime is not what
these tests are about.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from conftest import SERVES, declared, make_voice, piper_prepared, serves

from elvenspeak import piper
from elvenspeak.engine import Capability, Prosody
from elvenspeak.provisioning import ConfigError

KEY = "en_US-lessac-medium"


class _StubVoice:
    """A loaded model that produces no audio.

    Enough for every test here: what is under test is which files opening demands
    and what it reads out of them, not what the ONNX graph does with them.
    """

    @staticmethod
    def load(path: str, include_alignments: bool = False) -> "_StubVoice":
        return _StubVoice()

    def synthesize(self, text: str, **kwargs):
        return iter(())


@pytest.fixture(autouse=True)
def stub_sessions(monkeypatch):
    """Stands in for the ONNX loader, so a placeholder `.onnx` is loadable.

    The one symbol is replaced on the real module rather than the module being
    replaced wholesale. A stub object in `sys.modules["piper"]` has no
    `__path__`, so `from piper.download_voices import download_voice` — which
    `_install` runs before anything else — could only resolve if some earlier test
    had already cached that submodule. It did: `test_api.py` runs first and
    imports the real one, so this file passed in a full run and failed 9 of 11
    on its own, which is also how it would fail on a machine with no baked model.
    """
    monkeypatch.setattr("piper.PiperVoice", _StubVoice)


def _write_sidecar(models_dir: Path, sidecar: dict) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / f"{KEY}.onnx").write_bytes(b"not a real model")
    (models_dir / f"{KEY}.onnx.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )


def load(
    models_dir: Path, allow_download: bool = False, timings: bool = False
) -> piper.PiperEngine:
    return piper_prepared(
        models_dir, allow_download=allow_download, timings=timings
    ).open()


def test_an_installed_voice_becomes_an_engine(tmp_path):
    make_voice(tmp_path)
    engine = load(tmp_path)
    assert [v.id for v in engine.voices()] == [KEY]
    # Read from the sidecar rather than assumed, because it is the rate the
    # samples really have and the encoder is told it. Reported with the audio
    # rather than on the voice, which is where a caller can act on it.
    spoken = engine.speak(engine.voices()[0], "hello", Prosody())
    assert spoken.sample_rate == 22050


def test_voices_come_back_in_the_order_the_operator_named_them(tmp_path):
    """Configured order, not sorted — the first one offered is load-bearing.

    A deployment that names no fallback answers unknown ids in whichever voice
    the engine lists first, so sorting here silently overrides the operator's
    stated preference: `PIPER_VOICES=en_US-lessac-medium,en_GB-alba-medium` used
    to default to lessac and, sorted, would hand every substituted request to
    alba instead. The two voices below are chosen so the configured order and
    the alphabetical one disagree; with any two that happen to agree, this test
    passes either way and says nothing.
    """
    make_voice(tmp_path, key="en_US-lessac-medium")
    make_voice(tmp_path, key="en_GB-alba-medium")

    engine = piper_prepared(
        tmp_path, voices=("en_US-lessac-medium", "en_GB-alba-medium")
    ).open()

    assert [v.id for v in engine.voices()] == [
        "en_US-lessac-medium",
        "en_GB-alba-medium",
    ]


def test_a_missing_voice_refuses_to_boot_when_downloading_is_off(tmp_path):
    """[LAW:no-silent-failure] Serving zero voices is not a degraded success."""
    with pytest.raises(FileNotFoundError, match="not completely installed"):
        load(tmp_path)


def test_weights_without_their_config_are_not_installed(tmp_path):
    """Both halves, because an interrupted download leaves one without the other.

    Checking only the `.onnx` treats a half-copied voice as installed and defers
    the failure to `_describe` or to the first synthesis — later, and further from
    the cause, than the boot this check exists to fail.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{KEY}.onnx").write_bytes(b"not a real model")
    with pytest.raises(FileNotFoundError, match="not completely installed"):
        load(tmp_path)


def test_a_minimal_config_derives_its_metadata_from_the_key(tmp_path):
    """The fallback branch: what a sidecar omits is read back out of the key.

    Every other fixture here writes a complete sidecar, so this derivation and
    the three-part key check gating it were never exercised.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{KEY}.onnx").write_bytes(b"not a real model")
    (tmp_path / f"{KEY}.onnx.json").write_text(
        json.dumps({"audio": {"sample_rate": 16000}}), encoding="utf-8"
    )
    voice = load(tmp_path).voices()[0]
    labels = dict(voice.labels)
    assert voice.name == "lessac"
    # Both halves of the key-derived language: the display code the description
    # carries, and the ISO family a caller's `language_code` is matched on. A
    # sidecar this bare exercises the fallback for both, and they are derived one
    # from the other — asserting only the family would pass on a key split that
    # dropped the region entirely.
    assert "en_US" in voice.description
    assert voice.language == "en"
    assert labels["quality"] == "medium"


def test_an_explicitly_null_section_reads_as_an_absent_one(tmp_path):
    """`.get(key, {})` substitutes only for a missing key, not for a null value.

    A hand-edited or half-written sidecar carrying `"language": null` made the
    chained lookup raise AttributeError instead of falling back to the key —
    which is what the surrounding expression already promises to do.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{KEY}.onnx").write_bytes(b"not a real model")
    (tmp_path / f"{KEY}.onnx.json").write_text(
        json.dumps({"language": None, "audio": {"sample_rate": 22050}, "dataset": None}),
        encoding="utf-8",
    )
    voice = load(tmp_path).voices()[0]
    assert "en_US" in voice.description
    assert voice.language == "en"
    assert voice.name == "lessac"


def test_the_sidecars_own_family_is_what_a_voice_speaks(tmp_path):
    """The primary source for `Voice.language`, which nothing else here reaches.

    Every other fixture states only `language.code`, so every other test exercises
    the `code.split("_")` fallback — and an implementation that ignored the
    sidecar's `family` outright would pass all of them. Every real downloaded
    voice takes this branch instead, which makes it the one that decides the
    language of everything a deployment actually bakes.

    The fixture makes `family` **disagree** with the split, which is what gives
    the assertion teeth: `pt_BR` would derive `pt`, so a `family` of `es` can only
    come from the sidecar. No real voice disagrees this way — the point is that
    the test cannot pass by accident.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{KEY}.onnx").write_bytes(b"not a real model")
    (tmp_path / f"{KEY}.onnx.json").write_text(
        json.dumps(
            {
                "language": {"code": "pt_BR", "family": "es"},
                "audio": {"sample_rate": 22050},
                "dataset": "lessac",
            }
        ),
        encoding="utf-8",
    )

    voice = load(tmp_path).voices()[0]

    assert voice.language == "es"
    # The display code stays the sidecar's own, so the two are visibly separate
    # facts rather than one derived twice.
    assert "pt_BR" in voice.description


def test_a_voice_that_states_no_language_at_all_is_refused(tmp_path):
    """[LAW:no-silent-failure] The second field with no honest default.

    A sidecar stating neither `family` nor `code`, under a key that does not parse
    as `<family>_<REGION>-<name>-<quality>` — an operator-chosen `PIPER_VOICES` id
    beside a hand-written sidecar, which is the only way to reach it. The language
    was then the whole key, a code no caller's `language_code` can equal, so the
    voice was silently unreachable by language while listed in `GET /v1/models`'
    `languages` as though it spoke one.

    Refused at load like a missing `sample_rate` and for the same reason: the
    alternative is a value that looks like an answer and is not. Named in the
    message, so an operator learns which of their voices it was.
    """
    key = "customvoice"
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{key}.onnx").write_bytes(b"not a real model")
    (tmp_path / f"{key}.onnx.json").write_text(
        json.dumps({"audio": {"sample_rate": 22050}}), encoding="utf-8"
    )

    with pytest.raises(ValueError) as raised:
        piper_prepared(tmp_path, voices=(key,)).open().voices()

    assert key in str(raised.value)
    assert "language" in str(raised.value)


def test_engine_facts_with_no_elevenlabs_field_travel_as_labels(tmp_path):
    """The open map is how a voice keeps what the schema has no room for.

    `speakers` in particular: there is no ElevenLabs field to select a speaker
    with, so a multi-speaker model always speaks as its default, and a caller is
    better off reading that than discovering it by listening.
    """
    make_voice(tmp_path)
    (tmp_path / f"{KEY}.onnx.json").write_text(
        json.dumps({"audio": {"sample_rate": 22050}, "num_speakers": 4}),
        encoding="utf-8",
    )
    labels = dict(load(tmp_path).voices()[0].labels)
    assert labels["speakers"] == "4"
    assert labels["engine"] == "piper"


def test_a_missing_voice_is_downloaded_when_downloading_is_allowed(tmp_path, monkeypatch):
    """The branch that actually fetches, which nothing entered before."""
    calls = []

    def fake_download(key, directory):
        calls.append((key, Path(directory)))
        make_voice(Path(directory), key)

    monkeypatch.setitem(
        sys.modules,
        "piper.download_voices",
        type("M", (), {"download_voice": staticmethod(fake_download)}),
    )
    engine = load(tmp_path, allow_download=True)
    assert calls == [(KEY, tmp_path)]
    assert [v.id for v in engine.voices()] == [KEY]


def test_a_download_that_produces_nothing_refuses_to_boot(tmp_path, monkeypatch):
    """[LAW:no-silent-failure] Returning is not the same as having delivered.

    `download_voice` reports success by returning, and a half-written pair is the
    same realistic outcome the pre-check exists for. Without a check after the
    call it surfaced as a bare FileNotFoundError from reading the sidecar, which
    names the missing file but not the download that failed to produce it.
    """
    def writes_nothing(key, directory):
        return None

    monkeypatch.setitem(
        sys.modules,
        "piper.download_voices",
        type("M", (), {"download_voice": staticmethod(writes_nothing)}),
    )
    with pytest.raises(FileNotFoundError, match="did not produce"):
        load(tmp_path, allow_download=True)


@pytest.mark.parametrize("rate", [0, -1, None])
def test_a_voice_with_no_usable_sample_rate_refuses_to_boot(tmp_path, rate):
    """Falsy counts as missing, because a zero is worse than an absent key.

    `sample_rate=0` passed an `is None` check and was stored, then failed as a
    ZeroDivisionError inside the alignment's seconds-per-sample — at request
    time, on the timestamp endpoints, far from the sidecar that caused it. There
    is no safe default either: a guessed rate plays perfectly at the wrong pitch.
    """
    _write_sidecar(tmp_path, {"audio": {"sample_rate": rate}})
    with pytest.raises(ValueError, match="sample_rate"):
        load(tmp_path)


@pytest.mark.parametrize(
    "sidecar", ['{"audio": {"sample_rate": 0}}', "{not json at all", "{}"]
)
def test_a_sidecar_that_cannot_be_read_fails_the_bake(tmp_path, sidecar):
    """The image build is the last moment this failure is cheap.

    `_install`, reached through `Prepared.acquire`, is what the image build runs,
    and a version of it that checked only
    that both files existed let a truncated or malformed `.onnx.json` bake into a
    green image that then failed at every container start — the interrupted-write
    case the existence checks are already there for, surviving them because the
    second file was present but unreadable.

    Asserted against `acquire` rather than `open`, because opening inherits this
    and the image build only ever reaches the former.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{KEY}.onnx").write_bytes(b"not a real model")
    (tmp_path / f"{KEY}.onnx.json").write_text(sidecar, encoding="utf-8")
    with pytest.raises((ValueError, KeyError)):
        piper_prepared(tmp_path).acquire()


@pytest.mark.parametrize("timings", [True, False])
def test_the_engine_declares_the_flag_its_sessions_were_opened_under(tmp_path, timings):
    """[LAW:one-source-of-truth] Every answer the server gives reads this.

    `include_alignments` patches the graph at load time, so whether a session can
    report durations is decided once, here, and is not recoverable from anything
    else. The API used to read `settings.timestamps` instead — the same fact held
    in two places, agreeing only for as long as every caller passed one value to
    both.
    """
    make_voice(tmp_path)
    loaded = load(tmp_path, timings=timings)
    assert (Capability.TIMESTAMPS in declared(loaded)) is timings


def test_piper_always_declares_speed_whatever_timings_was(tmp_path):
    """`length_scale` is on every voice, so the rate is never not variable.

    Pinned because the capability set is assembled in one expression beside the
    timings flag, and the shape that fails is one where every capability ends up
    riding on that flag — which nothing else here would catch, since a request
    with no `speed` in it behaves identically either way.
    """
    make_voice(tmp_path)
    assert Capability.SPEED in declared(load(tmp_path))


def test_acquiring_does_not_open_any_session(tmp_path, monkeypatch):
    """The joint the Dockerfile found: the image bake wants files, not sessions.

    Opening an ONNX session per voice costs a minute and a gigabyte, and the
    build throws every one of them away. Asserted rather than described, because
    a later edit that folds the loader into `acquire` would cost that on every
    image build and break nothing visible.
    """
    make_voice(tmp_path)
    monkeypatch.setattr(
        "piper.PiperVoice",
        type("Never", (), {"load": staticmethod(lambda *a, **k: pytest.fail(
            "acquire opened an ONNX session"
        ))}),
    )
    voices = piper_prepared(tmp_path).acquire()
    # Described, not merely located: a path would prove nothing, which is what
    # let a malformed sidecar bake into a green image. A `Voice` cannot exist
    # without the sidecar having been read.
    assert [voice.id for voice in voices] == [KEY]
    assert dict(voices[0].labels)["engine"] == "piper"


def test_acquiring_fetches_whatever_the_deployment_said_about_downloading(
    tmp_path, monkeypatch
):
    """[LAW:one-source-of-truth] The build fetches; there is no flag saying so.

    `PIPER_ALLOW_DOWNLOAD` is off in the image, and the bake step used to
    override it to `True` from a constant beside its call — the same rule stated
    twice, in opposite directions, correct only while one caller remembered to
    contradict the other. The lifecycle moment is which method is called now, so
    this passes with the flag off, which is the configuration a real image bake
    runs under.
    """
    downloaded = []

    def fake_download(key, directory):
        downloaded.append(key)
        make_voice(Path(directory), key)

    monkeypatch.setitem(
        sys.modules,
        "piper.download_voices",
        type("M", (), {"download_voice": staticmethod(fake_download)}),
    )
    voices = piper_prepared(tmp_path, allow_download=False).acquire()
    assert downloaded == [KEY]
    assert [voice.id for voice in voices] == [KEY]


def test_opening_still_refuses_to_fetch_when_the_deployment_said_not_to(
    tmp_path, monkeypatch
):
    """The other half of the same distinction, and the one the image relies on.

    A container that reaches Hugging Face at boot is a container that stops
    booting when Hugging Face does. `acquire` fetching unconditionally must not
    have quietly made `open` do it too.
    """
    monkeypatch.setitem(
        sys.modules,
        "piper.download_voices",
        type("M", (), {"download_voice": staticmethod(
            lambda *a: pytest.fail("open fetched a voice")
        )}),
    )
    with pytest.raises(FileNotFoundError, match="not completely installed"):
        load(tmp_path, allow_download=False)


def test_a_voice_list_that_names_nothing_is_refused():
    with pytest.raises(ConfigError, match="PIPER_VOICES is empty"):
        piper.configure({"PIPER_VOICES": "  ,  "}, frozenset(), SERVES)


def test_voices_are_split_and_stripped():
    prepared = piper.configure(
        {"PIPER_VOICES": "a-b-c , d-e-f,  g-h-i "}, frozenset(), SERVES
    )
    assert prepared.keys == ("a-b-c", "d-e-f", "g-h-i")


def test_an_empty_models_dir_is_refused_not_taken_as_the_working_directory():
    """The one setting here that was not validated.

    `PIPER_MODELS_DIR=` is a present key, so `get` returns "" rather than the
    default, `Path("")` is `Path(".")`, and `mkdir` on it succeeds. Nothing
    fails — the server just reads and writes 60 MB models into whatever
    directory it was launched from, which is the silent wrong thing rather than
    the clean refusal a startup produces for every other misconfiguration.
    """
    with pytest.raises(ConfigError, match="PIPER_MODELS_DIR"):
        piper.configure({"PIPER_MODELS_DIR": "   "}, frozenset(), SERVES)


def test_an_absent_models_dir_still_gets_the_default():
    """Unset is not the same as set-to-empty, and only one of them is a problem."""
    assert piper.configure({}, frozenset(), SERVES).models_dir.name == "models"


@pytest.mark.parametrize("value,expected", [
    ("1", True), ("true", True), ("YES", True), ("on", True),
    ("0", False), ("false", False), ("No", False), ("off", False),
])
def test_flags_accept_both_spellings(value, expected):
    prepared = piper.configure({"PIPER_ALLOW_DOWNLOAD": value}, frozenset(), SERVES)
    assert prepared.allow_download is expected


@pytest.mark.parametrize("typo", ["tru", "enabled", "y", ""])
def test_a_boolean_typo_is_a_config_error_not_a_silent_false(typo):
    """[LAW:no-silent-failure] `PIPER_ALLOW_DOWNLOAD=tru` used to quietly mean off.

    In a parse whose stated job is catching configuration mistakes at startup, a
    typo in a boolean is exactly as much a mistake as a typo in a port.
    """
    with pytest.raises(ConfigError, match="PIPER_ALLOW_DOWNLOAD"):
        piper.configure({"PIPER_ALLOW_DOWNLOAD": typo}, frozenset(), SERVES)


def test_every_problem_in_this_engine_s_configuration_is_reported_together():
    """One restart, the whole list — and the list an engine contributes to it."""
    with pytest.raises(ConfigError) as raised:
        piper.configure({
            "PIPER_VOICES": "  ,  ",
            "PIPER_MODELS_DIR": " ",
            "PIPER_ALLOW_DOWNLOAD": "maybe",
        }, frozenset(), SERVES)
    joined = " ".join(raised.value.problems)
    assert len(raised.value.problems) == 3
    for expected in ("PIPER_VOICES", "PIPER_MODELS_DIR", "PIPER_ALLOW_DOWNLOAD"):
        assert expected in joined


def test_every_voice_a_multi_voice_deployment_opens_declares_the_same_thing(tmp_path):
    """`open()` stamps one set over a loop, and one voice is not evidence.

    Every other capability test in this file opens a single voice, so `declared()`
    — a union — is trivially that voice's own set, and a stamp that reached only
    the first of several would pass all of them. The published piper image bakes
    three voices, so the multi-voice case is the deployed one.
    """
    make_voice(tmp_path, key="en_US-lessac-medium")
    make_voice(tmp_path, key="en_GB-alba-medium")
    keys = ("en_US-lessac-medium", "en_GB-alba-medium")

    engine = piper_prepared(tmp_path, voices=keys, timings=True).open()

    offered = engine.voices()
    assert len(offered) == len(keys)
    assert all(Capability.TIMESTAMPS in voice.capabilities for voice in offered)
    assert all(Capability.SPEED in voice.capabilities for voice in offered)


def test_the_voices_a_build_reports_declare_what_the_ones_it_boots_will(tmp_path):
    """[FRAMING:representation] Two descriptions of one voice, kept true together.

    `acquire` and `open` both hand back `Voice`s, and a `Voice` now states what
    speaking in it really does. They came from one derivation, so a build cannot
    describe a voice as capability-less that boots able to measure — which is what
    it did while only `open` stamped them.
    """
    make_voice(tmp_path, key="en_US-lessac-medium")
    prepared = piper_prepared(
        tmp_path, voices=("en_US-lessac-medium",), timings=True
    )

    baked = {voice.id: voice.capabilities for voice in prepared.acquire()}
    booted = {voice.id: voice.capabilities for voice in prepared.open().voices()}

    assert baked == booted
    assert Capability.TIMESTAMPS in baked["en_US-lessac-medium"]

    # Same claim over the other thing a `Voice` states about its speaker. Asserted
    # against the real declaration rather than only across the two paths, so a
    # stamp that is consistently wrong fails here too — matching on both sides is
    # what a dropped `serves` argument would also do.
    baked_models = {voice.id: voice.models for voice in prepared.acquire()}
    booted_models = {voice.id: voice.models for voice in prepared.open().voices()}

    assert baked_models == booted_models
    assert baked_models["en_US-lessac-medium"] == serves("piper")
