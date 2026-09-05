"""Fixtures and stand-ins shared by more than one test module.

[LAW:one-source-of-truth] `make_voice` lives here because two files now need a
voice on disk that no real download produced, and a second copy of "what a
`.onnx.json` has to contain" would be free to drift from the first — leaving one
file testing against a sidecar shape the other has stopped believing in.

`DeclaredEngine` and the per-engine asset fixtures — `piper_installed`,
`kokoro_installed`, `kokoro_timeless_installed`, `chatterbox_installed` — are
here for the same reason
from the two other directions a test reaches an engine: the fake one every
capability test drives, and where the real ones' models are kept.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from elvenspeak import chatterbox, router
from elvenspeak.engine import (
    Capability,
    Prosody,
    Speech,
    TimedSpeech,
    Timing,
    Voice,
)

#: Every variable a startup answers for — the server's own and the engines' —
#: since `Settings.from_env` now splices an engine's parse into its own. One
#: copy, because a second one is free to drift: the next setting added would be
#: remembered in one test's clearing list and forgotten in the other, and the
#: test that forgot goes flaky later with nothing pointing at the cause.
#:
#: Retired names belong here too. `ELVENSPEAK_TIMESTAMPS` is no longer read, and
#: is refused rather than ignored — so a shell that still exports it fails a
#: startup just as surely as one that mistyped a port.
_ENVIRONMENT = (
    "ELVENSPEAK_ENGINE",
    "ELVENSPEAK_FALLBACK_VOICE",
    "ELVENSPEAK_API_KEY",
    "ELVENSPEAK_WITHHOLD",
    "ELVENSPEAK_TIMESTAMPS",
    "PIPER_VOICES",
    "PIPER_MODELS_DIR",
    "PIPER_ALLOW_DOWNLOAD",
    "KOKORO_VOICES",
    "KOKORO_MODELS_DIR",
    "KOKORO_MODEL",
    "KOKORO_ALLOW_DOWNLOAD",
    # Read from the module that owns the name rather than spelled again. Piper's
    # and Kokoro's are literals here because those modules expose no constant to
    # read; this one does, and a second spelling of it would stop clearing the
    # real variable the day it changed ([LAW:one-source-of-truth]).
    router.CONSUL_URL,
    router.BACKEND_API_KEY,
    chatterbox.MODELS_DIR,
    chatterbox.DEVICE,
    chatterbox.SPEAKERS,
    chatterbox.LANGUAGES,
    chatterbox.ALLOW_DOWNLOAD,
    "HOST",
    "PORT",
)


#: Third-party libraries that make a module a concrete engine, by the name an
#: `import` statement spells. One entry per engine, and the only line a new
#: engine has to add — reaching an engine's *module* is caught by the same walk,
#: since that module reaches its library.
#:
#: Shared because two files read it from opposite ends of one fact. The seam
#: check in `test_encoding.py` proves the ElevenLabs surface cannot reach any of
#: these; `test_packaging.py` proves the same libraries are installable only
#: through their own engine's extra. A second copy would let a third engine be
#: added to one list and forgotten in the other, and the file that forgot goes
#: quietly vacuous rather than red.
ENGINE_LIBRARIES = frozenset({"piper", "kokoro_onnx", "chatterbox"})


def declared(engine) -> frozenset[Capability]:
    """Everything any voice this engine offers can do — the union.

    The coarse answer, computed where it is wanted rather than stored, which is
    the whole shape of `piper-routing-7e2.4`: capability lives on the voice, and
    an engine-wide summary kept as its own value would be a second source free to
    disagree with the voices it summarises.

    Right for a test asking "does this engine do X at all". A test about what a
    *request* gets reads the voice, because behind a router those differ.
    """
    return frozenset().union(*(voice.capabilities for voice in engine.voices()))


def _use_a_working_espeak() -> None:
    """Points the phonemizer at a system espeak-ng where the bundled one is broken.

    Kokoro phonemizes through `espeakng-loader`, whose macOS wheel (0.2.4, the
    latest) ships a dylib that ignores the data path it is initialized with and
    aborts the process on a path from the machine it was built on. Verified
    against the library directly, so it is the wheel rather than anything above
    it — and it aborts rather than failing to load, so the fallback to a
    system-wide espeak that phonemizer already has never fires.

    `PHONEMIZER_ESPEAK_LIBRARY` is phonemizer's own documented override and is
    read by Kokoro, so this needs nothing from the engine: a developer with
    espeak-ng installed gets working tests, and an environment that already set
    the variable keeps its own answer. The container installs espeak-ng from
    apt and sets this in the Dockerfile, so nothing here is load-bearing for a
    deployment.
    """
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        return
    for candidate in (
        "/opt/homebrew/lib/libespeak-ng.dylib",
        "/usr/local/lib/libespeak-ng.dylib",
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
        "/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1",
    ):
        if Path(candidate).exists():
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = candidate
            return


_use_a_working_espeak()


#: The voice that a test wanting real audio needs installed, and where the image
#: build and the developer loop both put it. One copy, because three files now
#: skip themselves when it is missing and three spellings of "is the model there"
#: are free to disagree about which model or which directory — leaving one file
#: quietly testing nothing while the others ran.
INSTALLED_VOICE = "en_US-lessac-medium"
MODELS_DIR = Path(
    os.environ.get("PIPER_MODELS_DIR", Path(__file__).parent.parent / "models")
)

#: The Kokoro export the suite synthesizes with, and the voices it offers.
#: `KOKORO_TIMELESS_MODEL` is the published `model-files-v1.0` export, whose ONNX
#: graph has no `duration` output — the real deployment in which Kokoro cannot
#: place phonemes in time, and so the subject of the tests about refusing to.
KOKORO_MODEL = "kokoro-v1.0.int8.onnx"
KOKORO_TIMELESS_MODEL = "kokoro-v1.0-notimings.onnx"
KOKORO_VOICES = ("af_heart", "am_michael")
_KOKORO_TIMELESS_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/kokoro-v1.0.int8.onnx"
)


#: The accelerator the suite opens Chatterbox on, and the one setting this engine
#: refuses to guess for itself.
#:
#: `cpu` is the floor rather than a preference: it is the only device every
#: machine that runs this suite has, and a default naming anything else would
#: fail the run on the machines least able to diagnose it. It is also slow —
#: measured at 8-33x real time — so a developer with an accelerator says so
#: through the engine's own variable and gets the same tests several times
#: faster. Read here rather than in each caller, because a second spelling of
#: "which device do the tests use" is free to disagree with this one.
#:
#: Spelled through `chatterbox.DEVICE` for the reason `_ENVIRONMENT` reads the
#: same constant: the variable has one name, and it lives in the module that
#: parses it.
CHATTERBOX_DEVICE = os.environ.get(chatterbox.DEVICE, "cpu")

#: One language for the shared fixtures, against the engine's own two.
#:
#: A Chatterbox voice is `<speaker>-<language>` and one model speaks all of them,
#: so a second language doubles the voices — and therefore doubles every
#: synthesis in `test_conformance.py`, which speaks each voice on offer. What
#: that second voice would exercise is the voice *product*, which is this
#: engine's own property and is asserted in `test_chatterbox.py` where it costs
#: one description rather than four minutes of CPU synthesis.
CHATTERBOX_LANGUAGE = "en"


#: [LAW:no-silent-failure] These replaced a `skipif` that removed the tests when
#: the models were absent. A skip is indistinguishable from a pass in a summary,
#: so the suite's most expensive claims — that a real engine satisfies the
#: contract — silently stopped being made on exactly the machines least likely to
#: have run them before. Provisioning is what the marker should always have done:
#: the assets are obtainable, so a missing one is a thing to fetch rather than a
#: reason to assert nothing.
#:
#: [LAW:composability] One fixture per engine, rather than one that installs
#: everything. A module that exercises Piper should not need Kokoro's ~142 MB and
#: a working espeak-ng, and with a single fixture its asset bill grew every time
#: an engine was registered. Each is fetched through that engine's own `acquire`,
#: which is the door the image build uses, so a change that broke provisioning
#: breaks this too. Idempotent and session-scoped: a warm checkout fetches
#: nothing.


@pytest.fixture(scope="session")
def piper_installed() -> Path:
    """The Piper voice the tests that synthesize for real speak in."""
    piper_prepared(allow_download=True).acquire()
    return MODELS_DIR


@pytest.fixture(scope="session")
def kokoro_installed() -> Path:
    """Kokoro's default export and its style pack."""
    kokoro_prepared(allow_download=True).acquire()
    return MODELS_DIR


@pytest.fixture(scope="session")
def chatterbox_installed() -> Path:
    """Chatterbox's checkpoint set: ~3.06 GiB, and the reason it is its own fixture.

    Fifteen times Kokoro's export, so a module that exercises Piper must not pay
    for it — which is the argument every fixture here is already made under, at
    the scale that makes it obvious. Fetched through `acquire`, the door the
    image build uses, so a change that broke provisioning breaks this too.

    `acquire` also *opens* the model, unlike Piper's and Kokoro's: this engine
    has one model behind every voice, and loading it is the only way to learn
    that 3 GiB of checkpoints are a model rather than 3 GiB of bytes. So this
    fixture costs a load and a clone as well as a download.
    """
    chatterbox_prepared(allow_download=True).acquire()
    return MODELS_DIR


@pytest.fixture(scope="session")
def kokoro_timeless_installed(kokoro_installed: Path) -> Path:
    """The one asset no `acquire` installs: the export reporting no durations.

    Its own fixture, so only the two tests that need this second ~92 MB export
    pay for it. A deployment chooses one export, so no engine's provisioning has
    cause to fetch a second — but the property under test is that the capability
    follows the export rather than the engine's name, and that is unfalsifiable
    with only the export that reports durations.

    Downloaded through `kokoro._fetch` rather than by a copy of it here. This was
    a hand-rolled duplicate that had already lost the empty-body check and the
    partial-file cleanup its original grew, which is the drift a second copy is
    always free to do — and it would have poisoned the shared `models/` cache
    with a file every later run treated as installed. Reaching for a private is
    the smaller evil: there is one implementation of download-verify-rename, and
    the release this asset comes from is the caller's business, which is why
    `_fetch` takes the URL.
    """
    from elvenspeak import kokoro

    kokoro._fetch(MODELS_DIR / KOKORO_TIMELESS_MODEL, _KOKORO_TIMELESS_URL, True)
    return MODELS_DIR


def serves(name: str) -> frozenset[str]:
    """What a deployment running `name` answers to, derived the way the server does.

    [LAW:behavior-not-structure] Reads the real declaration through
    `models.declared_by` against the real registry, exactly as
    `elvenspeak.settings` does, rather than restating an engine's model ids here.
    A literal would be a second copy of `aliases/<name>.toml`, and it would keep
    passing after the file it duplicates stopped saying that.

    Imported inside the call because `elvenspeak.engines` pulls in every engine
    module, which the tests that never open one should not pay for.
    """
    from elvenspeak import engines, models

    return models.declared_by(name, engines.ENGINES)


#: A served set for the parse tests, which carry it through `configure` without
#: ever looking at it. Deliberately not a real engine's: what the set *contains*
#: is `models.declared_by`'s contract and is asserted there, and a parse test that
#: named piper's ids would fail the day piper declared one more.
SERVES = frozenset({"an-engine"})


def kokoro_prepared(
    models_dir: Path = MODELS_DIR,
    *,
    voices: tuple[str, ...] = KOKORO_VOICES,
    model: str = KOKORO_MODEL,
    allow_download: bool = False,
):
    """Kokoro configured for a test, through the door a deployment uses.

    [LAW:behavior-not-structure] Like `piper_prepared`, it goes through
    `kokoro.configure` rather than reaching past it. A test that built the engine
    some other way would keep passing after the parse it skipped stopped being
    able to produce that value.
    """
    from elvenspeak import kokoro

    return kokoro.configure(
        {
            "KOKORO_VOICES": ",".join(voices),
            "KOKORO_MODELS_DIR": str(models_dir),
            "KOKORO_MODEL": model,
            "KOKORO_ALLOW_DOWNLOAD": "1" if allow_download else "0",
        },
        frozenset(),
        serves("kokoro"),
    )


def chatterbox_prepared(
    models_dir: Path = MODELS_DIR,
    *,
    speakers: tuple[str, ...] = (chatterbox.BUILTIN_SPEAKER,),
    languages: tuple[str, ...] = (CHATTERBOX_LANGUAGE,),
    device: str = CHATTERBOX_DEVICE,
    allow_download: bool = False,
):
    """Chatterbox configured for a test, through the door a deployment uses.

    [LAW:behavior-not-structure] Like the two above it, this goes through
    `chatterbox.configure` rather than reaching past it — a test that built
    `_Prepared` directly would keep passing after the parse it skipped stopped
    being able to produce that value.

    The device is named on every call and has no fallback of its own here,
    because the engine has none: `configure` refuses an unset one, and a helper
    that quietly supplied `cpu` would be the default this engine deliberately
    does not have, reintroduced in the one place nothing would look for it.
    """
    return chatterbox.configure(
        {
            chatterbox.SPEAKERS: ",".join(speakers),
            chatterbox.LANGUAGES: ",".join(languages),
            chatterbox.MODELS_DIR: str(models_dir),
            chatterbox.DEVICE: device,
            chatterbox.ALLOW_DOWNLOAD: "1" if allow_download else "0",
        },
        frozenset(),
        serves("chatterbox"),
    )


def piper_prepared(
    models_dir: Path = MODELS_DIR,
    *,
    voices: tuple[str, ...] = (INSTALLED_VOICE,),
    allow_download: bool = False,
    timings: bool = False,
):
    """Piper configured for a test, through the door a deployment uses.

    [LAW:behavior-not-structure] Four files want a Piper engine over a chosen
    directory, and every one of them goes through `piper.configure` rather than
    reaching past it. A test that built the engine some other way would keep
    passing after the parse it skipped stopped being able to produce that value —
    which is the whole guarantee the parse exists to give.
    """
    from elvenspeak import piper

    return piper.configure(
        {
            "PIPER_VOICES": ",".join(voices),
            "PIPER_MODELS_DIR": str(models_dir),
            "PIPER_ALLOW_DOWNLOAD": "1" if allow_download else "0",
        },
        # Not an environment variable any more. Switching timestamps off is the
        # server's decision, made against the shared vocabulary and handed to
        # whichever engine is running — so a test asking Piper not to build
        # alignments now says it the way a deployment does.
        frozenset() if timings else frozenset({Capability.TIMESTAMPS}),
        serves("piper"),
    )

#: What a stand-in engine answers to by `model_id`, in the shape a real one does:
#: its own name, plus a foreign id it declares a mapping for. Every voice below
#: carries it, because a voice whose server names no model id is unreachable by
#: engine — which is a state worth being able to build deliberately and a bad one
#: to inherit by default.
#: `declared` is the name the settings these voices are served under carry, so the
#: stand-in agrees with itself about which engine it is — a deployment whose
#: `engine_name` and whose voices named different engines could not exist.
DECLARED_MODELS = frozenset({"declared", "eleven_fake_v1"})

#: Two, so that "every voice the engine offers" is a claim about more than one.
DECLARED_VOICES = (
    Voice(
        id="fake-voice",
        name="Fake",
        description="a test engine's voice",
        models=DECLARED_MODELS,
        language="en",
    ),
    Voice(
        id="fake-voice-two",
        name="Fake Two",
        description="its other voice",
        models=DECLARED_MODELS,
        language="en",
    ),
)

#: Audio per character of text, at [`_DECLARED_RATE`]. Arbitrary: what matters is
#: that it is a positive constant, which is what makes a longer text audibly
#: longer without the engine having to model anything about speech.
_SAMPLES_PER_CHARACTER = 100
_DECLARED_RATE = 22050


def declaring(
    capabilities: frozenset[Capability], voices: tuple[Voice, ...] = DECLARED_VOICES
) -> tuple[Voice, ...]:
    """`voices`, every one of them declaring `capabilities`.

    The ordinary case, and what the real single-engine implementations do: every
    voice piper opens was opened the same way, so they all say the same thing. An
    engine whose voices genuinely differ builds them separately instead.

    Capabilities only. Which `model_id` values reach a voice is carried on the
    voice a caller passes in — a fleet's whole point is that its backends answer
    to different ones, so a helper that stamped a single set over them would erase
    the difference every router test is looking for.
    """
    return tuple(replace(voice, capabilities=capabilities) for voice in voices)


def answering(models: frozenset[str]) -> tuple[Voice, ...]:
    """The stand-in voices, as a server that answers to `models` would publish them."""
    return tuple(replace(voice, models=models) for voice in DECLARED_VOICES)


class DeclaredEngine:
    """An engine that does exactly what it was told to declare, and nothing more.

    Not a mock of Piper. It declares an arbitrary capability set and makes an
    unconvincing noise, which is the shape of the second engine this seam exists
    for: something written elsewhere, never anticipated here, and described
    accurately anyway.

    [LAW:one-type-per-behavior] One type, rather than a `SpeedlessEngine` and a
    `TimelessEngine` beside it. What differs between the engines these tests need
    is a value — carried on each [`Voice`] it offers — so it is passed as one,
    which is the same argument the interface itself makes and is tested here by
    being relied upon.

    The voice list is a value for the same reason, and the router is what needed
    it: a fleet whose members all offer identical ids can only ever demonstrate
    the collision, never the routing, so proving a voice reached *its own* engine
    means two stand-ins that differ in what they offer.

    It takes finished voices rather than a capability set to spread over them,
    because a set cannot say "this voice declares nothing" and "this voice was not
    given an answer" apart — both are `frozenset()`. Reading emptiness as "inherit
    the engine's set" made an engine whose default is generous and whose one odd
    voice is capability-less inexpressible, and silently promoted that voice
    instead ([LAW:types-are-the-program]). [`declaring`] is the common case; a
    voice that differs is simply built differently.

    Honest in both directions, which is what lets it stand as a subject of the
    conformance suite rather than only as a foil for the API's headers: what it
    declares, it really does, and what it does not declare, it really refuses.
    """

    def __init__(self, voices: tuple[Voice, ...]) -> None:
        self._voices = voices

    def voices(self) -> tuple[Voice, ...]:
        return self._voices

    def speak(self, voice: Voice, text: str, prosody: Prosody) -> Speech:
        return Speech(
            sample_rate=_DECLARED_RATE, audio=_silence(self._length(voice, text, prosody))
        )

    def speak_timed(self, voice: Voice, text: str, prosody: Prosody) -> TimedSpeech:
        """One stretch covering the whole utterance, for an engine that measures.

        [LAW:no-silent-failure] Refuses outright without the capability, rather
        than returning something plausible. The server promises never to ask —
        the timestamp endpoints answer 501 first — and a stand-in that quietly
        obliged anyway would leave that promise resting on a gate no test failure
        would ever be traced back to.
        """
        if Capability.TIMESTAMPS not in voice.capabilities:
            raise AssertionError(
                f"speak_timed was called for voice {voice.id!r}, which did not "
                f"declare {Capability.TIMESTAMPS.name}"
            )
        samples = self._length(voice, text, prosody)
        return TimedSpeech(
            pcm=b"".join(_silence(samples)),
            sample_rate=_DECLARED_RATE,
            timings=(Timing(samples=samples, separates_words=False),),
        )

    def _length(self, voice: Voice, text: str, prosody: Prosody) -> int:
        """How many samples this utterance runs to.

        `prosody.speed` is read only when the *speaking voice* declared
        [`Capability.SPEED`]. A voice that applied a speed it never claimed would
        be the dishonest one the ignored header cannot describe — and reading it
        unconditionally would make this stand-in that engine, silently, for every
        test that constructs it declaring nothing.

        Per voice rather than per engine for the same reason the server asks per
        voice: this stand-in is honest in both directions or it is not a subject
        the conformance suite can trust, and an engine offering a measured voice
        beside an unmeasured one is now expressible.
        """
        speed = prosody.speed if Capability.SPEED in voice.capabilities else 1.0
        return int(len(text) * _SAMPLES_PER_CHARACTER / speed)


@dataclass(frozen=True)
class DeclaredPrepared:
    """[`DeclaredEngine`] as something a deployment could have chosen.

    Satisfies `elvenspeak.provisioning.Prepared` in the smallest honest way: a
    `Settings` needs one, and a test that is about the API surface should not
    have to own a models directory to get it. It is also the worked example of
    the second half of what supplying an engine costs — nothing beyond the two
    methods, no configuration, no assets to install.
    """

    capabilities: frozenset[Capability] = frozenset(Capability)

    def acquire(self) -> tuple[Voice, ...]:
        """Nothing to install, which an engine with no assets says by saying so.

        No voices rather than the two it will serve: this engine makes its noise
        in memory, so a build has nothing to put on disk for it and nothing to
        prove it put there. That is the case `provisioning.Prepared.acquire`
        describes for a remote API, and returning the served voices here would
        leave the only worked example of the assetless path modelling it wrongly.
        """
        return ()

    def open(self) -> DeclaredEngine:
        return DeclaredEngine(declaring(self.capabilities))


def _silence(samples: int) -> Iterator[bytes]:
    """`samples` of quiet, in more than one piece.

    Split because a single chunk would let a consumer that mishandles chunk
    boundaries pass — the streaming encoder is pumped from this iterator, and one
    chunk is the one case where there are no boundaries to get wrong.
    """
    half = samples // 2
    yield b"\x00\x00" * half
    yield b"\x00\x00" * (samples - half)


@pytest.fixture
def clean_env(monkeypatch):
    """An environment holding nothing this service reads.

    For the tests that call `from_env` with no argument — the process-facing
    path, which cannot be handed a dict. Without this they inherit whatever the
    shell exports, and an auto-injected `PORT` or a leftover
    `ELVENSPEAK_TIMESTAMPS` fails them for reasons unrelated to their subject.
    """
    for name in _ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def make_voice(
    models_dir: Path, key: str = "en_US-lessac-medium", sample_rate: int = 22050
) -> None:
    """A voice as far as installing and opening can tell: both halves present.

    No real Piper model is needed. `_describe` reads only the `.onnx.json`
    sidecar and never opens the weights, so the `.onnx` here is a placeholder
    whose only job is to exist.

    Every field the key states is read back out of it, rather than fixed at the
    values its first caller happened to use. A sidecar saying `en_US` for a key
    saying `en_GB` is a fixture contradicting itself, and it costs nothing until
    the first test to assert on a language label gets a wrong answer from the
    thing it was trusting to be right.

    The unpack is the check: a key that is not `<lang>-<name>-<quality>` fails
    here and says so, rather than being written to disk as a voice.
    """
    language, dataset, quality = key.split("-")
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / f"{key}.onnx").write_bytes(b"not a real model")
    (models_dir / f"{key}.onnx.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "language": {"code": language},
                "audio": {"sample_rate": sample_rate, "quality": quality},
                "num_speakers": 1,
            }
        ),
        encoding="utf-8",
    )
