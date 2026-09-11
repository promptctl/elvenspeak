"""Fixtures and stand-ins shared by more than one test module.

[LAW:one-source-of-truth] `make_voice` lives here because two files now need a
voice on disk that no real download produced, and a second copy of "what a
`.onnx.json` has to contain" would be free to drift from the first — leaving one
file testing against a sidecar shape the other has stopped believing in.

`DeclaredEngine` is here for the same reason: it is the engine every
capability test drives, and more than one module drives it.

The per-engine asset fixtures that used to sit beside it — `piper_installed`,
`kokoro_installed`, `kokoro_timeless_installed`, `chatterbox_installed` — are
gone, and with them the ~3.4 GB the suite fetched to run. Nothing here provisions
a model any more. What a real model proves about a real deployment is asked of
the built image by `speaks.py`, which runs the artifact rather than a session
this machine happened to assemble; each engine's own test module stands in for
the library at the seam it actually reads.

The reclaim that ran before and after every test went with them, and it went on
a measurement rather than an argument, because that is what its own docstring
asked for: 1537 calls cost 116s of cpu, and taking them out — with the file that
tested them — took the suite from 233s to 147s and moved the peak resident set
from 590.0 MiB to 601.2 MiB. Eleven mebibytes, against the runner that was
OOM-killed at 5.5 GiB carrying a 3412 MiB Chatterbox model no test was using.
Both halves of what it defended against left with the models: `malloc_trim` hands
back pages a model dirtied, and the FastAPI memos it cleared pin every app the
suite builds — which now weigh kilobytes apiece, four thousand entries before the
cache evicts one. `tests/rsscurve.py` still names the test that raises the
high-water mark, so a model finding its way back into this suite is still legible
from a killed run.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from itertools import groupby
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from elvenspeak import chatterbox, kokoro, memory, router, settings as settings_mod
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
#: The server's half is read from `settings.VOCABULARY` and not listed, which is
#: what makes that true rather than merely intended: it is derived from the
#: `Settings` fields, so a setting added there is cleared here having been typed
#: nowhere ([LAW:one-source-of-truth]). Retired names are in it too —
#: `ELVENSPEAK_TIMESTAMPS` is no longer read and is refused rather than ignored,
#: so a shell that still exports it fails a startup just as surely as one that
#: mistyped a port. `HOST` and `PORT` are in it for the same reason: unprefixed,
#: but still variables a `Settings` field declares it is filled from.
#:
#: The engines' half stays spelled out: those modules parse their own prefixes
#: privately and expose no equivalent set, so a copy is the honest shape here.
#: Where one does own its name as a constant, that constant is read rather than
#: spelled again — Piper's are literals because it exposes none, and Kokoro's own
#: prefix is literal for the same reason, but the espeak library it refuses to
#: boot without is a name three files have to agree on, so it owns that one.
_ENVIRONMENT = (
    *sorted(settings_mod.VOCABULARY),
    "PIPER_VOICES",
    "PIPER_MODELS_DIR",
    "PIPER_ALLOW_DOWNLOAD",
    "KOKORO_VOICES",
    "KOKORO_MODELS_DIR",
    "KOKORO_MODEL",
    "KOKORO_ALLOW_DOWNLOAD",
    kokoro.ESPEAK_LIBRARY,
    router.CONSUL_URL,
    router.BACKEND_API_KEY,
    chatterbox.MODELS_DIR,
    chatterbox.DEVICE,
    chatterbox.SPEAKERS,
    chatterbox.LANGUAGES,
    chatterbox.ALLOW_DOWNLOAD,
)


@pytest.fixture(autouse=True)
def _unconfined(monkeypatch):
    """Every test runs as though nothing caps this process's memory.

    [LAW:no-ambient-temporal-coupling] `settings.unsized` refuses to serve when a
    process is memory-confined and named no synthesis ceiling, and `_ENVIRONMENT`
    below clears that variable for every test — so on a memory-confined runner
    every test that reaches `main.build()`'s success path would fail for a reason
    it has nothing to do with. `.gitea/workflows/publish-image.yaml` instruments
    cgroup memory for this exact job because of past SIGKILL-137s, so that runner
    is a live candidate and the suite would go red there and green here.

    Worse than red, in one case: `_app` refuses before `Catalog` runs, so
    `test_a_fallback_naming_no_offered_voice_exits_the_same_way` would fail while
    the check it exists to pin was never reached.

    So the world is stated rather than inherited, in the same spirit as the
    variable clearing below. A test that is about confinement passes a limit to
    `unsized` directly, or overrides this fixture — see `tests/test_settings.py`
    and `tests/test_main.py`.
    """
    monkeypatch.setattr(memory, "limit", lambda *_: memory.Unconfined.UNCONFINED)


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

    `ESPEAK_LIBRARY` is phonemizer's own documented override and is the variable
    Kokoro refuses to boot without, so this needs nothing from the engine: a
    developer with espeak-ng installed gets working tests, and an environment
    that already set the variable keeps its own answer. The container installs
    espeak-ng from apt and sets this in the Dockerfile, so nothing here is
    load-bearing for a deployment.

    [LAW:one-source-of-truth] The name and the candidates are the engine's, read
    from it rather than spelled again — two lists of where espeak lives would be
    free to disagree about the platform added to one of them.

    That the engine may not do what this function does is the whole distinction,
    and it survives sharing the list. A *test harness* choosing a library for the
    machine it is running on is a convenience at the edge; the *engine* choosing
    one would be a silent fallback inside the component whose job is to fail
    loudly. Shared data, opposite permissions — see the constant's own comment.
    """
    if os.environ.get(kokoro.ESPEAK_LIBRARY):
        return
    for candidate in kokoro.ESPEAK_LIBRARY_CANDIDATES:
        if Path(candidate).exists():
            os.environ[kokoro.ESPEAK_LIBRARY] = candidate
            return


_use_a_working_espeak()

#: The library `kokoro_prepared` names, and a path nothing here ever opens.
#:
#: Stated rather than inherited from the machine, in the same spirit as
#: `_unconfined` stating an unconfined process. No test in `test_kokoro.py`
#: phonemizes — every `open` there runs against the `_Session` stand-in — so what
#: those tests need is a deployment that *named* a library, not this machine's
#: real one. Reading the real one would fail them on a host with no espeak-ng, or
#: with one outside the four candidates above, for a library they never call.
#:
#: `Prepared.open` cannot tell the difference, because it is forbidden from
#: looking at the path: it checks that a deployment named one and never that the
#: name resolves. Which is the property these tests are exercising.
NAMED_ESPEAK = "/stated-by-the-tests/libespeak-ng.so.1"


#: The voice that a test wanting real audio needs installed, and where the image
#: build and the developer loop both put it. One copy, because three files now
#: skip themselves when it is missing and three spellings of "is the model there"
#: are free to disagree about which model or which directory — leaving one file
#: quietly testing nothing while the others ran.
INSTALLED_VOICE = "en_US-lessac-medium"
MODELS_DIR = Path(
    os.environ.get("PIPER_MODELS_DIR", Path(__file__).parent.parent / "models")
)

#: The Kokoro voices the tests configure a deployment with.
KOKORO_VOICES = ("af_heart", "am_michael")
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
    model: str = kokoro.DEFAULT_MODEL,
    allow_download: bool = False,
):
    """Kokoro configured for a test, through the door a deployment uses.

    [LAW:behavior-not-structure] Like `piper_prepared`, it goes through
    `kokoro.configure` rather than reaching past it. A test that built the engine
    some other way would keep passing after the parse it skipped stopped being
    able to produce that value.

    It names `NAMED_ESPEAK` because `Prepared.open` refuses a deployment that
    named no espeak library, and stating one keeps these tests independent of
    whether the machine running them has espeak-ng at all.
    """
    return kokoro.configure(
        {
            "KOKORO_VOICES": ",".join(voices),
            "KOKORO_MODELS_DIR": str(models_dir),
            "KOKORO_MODEL": model,
            "KOKORO_ALLOW_DOWNLOAD": "1" if allow_download else "0",
            kokoro.ESPEAK_LIBRARY: NAMED_ESPEAK,
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
        """One stretch per word and per gap, for an engine that measures.

        [LAW:no-silent-failure] Refuses outright without the capability, rather
        than returning something plausible. The server promises never to ask —
        the timestamp endpoints answer 501 first — and a stand-in that quietly
        obliged anyway would leave that promise resting on a gate no test failure
        would ever be traced back to.

        `measured=True` because it is true: this engine decided where each word
        ends, which is the whole of what the flag claims. It reported one
        undivided stretch and left the flag at its default until
        `piper-pipeline-4mx`, which was the same engine claiming
        [`Capability.TIMESTAMPS`] and then handing back the answer of an engine
        that had measured nothing — honest in neither direction, and the reason
        the API's word-exact header had to be checked against a real Piper.
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
            timings=_stretches(text, samples),
            measured=True,
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


def _stretches(text: str, samples: int) -> tuple[Timing, ...]:
    """`text` carved into the word and gap stretches a measuring engine reports.

    The carve is by whitespace and nothing else, which is what makes this a
    stand-in for an engine that never heard of phonemes: `elvenspeak.alignment`
    asks only where the words are and how long each stretch ran, and an engine
    that answers those two questions at word level gets per-character timings out
    of the surface. `tests/test_supplied_engine.py` makes that same point from
    outside the package.

    [LAW:one-source-of-truth] Every boundary is a fraction of the *sample count*
    rather than a duration accumulated stretch by stretch, so the durations sum
    to exactly `samples` however the rounding falls — which is the promise
    `TimedSpeech` makes and the property `test_conformance` reads back.
    """
    runs = ["".join(run) for _, run in groupby(text, key=str.isspace)]
    timings: list[Timing] = []
    consumed = previous = 0
    for run in runs:
        consumed += len(run)
        boundary = round(samples * consumed / len(text))
        timings.append(
            Timing(samples=boundary - previous, separates_words=run[0].isspace())
        )
        previous = boundary
    return tuple(timings)


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
