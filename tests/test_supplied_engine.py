"""A project outside this one, supplying its own engine, in full.

The epic's promise is that the ElevenLabs surface is the reusable thing: a
project with a voice model nobody else has should get 28 output formats, four
synthesis endpoints, voice discovery and character timings without forking this
repository or vendoring it. Every other file here checks half of that path.
`test_settings.py` hands `from_env` a registry it made up; `test_capabilities.py`
hands `create_app` an engine it made up; neither joins the two, and the join is
where a project that is not this one actually lives.

So this file is the other project. It writes an engine, registers it, configures
it from an environment, installs its assets, opens it, and drives the real API
over it — and the last test asserts that everything above it was written using
nothing but `import elvenspeak`. That assertion is the deliverable: it is what
makes "the package root is the public surface" a fact rather than a claim, and
what turns a name quietly dropped from `__all__` into a red test instead of a
discovery made by the next person to try this.

# What the engine below is for

A tone generator. Deliberately not a speech synthesizer and deliberately unlike
either engine that ships here: no model format, no phonemizer, a 16 kHz rate
neither in-tree engine uses, and voices that are named frequencies. If the seam
only fits things shaped like Piper, this is what fails.

It is also the worked example of the obligation an outside engine takes on and
that no protocol can state for it:

  * It owns its own assets. `acquire` installs them and `open` refuses without
    them, loudly, rather than opening something degraded. A real engine's
    version of this is a model download, a native library, an espeak-ng that
    has to be on the machine — see the README. The rule is the same: the moment
    to fail is the build or the boot, never the request.
  * It parses its own environment and reports every problem at once, through the
    same [`ConfigError`] the server raises, so an operator sees one list.
  * It declares what it can do, and what it declares is true. Timings here are
    switched off by the deployment rather than by a setting this engine invented
    — the worked example of the one thing an engine is *told* rather than left to
    read, which is what lets the last-but-one test watch a capability withheld by
    the server reach an engine the server has never heard of.
"""

from __future__ import annotations

import ast
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
from conftest import declared
from fastapi.testclient import TestClient

from elvenspeak import (
    Capability,
    ConfigError,
    Prosody,
    Registry,
    Settings,
    Speech,
    TimedSpeech,
    Timing,
    Voice,
    create_app,
)

# --------------------------------------------------------------- the engine

#: 16 kHz, which is neither Piper's 22050 nor Kokoro's 24000. A rate the rest of
#: this project has never seen is the cheapest way to catch anything that
#: silently assumed one.
RATE = 16000
SAMPLES_PER_CHARACTER = 400
#: Silence between words. Long enough that an alignment derived from it is
#: recognisably word-shaped rather than an artefact of rounding.
GAP = 1600
AMPLITUDE = 8000


def timeline(text: str, speed: float) -> tuple[Timing, ...]:
    """What this engine intends to say, before it says any of it.

    The audio is generated *from* this, which is how [`TimedSpeech`]'s invariant
    — the durations sum to the sample count — is kept by construction rather than
    by a second calculation that could disagree with the first. An outside engine
    that measures a real model does the reverse and has to check; one that can
    arrange things this way should.
    """
    stretches: list[Timing] = []
    for word in text.split():
        if stretches:
            stretches.append(Timing(samples=GAP, separates_words=True))
        length = max(1, round(len(word) * SAMPLES_PER_CHARACTER / speed))
        stretches.append(Timing(samples=length, separates_words=False))
    return tuple(stretches)


def tone(hertz: int, stretch: Timing) -> bytes:
    """One stretch as signed 16-bit mono samples, little-endian.

    Packed explicitly rather than through `array`, whose byte order is the
    machine's. The seam asks for one sample format and this is the whole cost of
    meeting it from an engine that natively thinks in floats.
    """
    amplitude = AMPLITUDE * (not stretch.separates_words)
    step = 2 * math.pi * hertz / RATE
    return struct.pack(
        f"<{stretch.samples}h",
        *(int(amplitude * math.sin(step * index)) for index in range(stretch.samples)),
    )


class ToneEngine:
    """Speech, for a generous definition of speech.

    Satisfies `elvenspeak.Engine` structurally — it is a `Protocol`, so this
    class inherits nothing and imports no base to satisfy.
    """

    def __init__(
        self,
        pitches: tuple[tuple[str, int], ...],
        measures: bool,
        serves: frozenset[str],
    ) -> None:
        self._pitches = pitches
        self._serves = serves
        self._capabilities = frozenset(
            {Capability.SPEED, *([Capability.TIMESTAMPS] if measures else [])}
        )

    def voices(self) -> tuple[Voice, ...]:
        return tuple(
            Voice(
                id=f"tone-{name}",
                name=name,
                description=f"a {hertz} Hz tone",
                # Free-form, and where an engine puts what has no field of its
                # own. `labels` is pairs rather than a dict because a `Voice` is
                # frozen and shared by every request.
                labels=(("engine", "tone"), ("hertz", str(hertz))),
                # What speaking in this voice really does. Every tone is made the
                # same way so they all carry the same set — an engine whose voices
                # differ says so here, one voice at a time.
                capabilities=self._capabilities,
                # Every `model_id` this deployment answers to, handed down by
                # `Settings.from_env` from the name this engine is registered
                # under. An engine is never told its own key, so it could not
                # have derived this — and a router in front of it reads the
                # answer off each voice to know which engine speaks it.
                models=self._serves,
            )
            for name, hertz in self._pitches
        )

    def speak(self, voice: Voice, text: str, prosody: Prosody) -> Speech:
        hertz = self._hertz(voice)
        stretches = timeline(text, prosody.speed)
        return Speech(
            sample_rate=RATE,
            audio=(tone(hertz, stretch) for stretch in stretches),
        )

    def speak_timed(self, voice: Voice, text: str, prosody: Prosody) -> TimedSpeech:
        hertz = self._hertz(voice)
        stretches = timeline(text, prosody.speed)
        return TimedSpeech(
            pcm=b"".join(tone(hertz, stretch) for stretch in stretches),
            sample_rate=RATE,
            timings=stretches,
        )

    def _hertz(self, voice: Voice) -> int:
        """The pitch behind an id the server resolved for us.

        [LAW:parse-dont-validate] No guard for an unknown voice: the server hands
        back a `Voice` this engine returned from `voices`, so there is no id here
        to fail to recognise. An engine that re-checked would be answering a
        question the seam already answered.
        """
        return dict(self._pitches)[voice.name]


# -------------------------------------------------- how a deployment gets one

VOICES = "TONE_VOICES"
DIRECTORY = "TONE_DIRECTORY"

#: A voice pack is one line of text. Real ones are ONNX files; what matters for
#: the example is that `open` cannot proceed without what `acquire` wrote.
SUFFIX = ".tone"


def hertz_of(name: str) -> int:
    return 200 + 40 * (sum(name.encode()) % 12)


@dataclass(frozen=True)
class TonePrepared:
    """This deployment's tone engine: configured, checked, not yet built.

    Satisfies `elvenspeak.Prepared`. Constructing one does no I/O, so a whole
    environment — the server's and this engine's — is parsed and rejected before
    anything is written or opened.
    """

    directory: Path
    names: tuple[str, ...]
    measures: bool
    #: Every `model_id` a deployment running this engine answers to.
    serves: frozenset[str]

    def acquire(self) -> tuple[Voice, ...]:
        """Installs the voice packs. The image build's step, never a request's.

        Idempotent, and it returns what it installed rather than nothing, because
        an engine that cannot describe what it put on disk has not put it there —
        and the build is the last moment that failure is cheap.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        for name in self.names:
            self._pack(name).write_text(str(hertz_of(name)), encoding="utf-8")
        return self.open().voices()

    def open(self) -> ToneEngine:
        """Builds the engine from what is on disk, or refuses to.

        [LAW:no-silent-failure] A missing pack raises. The tempting alternative —
        skip that voice and serve the rest — turns a broken deploy into a service
        that answers 200 with the wrong voice, which is this project's own worst
        failure mode written into an engine.
        """
        missing = [name for name in self.names if not self._pack(name).exists()]
        if missing:
            raise FileNotFoundError(
                f"{DIRECTORY}={self.directory} has no voice pack for "
                f"{', '.join(missing)}; run the acquire step first"
            )
        pitches = tuple(
            (name, int(self._pack(name).read_text(encoding="utf-8")))
            for name in self.names
        )
        return ToneEngine(pitches, self.measures, self.serves)

    def _pack(self, name: str) -> Path:
        return self.directory / f"{name}{SUFFIX}"


def configure(
    environ: Mapping[str, str],
    withheld: frozenset[Capability],
    serves: frozenset[str],
) -> TonePrepared:
    """This engine's whole configuration, checked in one pass.

    Satisfies `elvenspeak.Configure`. `Settings.from_env` splices these problems
    into its own, so a bad `PORT` and a bad `TONE_VOICES` reach the operator
    together — which is the point of collecting a list rather than raising on the
    first one.

    Whether to measure comes from `withheld` and never from `environ`. This
    engine could spell it `TONE_TIMINGS` and parse it here, and that is the
    mistake worth showing the way round: a deployment that switched timestamps
    off would be switching them off for this engine only, and would silently get
    them back the day it ran another one. What can be switched off is named
    against the vocabulary every engine shares, parsed by the server once, and
    handed here — so this engine reads a deployment's decision rather than
    inventing a private way to be told it.
    """
    problems: list[str] = []

    names = tuple(part.strip() for part in environ.get(VOICES, "").split(",") if part.strip())
    if not names:
        problems.append(f"{VOICES} names no voices")
    problems.extend(
        f"{VOICES} entry {name!r} is not a voice name" for name in names if not name.isalnum()
    )

    directory = environ.get(DIRECTORY, "").strip()
    if not directory:
        problems.append(f"{DIRECTORY} is not set")

    if problems:
        raise ConfigError(problems)
    # Told, not overruled: the server subtracts what it withheld from whatever
    # this engine declares, so building the machinery anyway would only waste
    # the work. An engine with nothing to save is free to ignore this.
    return TonePrepared(
        Path(directory), names, Capability.TIMESTAMPS not in withheld, serves
    )


#: The whole registration. A name and something that turns an environment into a
#: `Prepared` — no entry in `elvenspeak.engines`, no edit to this package, no
#: fork. `Settings.from_env` takes this as an argument precisely so that a
#: registry written elsewhere is a first-class one.
SUPPLIED: Registry = {"tone": configure}


# --------------------------------------------------------------- the tests

TEXT = "A supplied engine sounds like this"


@pytest.fixture(scope="module")
def environ(tmp_path_factory) -> dict[str, str]:
    return {
        "ELVENSPEAK_ENGINE": "tone",
        VOICES: "low,high",
        DIRECTORY: str(tmp_path_factory.mktemp("tone")),
    }


def serving(environ: Mapping[str, str]) -> TestClient:
    """The other project's entry point, in three lines.

    Parse, install, open, serve — the same order `main.py` and the image's bake
    step use between them, and the only code this project has to write to get the
    whole ElevenLabs surface.
    """
    settings = Settings.from_env(SUPPLIED, environ)
    settings.engine.acquire()
    return TestClient(create_app(settings, settings.engine.open()))


@pytest.fixture(scope="module")
def client(environ: dict[str, str]) -> TestClient:
    return serving(environ)


def test_a_registry_this_package_never_saw_selects_the_engine(environ):
    """`ELVENSPEAK_ENGINE` names something no module of this package mentions.

    The whole extension point in one assertion: the lookup table is a parameter,
    so the set of selectable engines is the caller's to decide. Nothing here had
    to be registered anywhere, and `elvenspeak.engines` — the roster this
    repository ships — is not imported by this file at all.
    """
    prepared = Settings.from_env(SUPPLIED, environ).engine
    assert isinstance(prepared, TonePrepared)
    assert prepared.names == ("low", "high")


def test_an_unnamed_engine_is_the_supplied_registrys_own_default(environ):
    """A project supplying one engine never has to set the variable.

    The default is the registry's first entry rather than a name spelled inside
    this package, so a registry that never mentions Piper does not inherit it.
    """
    unnamed = {key: value for key, value in environ.items() if key != "ELVENSPEAK_ENGINE"}
    assert isinstance(Settings.from_env(SUPPLIED, unnamed).engine, TonePrepared)


def test_the_supplied_engines_problems_arrive_with_the_servers(environ):
    """One list, one moment, whoever wrote the engine.

    An outside engine raising `ConfigError` gets its complaints spliced into the
    server's, which is the difference between an operator learning their whole
    configuration is wrong and learning it one restart at a time. Both halves are
    broken here so the splice is what is being tested rather than either side.
    """
    with pytest.raises(ConfigError) as raised:
        Settings.from_env(SUPPLIED, {**environ, "PORT": "eighty", VOICES: "low,not a name"})

    problems = " ".join(raised.value.problems)
    assert "PORT" in problems
    assert VOICES in problems


def test_opening_what_was_never_acquired_fails_loudly(tmp_path):
    """The obligation no protocol can state: an engine owns its own assets.

    `Prepared.open` is the last moment before the port is bound, and an engine
    missing what it needs — a model, a native library, a voice pack — refuses
    here rather than degrading. The alternative is a service that boots, answers
    `/health`, and is wrong.
    """
    prepared = configure(
        {VOICES: "low", DIRECTORY: str(tmp_path)}, frozenset(), frozenset({"tone"})
    )
    with pytest.raises(FileNotFoundError, match="voice pack"):
        prepared.open()


def test_the_listing_is_the_supplied_engines_voices(client):
    """`GET /v1/voices`, answered entirely out of an engine written elsewhere.

    Compared as a set, because the listing is sorted by id for presentation while
    the engine's own order means something else — see the test below. Asserting
    the engine's order here would pin a coincidence and go red the day somebody
    renames a voice.
    """
    listed = client.get("/v1/voices").json()["voices"]
    assert {voice["voice_id"] for voice in listed} == {"tone-low", "tone-high"}
    assert all(voice["labels"]["engine"] == "tone" for voice in listed)


def test_the_supplied_engines_own_order_chooses_the_fallback_voice(client):
    """`voices()` promises "best first", and this deployment named no fallback.

    So an id this server does not know is answered in whatever the engine listed
    first — `tone-low`, not the alphabetically first `tone-high`. That is the
    load-bearing half of the ordering promise, and it belongs to the engine
    rather than to the listing.
    """
    response = client.post(
        "/v1/text-to-speech/an-elevenlabs-id-from-somewhere-else", json={"text": TEXT}
    )
    assert response.status_code == 200
    assert response.headers["x-elvenspeak-voice"] == "tone-low"


def test_synthesis_returns_the_supplied_engines_own_audio(client):
    """The engine's rate reaches the wire, and nothing resampled it on the way.

    `pcm_16000` is the format that matches what this engine makes, so the body is
    the samples it produced. Compared against the engine's own arithmetic rather
    than a constant: a hardcoded length would pass just as well against audio the
    server had quietly stretched to somebody else's rate.
    """
    expected = sum(stretch.samples for stretch in timeline(TEXT, 1.0))
    response = client.post(
        "/v1/text-to-speech/tone-low?output_format=pcm_16000", json={"text": TEXT}
    )
    assert response.status_code == 200
    assert len(response.content) == expected * 2


def test_a_supplied_engines_speed_reaches_it(client):
    """[`Capability.SPEED`] declared by an outside engine, honoured by the server.

    The parameter is neutralised for an engine that did not declare it, so this
    is also the check that the declaration crossing the seam is what decides —
    not a list of engines the API surface knows about.
    """
    def samples(speed: float) -> int:
        response = client.post(
            "/v1/text-to-speech/tone-low?output_format=pcm_16000",
            json={"text": TEXT, "voice_settings": {"speed": speed}},
        )
        return len(response.content)

    assert samples(2.0) < samples(0.5)
    assert "speed" not in client.post(
        "/v1/text-to-speech/tone-low",
        json={"text": TEXT, "voice_settings": {"speed": 1.5}},
    ).headers.get("x-elvenspeak-ignored", "")


def test_character_timings_come_out_word_exact(client):
    """The most valuable thing the surface does, over an engine written elsewhere.

    An outside engine reports stretches and says which of them fall between
    words; `elvenspeak.alignment` turns that into per-character timings. Nothing
    about this engine's stretches is phoneme-shaped, which is the point — the
    seam asks two questions a word-level, character-level or phoneme-level engine
    can all answer, and `word-exact` here is that carve being right rather than
    Piper-flavoured.
    """
    body = client.post(
        "/v1/text-to-speech/tone-low/with-timestamps", json={"text": TEXT}
    ).json()

    alignment = body["alignment"]
    assert "".join(alignment["characters"]) == TEXT
    assert alignment["character_start_times_seconds"][0] == 0.0
    assert alignment["character_end_times_seconds"] == sorted(
        alignment["character_end_times_seconds"]
    )
    total = sum(stretch.samples for stretch in timeline(TEXT, 1.0)) / RATE
    assert alignment["character_end_times_seconds"][-1] == pytest.approx(total, abs=1e-3)


def test_a_capability_this_deployment_withheld_reaches_the_supplied_engine(environ):
    """One setting, one vocabulary, and an outside engine that hears it.

    `ELVENSPEAK_WITHHOLD` is the server's own and names a `Capability`, so it
    means the same thing whichever engine is running — where the setting it
    replaced was `ELVENSPEAK_TIMESTAMPS`, read by Piper alone, which left a
    deployment that switched timestamps off and then ran a different engine being
    answered with timestamps anyway.

    Both halves are asserted because they are different promises. The engine
    really declines the capability, which is what makes the setting worth more
    than a filter over the response — an engine is told in time to not build the
    machinery. And the endpoint really refuses, which is the server's own
    enforcement and is what an engine that ignored the offer would still get.
    """
    withholding = {**environ, "ELVENSPEAK_WITHHOLD": "timestamps"}
    prepared = Settings.from_env(SUPPLIED, withholding).engine
    assert Capability.TIMESTAMPS not in declared(prepared.open())

    timeless = serving(withholding)
    response = timeless.post(
        "/v1/text-to-speech/tone-low/with-timestamps", json={"text": TEXT}
    )
    assert response.status_code == 501
    assert "ELVENSPEAK_" not in response.text
    assert timeless.post(
        "/v1/text-to-speech/tone-low", json={"text": TEXT}
    ).status_code == 200


def test_this_example_was_written_against_the_package_root_alone(client):
    """The claim the whole file exists to make, asserted about the file itself.

    Everything above imports from `elvenspeak` and nothing from
    `elvenspeak.engine`, `elvenspeak.provisioning` or anywhere else inside the
    package. So the public surface really is sufficient to supply an engine, and
    `__all__` is the one list an outside author has to read — which is what makes
    `test_packaging.py`'s check that `__all__` covers the seam worth having.

    Reads this file rather than trusting the import block at the top, because the
    failure is somebody adding a submodule import three hundred lines down when
    the root turns out to be missing something. That is exactly the moment the
    surface stopped being sufficient, and it is the moment nobody notices.
    """
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    reached = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("elvenspeak")
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name.startswith("elvenspeak")
    }
    assert reached == {"elvenspeak"}
