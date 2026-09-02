"""Piper, as one engine behind [`elvenspeak.engine`]'s interface.

Everything Piper-shaped in this service lives here: ONNX sessions, `.onnx.json`
sidecars, espeak's phoneme alphabet, `length_scale`, the voice download, and —
since [`configure`] — the environment variables that name all of it. Nothing
outside this module names any of it, which is what makes the ElevenLabs surface
above reusable with a different engine under it.

This module is `elvenspeak.piper`; the library it wraps is the top-level `piper`.
Imports here are absolute, so `from piper import ...` reaches the library and
never this file.

# Why this module reads its own environment

`PIPER_MODELS_DIR` and `PIPER_ALLOW_DOWNLOAD` describe a directory of ONNX files,
which is a fact about Piper and meaningless to an engine that reaches a remote
API. Parsed here, they are asked for by the one component that understands them;
held in [`Settings`] instead, they would be fields every other engine is handed
and ignores — and the first engine needing a credential would add a field to the
server's configuration that Piper in turn ignores ([LAW:types-are-the-program]).
[`configure`] raises the same [`ConfigError`] the server does, so a Piper problem
and a port problem arrive in one list at one moment.

# Why the voices are on disk and loaded before anything is served

A Piper voice is a ~60 MB ONNX file, and building a session from one is seconds
of work. Doing either inside a request charges an unbounded, silent delay to
whichever caller happened to be first, on the event loop, stalling every other
request including `/health`. [`_Prepared.open`] therefore fetches, describes and
opens every voice before it returns, so a missing, truncated or corrupt model is
one clean failure to boot — and so [`PiperEngine.voices`] can promise what the
interface asks of it: these can be spoken *now*.

# Why alignment support is decided once for the process

`include_alignments` patches a voice's graph in memory at load time to expose
per-phoneme durations. It is a property of the session, not of a call, so a
request asking for timings against an unpatched session could only fail or lie.
Whether to pay for it is settled before any voice is opened, and leaves this
module as [`engine.Capability.TIMESTAMPS`] declared or not declared — which is
all the server ever learns about it, and the reason the 501 it sends no longer
names a Piper environment variable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from . import engine
from .provisioning import ConfigError, flag

if TYPE_CHECKING:  # pragma: no cover - import cost is real, the symbol is not
    from collections.abc import Iterator, Mapping

    from piper import PiperVoice

_LOGGER = logging.getLogger("elvenspeak.piper")

#: The voice installed when none is named. Still lessac, which is the one Piper
#: voice that is MIT at the repository level, so a default install carries no
#: licence surprise — see README, "Voice licensing". `high` rather than `medium`
#: buys the better synthesis the tier names for 114 MB against 63 MB, paid once
#: per voice the image bakes.
#:
#: Load-bearing beyond the download. [`elvenspeak.voices.Substitution`]'s
#: shipped default answers an unknown id in whichever voice the engine lists
#: first, and the engine lists them in the order they were configured — so this
#: is what every deployment that names no fallback speaks in. The image bakes
#: more than this one voice, and `tests/test_dockerfile.py` holds this equal to
#: the first name in `ARG PIPER_VOICES` so that the two cannot disagree about
#: which one comes first.
DEFAULT_VOICE = "en_US-lessac-high"

#: Phonemes espeak emits that mark structure rather than sound — the run-up into
#: an utterance, the run-out, a line break. They consume real time but belong to
#: no word, which is exactly what [`engine.Timing.separates_words`] reports.
_BOUNDARY_PHONEMES = frozenset({"^", "$", "\n"})

# `<lang>-<name>-<quality>`, the shape every Piper voice key has. Used to read a
# key's parts back when the sidecar omits them — never to decide whether a voice
# exists, which only the files on disk can answer.
_KEY_PARTS = 3

#: What Piper does whatever it was configured with. `length_scale` is on every
#: voice's synthesis config, so the rate is never not variable — unlike
#: alignments, which cost memory and are settled at load time below.
_INHERENT = frozenset({engine.Capability.SPEED})


@dataclass(frozen=True)
class _Ready:
    """A voice whose files are present and whose sidecar has been read.

    [LAW:parse-dont-validate] What [`_install`] returns, and it exists because the
    obvious return — the path it was handed — proves nothing. A path says two
    files have the right names; this says the voice can actually be described,
    which is what "installed" has to mean for the image build that depends on it.
    """

    voice: engine.Voice
    sample_rate: int
    model_path: Path


@dataclass(frozen=True)
class _Installed:
    """One voice as this engine holds it: what the server sees, plus the model.

    The split is the seam. `voice` is the value that crosses it and carries no
    path and no session; the other two fields are why that is affordable — an id
    is all the server needs to name a voice, because this side keeps everything
    required to speak in one.
    """

    voice: engine.Voice
    sample_rate: int
    model: "PiperVoice"


class PiperEngine:
    """Speech from local Piper voices, opened and ready.

    Constructed by [`_Prepared.open`] rather than directly, because an instance
    holding a voice it cannot speak in would be the state this whole module is
    arranged to make unreachable.
    """

    def __init__(self, installed: dict[str, _Installed]) -> None:
        self._installed = installed

    def voices(self) -> tuple[engine.Voice, ...]:
        """Every configured voice, in the order the operator named them.

        Configured order rather than sorted, because the interface makes this
        order mean something: the first voice offered is what answers for an
        unknown id when the deployment named no fallback. Sorted, that default
        was whichever id happened to come first alphabetically — so an operator
        who listed their preferred voice first in `PIPER_VOICES` and left the
        fallback unset silently got a different one. Stable across calls either
        way, which is all [`engine.Engine.voices`] asks; this order is also
        true.

        `GET /v1/voices` is unaffected: [`elvenspeak.voices.Catalog.installed`]
        sorts for display on its own.
        """
        return tuple(installed.voice for installed in self._installed.values())

    def speak(
        self, voice: engine.Voice, text: str, prosody: engine.Prosody
    ) -> engine.Speech:
        installed = self._installed[voice.id]
        # The generator is unstarted: the rate is known from the sidecar, so
        # nothing is synthesized until the samples are actually pulled, and a
        # caller that goes away before reading has cost nothing.
        return engine.Speech(
            sample_rate=installed.sample_rate,
            audio=_stream(installed.model, text, prosody),
        )

    def speak_timed(
        self, voice: engine.Voice, text: str, prosody: engine.Prosody
    ) -> engine.TimedSpeech:
        installed = self._installed[voice.id]
        audio: list[bytes] = []
        timings: list[engine.Timing] = []
        measured = True

        for chunk in installed.model.synthesize(
            text, syn_config=_synthesis_config(prosody), include_alignments=True
        ):
            samples = chunk.audio_int16_bytes
            audio.append(samples)
            alignments = chunk.phoneme_alignments
            if alignments:
                timings += [
                    engine.Timing(
                        samples=int(item.num_samples),
                        separates_words=_separates_words(item.phoneme),
                    )
                    for item in alignments
                ]
            elif samples:
                # Audio the model produced without saying which phonemes made
                # it. Dropping it would leave the timings summing to less than
                # the audio they describe, and every timing derived from that sum
                # would be short — a whole timeline quietly compressed against
                # real audio. Recorded instead as an unattributed stretch, marked
                # as a separator so it reads as silence between words rather than
                # as part of one, and the result stops claiming to be measured.
                timings.append(
                    engine.Timing(samples=len(samples) // 2, separates_words=True)
                )
                measured = False

        return engine.TimedSpeech(
            pcm=b"".join(audio),
            sample_rate=installed.sample_rate,
            timings=tuple(timings),
            measured=measured,
        )


@dataclass(frozen=True)
class _Prepared:
    """Piper as this deployment configured it, before anything was fetched.

    Satisfies [`elvenspeak.provisioning.Prepared`]. Every field came out of
    [`configure`], so both methods take no arguments and there is no way to reach
    either of them with a value the environment check has not already seen.
    """

    keys: tuple[str, ...]
    models_dir: Path
    #: Whether [`open`] may fetch a voice that is missing at boot. Not consulted
    #: by [`acquire`], which fetches by definition — see its docstring.
    allow_download: bool
    timings: bool
    #: Every `model_id` this deployment answers to, stamped onto each voice beside
    #: its capabilities and for the same reason — it is a fact about what will
    #: speak. Arrives from [`configure`] because the name it was derived from is
    #: the key this module is registered under, which this module never learns.
    serves: frozenset[str]

    def acquire(self) -> tuple[engine.Voice, ...]:
        """Puts every configured voice on disk, and says what they turned out to be.

        Downloading is unconditional here, and that is the whole difference
        between this method and [`open`]. It used to be a flag both paths read,
        which left the build overriding the deployment's own setting to `True`
        from a constant beside the call — a fact stated twice, in opposite
        directions, that only agreed because one caller remembered to disagree.
        The lifecycle moment is carried by which method the caller reached for.
        """
        return tuple(
            replace(ready.voice, capabilities=self.capabilities())
            for ready in _install(
                self.keys, self.models_dir, True, self.serves
            ).values()
        )

    def capabilities(self) -> frozenset[engine.Capability]:
        """What every voice this deployment opens will declare.

        [LAW:one-source-of-truth] Read by both lifecycle methods rather than
        computed in each. A `Voice` now states what speaking in it really does, so
        the voices [`acquire`] describes have to say the same thing as the ones
        [`open`] serves — two spellings of one derivation would let a build report
        a voice that boots differently.

        `include_alignments` patches the graph at load time, so this is decided by
        the flag those sessions will be opened under. Every Piper voice in one
        process is opened the same way, so they all carry the same set.
        """
        return _INHERENT | (
            frozenset({engine.Capability.TIMESTAMPS}) if self.timings else frozenset()
        )

    def open(self) -> PiperEngine:
        """Opens every configured voice and returns the engine that speaks them."""
        from piper import PiperVoice

        capabilities = self.capabilities()

        installed: dict[str, _Installed] = {}
        for key, ready in _install(
            self.keys, self.models_dir, self.allow_download, self.serves
        ).items():
            _LOGGER.info("loading voice %s", key)
            installed[key] = _Installed(
                voice=replace(ready.voice, capabilities=capabilities),
                sample_rate=ready.sample_rate,
                model=PiperVoice.load(
                    str(ready.model_path), include_alignments=self.timings
                ),
            )

        return PiperEngine(installed)


def configure(
    env: "Mapping[str, str]",
    withheld: frozenset[engine.Capability],
    serves: frozenset[str],
) -> _Prepared:
    """Reads Piper's own environment, or says everything wrong with it at once.

    [LAW:parse-dont-validate] The checkpoint for this engine. Nothing below holds
    a string out of `env`, and nothing above holds a `models_dir`: what crosses
    is a [`_Prepared`] that could not have been built before these checks ran.

    Every problem is collected rather than raised at the first, because this list
    is spliced into the server's own — an operator bringing the service up should
    not discover a bad voice name and then, one restart later, a bad port.

    Whether to build alignments comes from `withheld` and not from `env`. It was
    `ELVENSPEAK_TIMESTAMPS`, parsed here — this engine's private name for a thing
    every engine has, which meant a deployment that switched timestamps off and
    then ran a different engine got them anyway. The server owns that decision
    now; this engine is only told, and only so it can decline to pay for what
    nobody will ask it for.
    """
    problems: list[str] = []

    keys = tuple(
        name.strip()
        for name in env.get("PIPER_VOICES", DEFAULT_VOICE).split(",")
        if name.strip()
    )
    if not keys:
        problems.append("PIPER_VOICES is empty; name at least one voice")

    # Stripped and checked like everything else here. `PIPER_MODELS_DIR=` is a
    # present key, so `get` returns "" rather than the default, `Path("")` is the
    # working directory, and `mkdir` on it succeeds — the server then reads and
    # writes 60 MB models wherever it happened to be launched from, having
    # reported nothing. An unset variable interpolated into a compose file is an
    # ordinary way to arrive there.
    models_text = env.get("PIPER_MODELS_DIR", "").strip()
    if "PIPER_MODELS_DIR" in env and not models_text:
        problems.append("PIPER_MODELS_DIR is empty; name a directory or unset it")
    models_dir = Path(models_text or str(Path(__file__).parent.parent / "models"))

    try:
        allow_download = flag(env, "PIPER_ALLOW_DOWNLOAD", default=True)
    except ValueError as error:
        problems.append(str(error))
        allow_download = True

    if problems:
        raise ConfigError(problems)

    return _Prepared(
        keys=keys,
        models_dir=models_dir,
        allow_download=allow_download,
        # The saving is the point: an unpatched graph is the memory a withheld
        # TIMESTAMPS buys back, and it can only be unpatched by a session that
        # was never opened patched. Decided here, before anything is opened.
        timings=engine.Capability.TIMESTAMPS not in withheld,
        serves=serves,
    )


def _install(
    keys: tuple[str, ...],
    models_dir: Path,
    allow_download: bool,
    serves: frozenset[str],
) -> dict[str, _Ready]:
    """Makes every named voice present and readable, and says what they are.

    [LAW:decomposition] The joint between [`_Prepared.acquire`] and
    [`_Prepared.open`], and a real one: the image bakes voices at build time and
    wants exactly this and no more, because opening an ONNX session per voice
    only to discard it costs a minute and a gigabyte for nothing. Both callers
    want this whole function and differ only in what they do afterwards — and in
    whether a missing voice may be fetched, which is the one thing they are
    allowed to disagree about.

    The sidecar is read here rather than only by the caller that opens sessions.
    An earlier cut stopped
    at "both files exist", which let a truncated or malformed `.onnx.json` — the
    interrupted-write case the checks below already exist for — produce a green
    image that failed at container startup instead. The bake is the last moment
    that failure is cheap, so the description happens on this side of the joint.
    """
    from piper.download_voices import download_voice

    models_dir.mkdir(parents=True, exist_ok=True)
    ready: dict[str, _Ready] = {}

    for key in keys:
        model_path = models_dir / f"{key}.onnx"
        # Both halves, not just the weights. A voice is an .onnx and the
        # .onnx.json beside it that `_describe` reads, and an interrupted
        # download can leave one without the other — a killed container, a full
        # disk, a bind mount that received a partial copy. Checking only the
        # weights treats that as installed and defers the failure to `_describe`
        # or, worse, to the first synthesis, instead of re-fetching.
        config_path = models_dir / f"{key}.onnx.json"
        if not (model_path.exists() and config_path.exists()):
            if not allow_download:
                raise FileNotFoundError(
                    f"voice {key!r} is not completely installed in {models_dir} "
                    f"(need both {model_path.name} and {config_path.name}) "
                    f"and downloading is off"
                )
            _LOGGER.info("downloading voice %s into %s", key, models_dir)
            download_voice(key, models_dir)
            # Checked again after the call, not only before it. `download_voice`
            # reports success by returning, and a half-written pair is the same
            # realistic outcome the check above exists for — an interrupted
            # write, a full disk. Without this the gap surfaces as a bare
            # FileNotFoundError from `_describe` opening the sidecar, which names
            # the missing file but not the download that failed to make it.
            if not (model_path.exists() and config_path.exists()):
                raise FileNotFoundError(
                    f"downloading voice {key!r} into {models_dir} did not produce "
                    f"both {model_path.name} and {config_path.name}"
                )
        voice, sample_rate = _describe(key, model_path, serves)
        ready[key] = _Ready(
            voice=voice, sample_rate=sample_rate, model_path=model_path
        )

    return ready


def _stream(
    model: "PiperVoice", text: str, prosody: engine.Prosody
) -> "Iterator[bytes]":
    """Piper's raw samples, one chunk at a time, as they are produced."""
    for chunk in model.synthesize(text, syn_config=_synthesis_config(prosody)):
        yield chunk.audio_int16_bytes


def _synthesis_config(prosody: engine.Prosody):
    """[`engine.Prosody`] in the shape Piper takes it.

    No speaker_id. Piper's multi-speaker models take an index, and there is no
    ElevenLabs body field to source one from — so the knob was declared,
    forwarded, and set by nothing, which claims a capability this API does not
    have. Reaching it would mean inventing a request field only this server
    understands, and a field no ElevenLabs client will ever send is the opposite
    of what this service is for. A multi-speaker voice speaks as its default,
    which is what Piper does with no id.
    """
    from piper.config import SynthesisConfig

    # `length_scale` stretches audio, so it is the inverse of speed: a caller
    # asking for 2.0 wants each phoneme to last half as long.
    return SynthesisConfig(length_scale=1.0 / prosody.speed if prosody.speed else 1.0)


def _separates_words(phoneme: str) -> bool:
    """Whether this phoneme is a gap between words rather than part of one.

    The one place espeak's alphabet is interpreted. Downstream sees only the
    answer, so [`elvenspeak.alignment`] derives word boundaries without holding
    an opinion about any phonemizer's notation — it previously held this set
    itself, which made the module that is supposed to be engine-agnostic the
    owner of a Piper fact.
    """
    return phoneme.isspace() or phoneme in _BOUNDARY_PHONEMES


def _describe(
    key: str, model_path: Path, serves: frozenset[str]
) -> tuple[engine.Voice, int]:
    """Reads a voice's metadata, and the rate its samples will really have.

    From the `.onnx.json` beside the weights rather than from the remote catalog,
    because that file is what the loaded model actually runs on.

    The rate is returned separately rather than on the [`engine.Voice`]: it is a
    fact about the audio a synthesis produces, and every result carries it, so
    putting it on the voice as well would be a second copy of it that an engine
    with a per-utterance rate could not keep true.
    """
    with (model_path.parent / f"{key}.onnx.json").open(encoding="utf-8") as handle:
        config = json.load(handle)

    parts = key.split("-")
    # `or {}` throughout rather than a default argument: `.get(name, {})`
    # substitutes only for an absent key, so an explicit null in a hand-edited or
    # half-written sidecar returned None and the chained lookup raised
    # AttributeError — instead of the key-derived fallback these expressions
    # already promise. Read into locals so each section is spelled once; the two
    # spellings of `audio` were how they came to disagree.
    audio = config.get("audio") or {}
    language = (config.get("language") or {}).get("code") or (
        parts[0] if len(parts) == _KEY_PARTS else key
    )
    name = config.get("dataset") or (parts[1] if len(parts) == _KEY_PARTS else key)
    quality = audio.get("quality") or (
        parts[2] if len(parts) == _KEY_PARTS else "medium"
    )

    # The one field with no fallback. Everything else can be recovered from the
    # voice key, but a guessed sample rate is a wrong-pitch answer that plays
    # perfectly — the silent wrong answer this service refuses elsewhere — so a
    # sidecar that does not state it is a voice that cannot be served, named here
    # rather than as a bare KeyError from inside a dict lookup.
    # Falsy as well as absent: a sidecar carrying `"sample_rate": 0` passed an
    # `is None` check and stored a zero, which does not fail here at all — it
    # fails as a ZeroDivisionError inside `align`'s seconds-per-sample, at request
    # time on the timestamp endpoints, which is the deferred and misplaced failure
    # this check exists to replace.
    rate = audio.get("sample_rate")
    if not rate or int(rate) <= 0:
        raise ValueError(
            f"voice {key!r} has no positive audio.sample_rate in its .onnx.json "
            f"(found {rate!r}); the rate its samples will have cannot be inferred"
        )

    return (
        engine.Voice(
            # Piper's own identifier doubles as the voice_id a caller names, so a
            # client that reads `GET /v1/voices` and echoes an id back always
            # names something real.
            id=key,
            name=name,
            description=f"Piper {name} ({language}, {quality})",
            # The Piper facts with no ElevenLabs field of their own. Carried
            # rather than dropped: a caller choosing a voice wants the quality
            # tier, and `speakers` says out loud what a listener would otherwise
            # discover — there is no ElevenLabs field to select a speaker with, so
            # a multi-speaker model always speaks as its default.
            labels=(
                ("language", language),
                ("quality", quality),
                ("engine", "piper"),
                ("speakers", str(int(config.get("num_speakers") or 1))),
            ),
            models=serves,
        ),
        int(rate),
    )
