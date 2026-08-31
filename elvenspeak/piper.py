"""Piper, as one engine behind [`elvenspeak.engine`]'s interface.

Everything Piper-shaped in this service lives here: ONNX sessions, `.onnx.json`
sidecars, espeak's phoneme alphabet, `length_scale`, the voice download. Nothing
outside this module names any of it, which is what makes the ElevenLabs surface
above reusable with a different engine under it.

This module is `elvenspeak.piper`; the library it wraps is the top-level `piper`.
Imports here are absolute, so `from piper import ...` reaches the library and
never this file.

# Why the voices are on disk and loaded before anything is served

A Piper voice is a ~60 MB ONNX file, and building a session from one is seconds
of work. Doing either inside a request charges an unbounded, silent delay to
whichever caller happened to be first, on the event loop, stalling every other
request including `/health`. [`load`] therefore fetches, describes and opens
every voice before it returns, so a missing, truncated or corrupt model is one
clean failure to boot — and so [`PiperEngine.voices`] can promise what the
interface asks of it: these can be spoken *now*.

# Why alignment support is decided once for the process

`include_alignments` patches a voice's graph in memory at load time to expose
per-phoneme durations. It is a property of the session, not of a call, so a
request asking for timings against an unpatched session could only fail or lie.
Whether to pay for it is settled before any voice is opened.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from . import engine

if TYPE_CHECKING:  # pragma: no cover - import cost is real, the symbol is not
    from collections.abc import Iterator

    from piper import PiperVoice

_LOGGER = logging.getLogger("elvenspeak.piper")

#: Phonemes espeak emits that mark structure rather than sound — the run-up into
#: an utterance, the run-out, a line break. They consume real time but belong to
#: no word, which is exactly what [`engine.Timing.separates_words`] reports.
_BOUNDARY_PHONEMES = frozenset({"^", "$", "\n"})

# `<lang>-<name>-<quality>`, the shape every Piper voice key has. Used to read a
# key's parts back when the sidecar omits them — never to decide whether a voice
# exists, which only the files on disk can answer.
_KEY_PARTS = 3


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

    Constructed by [`load`] rather than directly, because an instance holding a
    voice it cannot speak in would be the state this whole module is arranged to
    make unreachable.
    """

    def __init__(self, installed: dict[str, _Installed], timings: bool) -> None:
        self._installed = installed
        self._timings = timings

    def voices(self) -> tuple[engine.Voice, ...]:
        return tuple(self._installed[key].voice for key in sorted(self._installed))

    def can_time(self) -> bool:
        # The flag the sessions were opened under, not a second copy of the
        # setting that produced it: `include_alignments` patches the graph at
        # load time, so this is the only place that knows whether these
        # particular sessions can report durations.
        return self._timings

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


def load(
    keys: tuple[str, ...],
    models_dir: Path,
    allow_download: bool,
    timings: bool,
) -> PiperEngine:
    """Puts every named voice on disk, opens it, and returns the engine.

    Runs before the server accepts a request, so that a deployment problem is a
    refusal to boot rather than an unbounded delay inside somebody's first call.
    """
    from piper import PiperVoice

    installed: dict[str, _Installed] = {}
    for key, model_path in install(keys, models_dir, allow_download).items():
        voice, sample_rate = _describe(key, model_path)
        _LOGGER.info("loading voice %s", key)
        installed[key] = _Installed(
            voice=voice,
            sample_rate=sample_rate,
            model=PiperVoice.load(str(model_path), include_alignments=timings),
        )

    return PiperEngine(installed, timings=timings)


def install(
    keys: tuple[str, ...], models_dir: Path, allow_download: bool
) -> dict[str, Path]:
    """Makes sure every named voice's files are on disk, and says where.

    Separate from [`load`] because the container image bakes voices in at build
    time and wants exactly this and no more — opening an ONNX session per voice
    only to discard it costs a minute and a gigabyte for nothing. A second caller
    that wants only half of a function is what a real joint looks like
    ([LAW:decomposition]).
    """
    from piper.download_voices import download_voice

    models_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

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
        paths[key] = model_path

    return paths


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


def _describe(key: str, model_path: Path) -> tuple[engine.Voice, int]:
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
        ),
        int(rate),
    )
