"""Kokoro-82M, as one engine behind [`elvenspeak.engine`]'s interface.

Everything Kokoro-shaped in this service lives here: the ONNX export, the voice
style pack, espeak's phoneme alphabet, the `af_heart` voice namespace, the asset
download, and the environment variables that name all of it. Nothing outside
this module names any of it — which is the claim this module exists to test,
because it was written after the seam rather than alongside it.

This module is `elvenspeak.kokoro`; the library it wraps is the top-level
`kokoro_onnx`. Imports here are absolute, so they reach the library and never
this file.

# What this engine is for

It runs at an RTF of ~0.77 against Piper's ~0.03, and sounds considerably
better. It is a deployment's choice, not one to inherit, which is why
[`elvenspeak.engines`] lists it second.

It also has a different voice namespace: `af_heart`, `am_michael`, `bf_emma`,
not `<lang>-<name>-<quality>`. Nothing above the seam had to learn that, which is
what "a voice id is opaque to the server" means when it is true rather than
asserted.

# Why the timestamp capability is read off the session

Whether Kokoro can report durations is a property of the *export file*, not of
Kokoro: `kokoro_onnx` decides it as `"duration" in session.get_outputs()`, and
the two published exports differ. The `model-files-v1.0` export emits only
`audio`; every `model-files-v1.1` export emits `waveform` and `duration`. So a
deployment running the older export genuinely cannot place phonemes in time, and
one running the newer one genuinely can.

[LAW:one-source-of-truth] The capability is therefore derived from the opened
session and never from a constant here. A hard-coded "Kokoro cannot do
timestamps" would be a second map of a fact the session already holds — true for
one export, a lie for the other, and the lie is the expensive direction: the
server would report timings it never measured. This is the same shape as Piper's
`include_alignments`, decided once at load time for the same reason.

# Why the assets are on disk and opened before anything is served

The export is ~110 MB and the style pack ~28 MB. Fetching or opening either
inside a request charges an unbounded, silent delay to whichever caller happened
to be first, on the event loop. [`_Prepared.open`] therefore has the session
built and every configured voice checked before it returns, so a missing or
truncated asset is one clean failure to boot.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from . import engine
from .provisioning import ConfigError, flag

if TYPE_CHECKING:  # pragma: no cover - import cost is real, the symbol is not
    from collections.abc import Iterator, Mapping

    from kokoro_onnx import Kokoro

_LOGGER = logging.getLogger("elvenspeak.kokoro")

#: Where the published assets come from — pinned, not configurable;
#: `KOKORO_MODEL` names an export within it. `model-files-v1.1` because its
#: exports carry the `duration` output, so a default deployment can answer the
#: timestamp endpoints — see this module's header.
_RELEASE = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"
)

#: The style pack every voice is read from — one file for all 54, so offering
#: more voices costs no more disk and no more memory.
_VOICES_FILE = "voices-v1.0.bin"

#: The export installed when none is named. Quantized to int8 because it is
#: 114 MB against the full precision export's 325 MB, on a service that is CPU
#: bound and already the slow option; an operator wanting the full export names
#: it in `KOKORO_MODEL`.
DEFAULT_MODEL = "kokoro-v1.0.int8.onnx"

#: The voices offered when none are named, best first — the order is
#: load-bearing, see [`KokoroEngine.voices`]. `af_heart` leads because it is the
#: pack's reference voice. Four rather than all 54 so that a default deployment
#: proves every voice it offers at boot without paying for fifty it will not use;
#: any of the 54 can be named instead.
DEFAULT_VOICES = ("af_heart", "am_michael", "bf_emma", "bm_george")

#: A voice id's first character names the language it was trained to speak, and
#: the phonemizer has to be told which one or an `ef_dora` reads Spanish text
#: with English phonemes and is merely fluent-sounding nonsense. Values are
#: espeak's own codes: Mandarin is `cmn`, not `zh`, which the backend refuses.
_VOICE_LANGUAGES = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt-br",
    "z": "cmn",
}

#: A voice id's second character. Carried into the labels because a caller
#: choosing a voice wants it and ElevenLabs publishes `labels` as an open map.
_VOICE_GENDERS = {"f": "female", "m": "male"}

#: Phonemes that mark a gap between words rather than a sound inside one. Space
#: is the separator espeak emits between words; the punctuation marks are the
#: pauses it holds time for. The one place Kokoro's alphabet is interpreted, so
#: that [`elvenspeak.alignment`] never holds an opinion about a phonemizer.
_BOUNDARY_PHONEMES = frozenset(",.;:!?…")

#: What Kokoro does whatever export it was given. The `speed` input is on every
#: published export, so the rate is always variable — unlike durations, which
#: only some exports emit and which are settled from the session below.
_INHERENT = frozenset({engine.Capability.SPEED})

#: How much audio [`KokoroEngine.speak`] hands over at a time. Chunked rather
#: than returned whole because the streaming encoder is pumped from this
#: iterator, and a single chunk is the one case where there are no chunk
#: boundaries for a consumer to get wrong.
_CHUNK_SAMPLES = 4096


class KokoroEngine:
    """Speech from a Kokoro export, opened and ready.

    Constructed by [`_Prepared.open`] rather than directly, because an instance
    holding a voice it cannot speak in would be the state this module is
    arranged to make unreachable.
    """

    def __init__(
        self,
        model: "Kokoro",
        installed: dict[str, engine.Voice],
        capabilities: frozenset[engine.Capability],
        sample_rate: int,
    ) -> None:
        self._model = model
        self._installed = installed
        self._capabilities = capabilities
        self._sample_rate = sample_rate

    def voices(self) -> tuple[engine.Voice, ...]:
        """Every configured voice, in the order the operator named them.

        Configured order rather than sorted, because the interface makes this
        order mean something: the first voice offered is what answers for an
        unknown id when the deployment named no fallback. Sorting for tidiness
        would silently hand every such deployment `af_alloy` instead of the
        voice its operator listed first.
        """
        return tuple(self._installed.values())

    def capabilities(self) -> frozenset[engine.Capability]:
        # Settled by `_Prepared.open` from the session that was really opened,
        # not recomputed from the filename that produced it: whether durations
        # can be reported is a property of the export's graph outputs, so the
        # opened session is the only thing that knows.
        return self._capabilities

    def speak(
        self, voice: engine.Voice, text: str, prosody: engine.Prosody
    ) -> engine.Speech:
        # The lookup is not a guard: it is how an id the server resolved becomes
        # a voice this engine really installed, and the rate is fixed for the
        # model. The generator is unstarted, so nothing is synthesized until the
        # samples are pulled and a caller that goes away has cost nothing.
        spoken = self._installed[voice.id]
        return engine.Speech(
            sample_rate=self._sample_rate,
            audio=_stream(self._model, spoken, text, prosody),
        )

    def speak_timed(
        self, voice: engine.Voice, text: str, prosody: engine.Prosody
    ) -> engine.TimedSpeech:
        spoken = self._installed[voice.id]
        audio, _, timings = self._model.create_timed(
            text,
            voice=spoken.id,
            speed=prosody.speed,
            lang=_language(spoken.id),
        )
        pcm = _pcm(audio)
        return engine.TimedSpeech(
            pcm=pcm,
            sample_rate=self._sample_rate,
            timings=_stretches(timings, len(pcm) // 2, self._sample_rate),
            # An export without a `duration` output returns no timings at all,
            # so this says exactly what happened rather than being read back off
            # the capability that gated the call. The server does not ask an
            # engine that withheld TIMESTAMPS; if it ever did, the answer would
            # be one unattributed stretch honestly marked, not invented numbers.
            measured=bool(timings),
        )


@dataclass(frozen=True)
class _Prepared:
    """Kokoro as this deployment configured it, before anything was fetched.

    Satisfies [`elvenspeak.provisioning.Prepared`]. Every field came out of
    [`configure`], so both methods take no arguments and there is no way to reach
    either of them with a value the environment check has not already seen.
    """

    keys: tuple[str, ...]
    models_dir: Path
    model: str
    #: Whether [`open`] may fetch an asset that is missing at boot. Not consulted
    #: by [`acquire`], which fetches by definition — see its docstring.
    allow_download: bool

    def acquire(self) -> tuple[engine.Voice, ...]:
        """Puts this engine's assets on disk, and says what they turned out to be.

        Downloading is unconditional here, and that is the whole difference
        between this method and [`open`]: the build is the moment a download is
        the right answer, so the lifecycle moment is carried by which method the
        caller reached for rather than by a flag both of them read.

        It opens the session as well as fetching, which Piper's equivalent
        deliberately does not — and the difference is a fact about the engines,
        not a disagreement about the interface. Piper pays one ~60 MB session
        per voice, so opening at bake time costs a minute and a gigabyte for
        nothing; Kokoro has exactly one session for all 54 voices, so opening it
        is seconds and proves the export is intact. Presence is not readability:
        an export that downloaded completely and is still not a loadable graph
        passes every file check there is, and the build is the last moment that
        failure is cheap.

        The session is discarded. What is wanted from it is that building it
        succeeded.
        """
        _, installed, _ = _open(
            self.keys, self.models_dir, self.model, allow_download=True
        )
        return tuple(installed.values())

    def open(self) -> KokoroEngine:
        """Opens the export and returns the engine that speaks every named voice."""
        model, installed, sample_rate = _open(
            self.keys, self.models_dir, self.model, self.allow_download
        )
        # Decided here, beside the session that was opened, rather than stored
        # as a setting the engine re-reads later: this is the only line that can
        # see what the export really offers, and an engine holding the filename
        # instead would keep answering for the filename.
        return KokoroEngine(
            model,
            installed,
            capabilities=_INHERENT
            | ({engine.Capability.TIMESTAMPS} if model.has_timings else set()),
            sample_rate=sample_rate,
        )


def configure(env: "Mapping[str, str]") -> _Prepared:
    """Reads Kokoro's own environment, or says everything wrong with it at once.

    [LAW:parse-dont-validate] The checkpoint for this engine. Nothing below holds
    a string out of `env`, and nothing above holds a `models_dir`: what crosses
    is a [`_Prepared`] that could not have been built before these checks ran.

    Every problem is collected rather than raised at the first, because this list
    is spliced into the server's own — an operator bringing the service up should
    not discover a bad voice name and then, one restart later, a bad port.
    """
    problems: list[str] = []

    keys = tuple(
        name.strip()
        for name in env.get("KOKORO_VOICES", ",".join(DEFAULT_VOICES)).split(",")
        if name.strip()
    )
    if not keys:
        problems.append("KOKORO_VOICES is empty; name at least one voice")

    # A voice id's first character selects the phonemizer, so an id this module
    # cannot read is one it would have to guess a language for — and a guessed
    # language is fluent-sounding nonsense that plays perfectly, which is the
    # silent wrong answer this service refuses. Named here rather than as an
    # IndexError from inside a lookup at synthesis time.
    problems += [
        f"KOKORO_VOICES names {key!r}, which is not "
        f"<language><gender>_<name> (for example af_heart)"
        for key in keys
        if not _is_voice_id(key)
    ]

    # Stripped and checked like everything else here. `KOKORO_MODELS_DIR=` is a
    # present key, so `get` returns "" rather than the default, `Path("")` is the
    # working directory, and the server then reads and writes ~140 MB of assets
    # wherever it happened to be launched from, having reported nothing.
    models_text = env.get("KOKORO_MODELS_DIR", "").strip()
    if "KOKORO_MODELS_DIR" in env and not models_text:
        problems.append("KOKORO_MODELS_DIR is empty; name a directory or unset it")
    models_dir = Path(models_text or str(Path(__file__).parent.parent / "models"))

    model = env.get("KOKORO_MODEL", DEFAULT_MODEL).strip()
    if not model:
        problems.append("KOKORO_MODEL is empty; name an export or unset it")
        model = DEFAULT_MODEL

    try:
        allow_download = flag(env, "KOKORO_ALLOW_DOWNLOAD", default=True)
    except ValueError as error:
        problems.append(str(error))
        allow_download = True

    if problems:
        raise ConfigError(problems)

    return _Prepared(
        keys=keys,
        models_dir=models_dir,
        model=model,
        allow_download=allow_download,
    )


def _is_voice_id(key: str) -> bool:
    """Whether `key` is a voice id this module can read a language out of."""
    prefix, _, name = key.partition("_")
    return bool(
        name
        and len(prefix) == 2
        and prefix[0] in _VOICE_LANGUAGES
        and prefix[1] in _VOICE_GENDERS
    )


def _install(
    keys: tuple[str, ...], models_dir: Path, model: str, allow_download: bool
) -> dict[str, engine.Voice]:
    """Makes the assets present and every named voice readable out of them.

    [LAW:decomposition] The joint between [`_Prepared.acquire`] and
    [`_Prepared.open`]. Both want the assets on disk and every configured voice
    proved to exist in the pack; they differ only in what they do afterwards,
    and in whether a missing asset may be fetched.

    The style pack is read here rather than only by the caller that opens a
    session. Stopping at "the file exists" would let a truncated download — the
    interrupted-write case the checks below exist for — produce a green image
    that failed at container startup instead.
    """
    import numpy

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = _fetch(models_dir / model, f"{_RELEASE}/{model}", allow_download)
    voices_path = _fetch(
        models_dir / _VOICES_FILE, f"{_RELEASE}/{_VOICES_FILE}", allow_download
    )

    # The pack is a mapping of id to style vector, so this both proves the file
    # parses and answers whether each configured voice is really in it. A voice
    # named but absent is caught here, at the build, rather than as a KeyError
    # inside the first request that asked for it.
    with numpy.load(voices_path) as pack:
        missing = [key for key in keys if key not in pack]
        offered = sorted(pack.files)
    if missing:
        raise ValueError(
            f"{voices_path} has no voice named {', '.join(repr(k) for k in missing)}; "
            f"it offers {len(offered)}, including {', '.join(offered[:4])}"
        )

    _LOGGER.info("kokoro assets ready in %s (%s)", models_dir, model_path.name)
    return {key: _describe(key) for key in keys}


def _open(
    keys: tuple[str, ...], models_dir: Path, model: str, allow_download: bool
) -> tuple["Kokoro", dict[str, engine.Voice], int]:
    """The opened session, the voices it will speak in, and the rate it speaks at.

    The rate is taken from `kokoro_onnx` rather than restated here. It is the
    library's own constant — the same value it returns beside the samples from
    every `create` — and a copy of it in this module would be free to disagree
    with the audio it is labelling, which is unfixable from the outside: wrong
    pitch and wrong duration, no error anywhere. Read here rather than at import
    because this is where the library is already being paid for.
    """
    from kokoro_onnx import SAMPLE_RATE, Kokoro

    installed = _install(keys, models_dir, model, allow_download)
    _LOGGER.info("opening kokoro export %s", model)
    return (
        Kokoro(str(models_dir / model), str(models_dir / _VOICES_FILE)),
        installed,
        SAMPLE_RATE,
    )


def _fetch(path: Path, url: str, allow_download: bool) -> Path:
    """Makes one published asset present, and says where it is.

    The URL is the caller's to supply rather than derived from `path.name`,
    because an asset's name does not determine where it comes from: the two
    published releases both contain a `kokoro-v1.0.int8.onnx`, and they differ
    in whether its graph reports durations. Callers own which release they mean;
    this owns downloading it exactly once and refusing anything short.

    Idempotent: a file already there is left alone, so a rebuild over a warm
    cache re-fetches nothing.

    Downloaded to a neighbouring temporary name and renamed into place, because
    a rename within one directory is atomic. Writing the final path directly
    would leave a killed container or a full disk holding a half-written asset
    that every later run treats as installed — the failure the existence check
    above would otherwise wave through.
    """
    if path.exists():
        return path
    if not allow_download:
        raise FileNotFoundError(
            f"kokoro asset {path.name!r} is not installed in {path.parent} "
            f"and downloading is off"
        )

    _LOGGER.info("downloading %s into %s", path.name, path.parent)
    partial = path.with_name(f"{path.name}.partial")
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
            # Copied in blocks rather than read whole: the full-precision export
            # is 325 MB, and reading it into memory to write it back out is a
            # gratuitous spike on a build host.
            while block := response.read(1 << 20):
                handle.write(block)

        # Judged before the rename, not after it. `urlopen` reports success by
        # returning, and an empty or truncated body is a realistic outcome — a
        # proxy error page, a withdrawn asset, a full disk. Checked afterwards,
        # the empty file is already at the name every later run tests for
        # existence, so the build fails once and then succeeds forever on a file
        # that cannot be opened: the rename is the commit, so nothing may reach
        # it unjudged.
        if not partial.stat().st_size:
            raise OSError(f"downloading {url} produced an empty file")
        partial.rename(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def _language(key: str) -> str:
    """The phonemizer language a voice id was trained for."""
    return _VOICE_LANGUAGES[key[0]]


def _describe(key: str) -> engine.Voice:
    """One voice as the API surface has to show it, read out of its id.

    There is no per-voice metadata to read: the style pack carries vectors, not
    descriptions, so the id is the only thing that knows. It is a total
    description — `af_heart` is American English, female, "Heart" — which is why
    [`configure`] refuses an id it cannot read rather than letting a guess
    through.
    """
    language, gender = _language(key), _VOICE_GENDERS[key[1]]
    name = key.partition("_")[2].replace("_", " ").title()
    return engine.Voice(
        # Kokoro's own identifier doubles as the voice_id a caller names, so a
        # client that reads `GET /v1/voices` and echoes an id back always names
        # something real.
        id=key,
        name=name,
        description=f"Kokoro {name} ({language}, {gender})",
        # The Kokoro facts with no ElevenLabs field of their own.
        labels=(
            ("language", language),
            ("gender", gender),
            ("engine", "kokoro"),
        ),
    )


def _stream(
    model: "Kokoro", voice: engine.Voice, text: str, prosody: engine.Prosody
) -> "Iterator[bytes]":
    """Kokoro's samples, in pieces small enough for the encoder to be pumped with.

    The whole utterance is synthesized before the first piece is yielded, which
    is a real cost on the slow engine and is stated rather than hidden: the
    library streams only through an async generator, and driving one from this
    synchronous method would mean standing up an event loop per pull. The
    endpoints are unaffected in what they return — `/stream` sends the same
    bytes in the same order — and pay in latency to the first byte.
    """
    audio, _ = model.create(
        text,
        voice=voice.id,
        speed=prosody.speed,
        lang=_language(voice.id),
    )
    pcm = _pcm(audio)
    step = _CHUNK_SAMPLES * 2
    for start in range(0, len(pcm), step):
        yield pcm[start : start + step]


def _pcm(audio) -> bytes:
    """Kokoro's float samples as the signed 16-bit mono the seam carries.

    Clipped before scaling. A model output above 1.0 wraps around on conversion
    — a loud sample becoming a maximally negative one — which is an audible
    click that no test of lengths or rates would ever catch.
    """
    import numpy

    return (numpy.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _stretches(timings, total: int, rate: int) -> tuple[engine.Timing, ...]:
    """Kokoro's per-phoneme spans as stretches accounting for every sample.

    [`engine.TimedSpeech`] requires the durations to sum to the audio's sample
    count, and Kokoro reports neither of the two things that would give that for
    free: its spans are floating-point seconds, and they cover only the phonemes
    — the lead-in before the first and the run-out after the last belong to no
    phoneme at all.

    Both are handled by walking a cursor. Each boundary is rounded to samples
    once and the durations are taken as differences between consecutive
    boundaries, so the rounding telescopes and the sum is exact by construction;
    rounding each duration on its own would drift by a sample per phoneme and
    leave a timeline that slowly parts company with the audio. Audio the cursor
    has not reached is emitted as a separator, which is what the lead-in and
    run-out are — the same account Piper gives of espeak's `^` and `$`.
    """
    stretches: list[engine.Timing] = []
    cursor = 0
    for timing in timings:
        for boundary, separates in (
            (round(timing.start * rate), True),
            (round(timing.end * rate), _separates_words(timing.phoneme)),
        ):
            edge = min(max(boundary, cursor), total)
            if edge > cursor:
                stretches.append(
                    engine.Timing(samples=edge - cursor, separates_words=separates)
                )
                cursor = edge

    if cursor < total:
        stretches.append(
            engine.Timing(samples=total - cursor, separates_words=True)
        )
    return tuple(stretches)


def _separates_words(phoneme: str) -> bool:
    """Whether this phoneme is a gap between words rather than part of one.

    The one place Kokoro's alphabet is interpreted. Downstream sees only the
    answer, so [`elvenspeak.alignment`] derives word boundaries without holding
    an opinion about any phonemizer's notation.
    """
    return phoneme.isspace() or phoneme in _BOUNDARY_PHONEMES
