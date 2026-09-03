"""Chatterbox Multilingual, as one engine behind [`elvenspeak.engine`]'s interface.

Everything Chatterbox-shaped in this service lives here: the Hugging Face
checkpoint set, the reference audio a speaker is cloned from, the 23 languages
the model was trained on, the accelerator it has to run on, and the environment
variables that name all of it. Nothing outside this module names any of it.

This module is `elvenspeak.chatterbox`; the library it wraps is the top-level
`chatterbox`. Imports here are absolute, so they reach the library and never this
file.

# What this engine is for, and why it is not a third of the same thing

Piper and Kokoro both have fixed voice packs, and their per-language voices are
different *people*: `es_ES-davefx-medium` and `en_US-lessac-high` are two
speakers, so a sentence that changes language mid-way changes who is talking.
Chatterbox clones one speaker from reference audio and then speaks any of its 23
languages in that speaker's voice, which makes the language a property of
pronunciation rather than of identity. That is the only reason this engine is
worth what it costs, and the cost is not small — see the header of `_Prepared`.

Nothing multilingual happens *here*. A request names one voice, a voice speaks
one language, and this engine renders one language at a time like the other two.
What it adds is that two of its voices can be the same person.

# What a voice is when the model clones one

[LAW:types-are-the-program] A voice id is `<speaker>-<language>` — `builtin-en`,
`builtin-es` — and the voice set is the product of the configured speakers and
the configured languages. That falls out of the two facts either side of the
seam and is not a scheme invented for tidiness:

[`Voice.language`] is required and singular, because it is compared against one
ElevenLabs `language_code`. A Chatterbox speaker is not singular in that way; it
speaks 23. A voice per speaker would therefore have to state one language and lie
about the other 22, and `Catalog.speaking` would match it for one caller and miss
it for the rest. A voice per (speaker, language) states something true of each.

It costs nothing to say it this way. Piper pays ~60 MB of ONNX session per voice,
so its voice list is a budget; here one model speaks every voice on offer, so the
languages are a list in the environment and adding one buys a voice for the price
of a string.

And it is the arrangement that makes this engine's whole point legible in `GET
/v1/voices`: two ids differing only in their language half are *the same person*,
which is a sentence Piper's catalogue cannot express at all.

`Voice.id` must be stable across restarts — a client reads the listing and echoes
an id back — so a speaker is never a request-time upload. A speaker is an asset:
either the model's own bundled `conds.pt`, which ships inside the checkpoint set
and is always available, or a reference `.wav` baked into the image beside the
model files. Both are present before the port is bound, which is also what
[`Engine.voices`]' "now, not eventually" requires.

# Why this engine declares no capabilities at all

Neither [`Capability.SPEED`] nor [`Capability.TIMESTAMPS`], and both absences are
facts about the model rather than caution.

`generate` has no rate parameter. `cfg_weight`, `temperature` and `exaggeration`
change how the sampler behaves and the pacing moves as a side effect, which is
not the thing `voice_settings.speed` promises — 2.0 is meant to be twice as fast,
not differently delivered. An engine that declared SPEED on the strength of that
would have the ignored header report the speed as honoured while the audio
disagreed, and `test_conformance` states the property as an equivalence precisely
so both directions of that lie fail.

Nothing in this model measures durations. Piper reports real per-phoneme spans
because [`elvenspeak.alignment`] patches its ONNX graph to expose them; Chatterbox
emits a waveform from speech tokens and accounts for nothing. So the timestamp
endpoints refuse, which is the designed behaviour — a fabricated alignment is
worse than a 501, because a caption renderer has no way to tell it from a real
one.

# The measurements this module was designed around

Measured, on the hardware named, with `chatterbox-tts` 0.1.7 and torch 2.6.0.
RTF is real time factor, so 1.0 is "as long to make as to play"; Piper is ~0.03
and Kokoro ~0.77 on this class of machine.

    device                          RTF, warm     resident
    RTX 2070 (CUDA, fp32)           0.76 - 1.09   3.2 GiB VRAM + 3.5 GiB host
    Apple M-series GPU (MPS, fp32)  2.33 - 4.30   ~4.8 GiB unified
    CPU (12 cores)                  7.62 - 33.3   4.8 GiB, 6.8 GiB peak

Three things follow, and each of them is a decision in the code below.

CPU IS NOT A FALLBACK. An order of magnitude past useful is not a degraded
service, it is a different one: "Yes." took 22.7 seconds. So the device is named
by the deployment and never chosen for it — see [`configure`].

MORE CORES DO NOT HELP. 12 threads measured RTF 9.96-30.9 against 4 threads'
7.62-33.3. The bottleneck is an autoregressive sampling loop running at ~4.7
tokens/s, which is latency, not throughput.

HALVING THE WEIGHTS IS NOT AVAILABLE. bfloat16 would halve the resident set, and
on CUDA it fails with `mat1 and mat2 must have the same dtype` because the
conditionals stay float32; casting those too, float16 on MPS aborts the process
inside Metal with `'mps.add' op requires the same element type for all operands`.
Both are upstream shapes rather than something this module can hold, and the
Turing and Zen 2 hardware in reach has no bfloat16 unit anyway.

SHORT UTTERANCES ARE FINE, which is the one measurement that came back better
than expected. Kokoro has a zero-sample defect on short text — `ef_dora` returned
nothing for 15 of 16 one- and two-word Spanish lines, which is why no Kokoro
Spanish voice is baked. Chatterbox produced healthy audio for every one of
`Yes.`, `Go on.`, `Si.` and `?Y tu?` in both languages, on CPU, on CUDA and on
MPS, cold and warm. Whatever eventually renders mixed language will be feeding
this engine short spans, so this is the property that mattered most.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from . import engine
from .provisioning import ConfigError, flag

if TYPE_CHECKING:  # pragma: no cover - import cost is real, the symbol is not
    from collections.abc import Iterator, Mapping

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals

_LOGGER = logging.getLogger("elvenspeak.chatterbox")

#: The published checkpoint set, pinned to a commit rather than to `main`.
#:
#: [LAW:no-ambient-temporal-coupling] `chatterbox.mtl_tts.from_pretrained` passes
#: `revision="main"`, which makes two builds of one commit different artifacts and
#: turns an upstream re-upload into a change nobody in this repository asked for.
#: The whole reason images are built in CI from a commit is that the artifact's
#: source can be named; a floating model revision gives that back.
_REPO = "ResembleAI/chatterbox"
_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"

#: Exactly the files the multilingual model opens, and no others. The repository
#: holds 13.2 GiB across several generations of export — `t3_cfg`, `t3_23lang`,
#: `s3gen_v3`, `t3_mtl23ls_v3` — of which this model reads 3.06 GiB. Named rather
#: than mirrored, because a full snapshot would quadruple the image for exports
#: nothing loads.
_ASSETS = (
    "ve.pt",
    "t3_mtl23ls_v2.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
)

#: The speaker that needs no reference audio: the model ships precomputed
#: conditionals for one voice as `conds.pt`, inside the checkpoint set above. It
#: is always available and is the default, so an image that bakes no reference
#: `.wav` still offers a working voice per configured language.
BUILTIN_SPEAKER = "builtin"

#: Where a cloned speaker's reference audio lives, under the models directory.
#: One `<speaker>.wav` per speaker, baked at build time like any model file
#: because [`engine.Voice`] requires ids to be stable across restarts.
REFERENCES_DIR = "chatterbox-references"

#: The languages offered when none are named. English and Spanish because that
#: is the pair openconv's language-learning work speaks, and because they are the
#: two this engine has actually been heard in — the other 21 are reachable by
#: naming them and have not been listened to.
DEFAULT_LANGUAGES = ("en", "es")

#: What this engine can do beyond plain synthesis, which is nothing. Kept as a
#: named empty set rather than written inline at each use, so that the day the
#: model gains a rate control there is one place that stops being empty.
_INHERENT: frozenset[engine.Capability] = frozenset()

#: How much audio [`ChatterboxEngine.speak`] hands over at a time. Chunked rather
#: than returned whole because the streaming encoder is pumped from this iterator.
_CHUNK_SAMPLES = 4096

#: Every accelerator this engine will run on, by torch's own name for it.
#:
#: A closed set rather than anything torch accepts, because the point of naming a
#: device is to be refused when the deployment and the hardware disagree, and
#: `torch.device` accepts strings this engine has never been measured on. `cpu` is
#: in the set and is deliberately not the default — see [`configure`].
DEVICES = ("cuda", "mps", "cpu")

#: Environment variables this engine parses. Named here so `configure` and its
#: failure messages spell each one once.
MODELS_DIR = "CHATTERBOX_MODELS_DIR"
DEVICE = "CHATTERBOX_DEVICE"
SPEAKERS = "CHATTERBOX_SPEAKERS"
LANGUAGES = "CHATTERBOX_LANGUAGES"
ALLOW_DOWNLOAD = "CHATTERBOX_ALLOW_DOWNLOAD"


@dataclass(frozen=True)
class _Spoken:
    """One voice on offer, and everything needed to speak in it.

    [LAW:decomposition] The voice and the two values synthesis needs travel
    together because they are one fact — this voice is this speaker saying this
    language — and splitting them into parallel maps keyed by id is three
    dictionaries that can disagree about which speaker `builtin-es` is.

    `conditionals` is shared between every voice of one speaker: cloning is per
    speaker, not per language, and computing it once is the whole economy of
    offering 23 languages for the price of one model.
    """

    voice: engine.Voice
    #: The speaker's cloned identity, already on the device and ready to speak.
    conditionals: "Conditionals"
    #: The language tag `generate` wants, which is ISO 639-1 and so is the same
    #: spelling [`engine.Voice.language`] holds. Carried rather than sliced back
    #: out of the id, because an id is a name and this is a parameter.
    language: str


class ChatterboxEngine:
    """Speech from an opened Chatterbox model, ready in every voice it offers.

    Constructed by [`_Prepared.open`] rather than directly, because an instance
    holding a voice whose conditionals were never computed would be the state
    this module is arranged to make unreachable.

    [LAW:no-shared-mutable-globals] This engine has a lock and the other two do
    not, and the difference is a fact about the library rather than a precaution.
    `ChatterboxMultilingualTTS.generate` reads the speaker off `self.conds`, so
    selecting a voice means *writing to the model object*. The server calls
    engines off the event loop, so two requests naming two voices can be inside
    `speak` at once — and without this lock the second one's write lands between
    the first one's write and its read, and the first caller is answered in
    somebody else's voice. That failure is inaudible as a failure: the audio is
    fluent, it is simply the wrong person. The model has one owner, which is this
    object, and every write to it happens here.

    Serialising synthesis costs nothing that was available anyway. The model
    saturates one accelerator, so two concurrent syntheses were never going to
    run in parallel — they would interleave on the same device and each finish
    later.
    """

    def __init__(
        self,
        model: "ChatterboxMultilingualTTS",
        spoken: dict[str, _Spoken],
        sample_rate: int,
    ) -> None:
        self._model = model
        self._spoken = spoken
        self._sample_rate = sample_rate
        self._speaking = threading.Lock()

    def voices(self) -> tuple[engine.Voice, ...]:
        """Every configured voice, speakers in the order the operator named them.

        Configured order rather than sorted, because the interface makes this
        order mean something: the first voice offered is what answers for an
        unknown id when the deployment named no fallback. Sorting for tidiness
        would silently hand every such deployment whichever language sorts first.
        """
        return tuple(item.voice for item in self._spoken.values())

    def speak(
        self, voice: engine.Voice, text: str, prosody: engine.Prosody
    ) -> engine.Speech:
        # The lookup is not a guard: it is how an id the server resolved becomes
        # a voice this engine really prepared. The rate is fixed for the model.
        # The generator is unstarted, so nothing is synthesized until the samples
        # are pulled and a caller that goes away has cost nothing.
        #
        # `prosody` is read by nothing here, which is what declaring no
        # capabilities means: the server has already neutralised every field this
        # voice did not claim, so `speed` arrives at 1.0 and honouring it would be
        # honouring a value the caller was told was ignored.
        spoken = self._spoken[voice.id]
        return engine.Speech(
            sample_rate=self._sample_rate, audio=self._stream(spoken, text)
        )

    def speak_timed(
        self, voice: engine.Voice, text: str, prosody: engine.Prosody
    ) -> engine.TimedSpeech:
        """Synthesizes `text` in one piece, measuring none of it.

        Reachable only by a caller that went around the capability gate, since no
        voice here declares TIMESTAMPS and the server does not ask an engine that
        did not. It answers honestly rather than raising, because the honest
        answer exists: the audio is real, and `measured=False` with the whole
        utterance recorded as one separator says exactly what happened — every
        sample accounted for, none of it attributed. Inventing boundaries to fill
        the tuple is the one thing that must not happen here.
        """
        pcm = self._synthesized(self._spoken[voice.id], text)
        samples = len(pcm) // 2
        return engine.TimedSpeech(
            pcm=pcm,
            sample_rate=self._sample_rate,
            # [`TimedSpeech`] requires the durations to sum to the sample count,
            # and an empty utterance sums to zero with no stretches at all.
            timings=(
                (engine.Timing(samples=samples, separates_words=True),)
                if samples
                else ()
            ),
            measured=False,
        )

    def _stream(self, spoken: _Spoken, text: str) -> "Iterator[bytes]":
        """The samples, in pieces small enough for the encoder to be pumped with.

        The whole utterance is synthesized before the first piece is yielded,
        which is a real cost on a slow engine and is stated rather than hidden:
        `generate` returns a finished waveform and has no incremental door. The
        endpoints are unaffected in what they return — `/stream` sends the same
        bytes in the same order — and pay in latency to the first byte.
        """
        pcm = self._synthesized(spoken, text)
        step = _CHUNK_SAMPLES * 2
        for start in range(0, len(pcm), step):
            yield pcm[start : start + step]

    def _synthesized(self, spoken: _Spoken, text: str) -> bytes:
        """One utterance's samples, as the signed 16-bit mono the seam carries.

        [LAW:single-enforcer] The one place this model is asked to speak, so the
        lock covering the model's mutable speaker selection is taken once and
        both entry points reach it. A second call site would be a second writer
        to `model.conds` and the lock would stop meaning anything.

        The assignment is safe to repeat and safe to leave in place: `generate`
        replaces `conds.t3` only when the `exaggeration` it was passed differs
        from the stored one, and it is passed neither here, so the default 0.5
        equals what every speaker was prepared with and nothing this engine holds
        is rewritten underneath it.
        """
        import torch

        with self._speaking:
            self._model.conds = spoken.conditionals
            wav = self._model.generate(text, language_id=spoken.language)

        # Clipped before scaling. A model output above 1.0 wraps around on
        # conversion -- a loud sample becoming a maximally negative one -- which
        # is an audible click that no test of lengths or rates would ever catch.
        # `.float()` first because clamping a half-precision tensor at 1.0 and
        # scaling by 32767 overflows the format it is still in.
        return (
            (wav.squeeze(0).float().clamp(-1.0, 1.0) * 32767.0)
            .to(torch.int16)
            .cpu()
            .numpy()
            .tobytes()
        )


@dataclass(frozen=True)
class _Prepared:
    """Chatterbox as this deployment configured it, before anything was fetched.

    Satisfies [`elvenspeak.provisioning.Prepared`]. Every field came out of
    [`configure`], so both methods take no arguments and there is no way to reach
    either of them with a value the environment check has not already seen.

    Both methods open the model, which Piper's equivalent deliberately does not,
    and the difference is a fact about the engines. Piper pays one ~60 MB session
    per voice, so opening at bake time costs a minute and a gigabyte for nothing;
    Chatterbox has one model for every voice on offer, and opening it is the only
    way to learn that 3.06 GiB of checkpoints are a loadable model rather than
    3.06 GiB of bytes. Presence is not readability, and the build is the last
    moment that failure is cheap.
    """

    speakers: tuple[str, ...]
    languages: tuple[str, ...]
    models_dir: Path
    device: str
    #: Whether [`open`] may fetch an asset that is missing at boot. Not consulted
    #: by [`acquire`], which fetches by definition — see its docstring.
    allow_download: bool
    #: Every `model_id` this deployment answers to, stamped onto each voice beside
    #: its capabilities and for the same reason — it is a fact about what will
    #: speak. Arrives from [`configure`] because the name it was derived from is
    #: the key this module is registered under, which this module never learns.
    serves: frozenset[str]

    def acquire(self) -> tuple[engine.Voice, ...]:
        """Puts this engine's assets on disk, and says what they turned out to be.

        Downloading is unconditional here, and that is the whole difference
        between this method and [`open`]: the build is the moment a download is
        the right answer, so the lifecycle moment is carried by which method the
        caller reached for rather than by a flag both of them read.

        It clones every configured speaker as well as fetching, because a
        reference `.wav` that is present and unreadable is a container that boots
        3 GiB of model and then dies on the speaker it was built for.
        """
        return tuple(item.voice for item in self._open(allow_download=True)[1].values())

    def open(self) -> ChatterboxEngine:
        """Opens the model and returns the engine that speaks every named voice."""
        model, spoken = self._open(allow_download=self.allow_download)
        from chatterbox.models.s3gen import S3GEN_SR

        # Read off the library rather than restated here. It is the rate
        # `generate` really produces, and a copy of it in this module would be
        # free to disagree with the audio it is labelling — wrong pitch and wrong
        # duration, with no error anywhere.
        return ChatterboxEngine(model, spoken, sample_rate=S3GEN_SR)

    def _open(
        self, allow_download: bool
    ) -> tuple["ChatterboxMultilingualTTS", dict[str, _Spoken]]:
        """The opened model and every voice it will speak in.

        [LAW:decomposition] The joint between [`acquire`] and [`open`]. Both want
        the checkpoints present, the model loaded and every speaker cloned; they
        differ only in what they do with the result, and in whether a missing
        asset may be fetched.
        """
        from chatterbox.mtl_tts import SUPPORTED_LANGUAGES, ChatterboxMultilingualTTS

        checkpoints = _fetch(self.models_dir, allow_download)
        _LOGGER.info("opening chatterbox on %s from %s", self.device, checkpoints)
        model = ChatterboxMultilingualTTS.from_local(checkpoints, self.device)

        # Refused here rather than in `configure`, because the authority on which
        # languages exist is the library and `configure` must parse the
        # environment without importing it — see this module's registration in
        # `elvenspeak.engines`, and `tests/test_encoding.py`, which proves the
        # ElevenLabs surface cannot reach an engine library through configuration.
        unknown = [name for name in self.languages if name not in SUPPORTED_LANGUAGES]
        if unknown:
            raise ConfigError(
                [
                    f"{LANGUAGES} names {name!r}, which this model does not speak; "
                    f"it speaks {', '.join(sorted(SUPPORTED_LANGUAGES))}"
                    for name in unknown
                ]
            )

        spoken: dict[str, _Spoken] = {}
        for speaker in self.speakers:
            conditionals = _cloned(model, speaker, self.models_dir)
            for language in self.languages:
                item = _describe(speaker, language, conditionals, self.serves)
                spoken[item.voice.id] = item
        _LOGGER.info(
            "chatterbox ready: %d voices from %d speaker(s) in %s",
            len(spoken),
            len(self.speakers),
            ", ".join(self.languages),
        )
        return model, spoken


def configure(
    env: "Mapping[str, str]",
    withheld: frozenset[engine.Capability],
    serves: frozenset[str],
) -> _Prepared:
    """Reads Chatterbox's own environment, or says everything wrong with it at once.

    [LAW:parse-dont-validate] The checkpoint for this engine. Nothing below holds
    a string out of `env`, and nothing above holds a `models_dir`: what crosses
    is a [`_Prepared`] that could not have been built before these checks ran.

    Every problem is collected rather than raised at the first, because this list
    is spliced into the server's own — an operator bringing the service up should
    not discover a bad device and then, one restart later, a bad port.

    `withheld` is accepted and unused, and here that is not even a declined
    economy: this engine declares nothing to withhold. What the deployment
    withheld is enforced by the server, once, against what this engine declares,
    so ignoring the offer can never be the wrong answer.

    THE DEVICE HAS NO DEFAULT, which is this function's one real decision. Every
    candidate default is wrong somewhere and silently so: `cuda` refuses to boot
    on the Apple hardware this has been run on, and `cpu` boots everywhere and
    then serves at 10 to 33 times real time — measured, and an order of magnitude
    past useful. A default is a claim that one answer is right when the
    deployment says nothing, and there is no such answer here, so the deployment
    says something. This is the same reasoning [`engine.Voice.language`] and
    [`engine.Voice.models`] are required under: an omission that would be read as
    a real answer is worse than an omission that is refused.
    """
    problems: list[str] = []

    speakers = _named(env, SPEAKERS, (BUILTIN_SPEAKER,))
    if not speakers:
        problems.append(f"{SPEAKERS} is empty; name at least one speaker")

    languages = _named(env, LANGUAGES, DEFAULT_LANGUAGES)
    if not languages:
        problems.append(f"{LANGUAGES} is empty; name at least one language")

    # Stripped and checked like everything else here. `CHATTERBOX_MODELS_DIR=` is
    # a present key, so `get` returns "" rather than the default, `Path("")` is
    # the working directory, and the server then reads and writes 3 GiB of
    # checkpoints wherever it happened to be launched from, having reported
    # nothing.
    models_text = env.get(MODELS_DIR, "").strip()
    if MODELS_DIR in env and not models_text:
        problems.append(f"{MODELS_DIR} is empty; name a directory or unset it")
    models_dir = Path(models_text or str(Path(__file__).parent.parent / "models"))

    device = env.get(DEVICE, "").strip()
    if device not in DEVICES:
        problems.append(
            f"{DEVICE}={device or '(unset)'!s} is not one of {', '.join(DEVICES)}; "
            f"name the accelerator this deployment has. There is no default: cpu "
            f"runs this model at 10-33x real time and would boot anywhere"
        )

    try:
        allow_download = flag(env, ALLOW_DOWNLOAD, default=True)
    except ValueError as error:
        problems.append(str(error))
        allow_download = True

    if problems:
        raise ConfigError(problems)

    return _Prepared(
        speakers=speakers,
        languages=languages,
        models_dir=models_dir,
        device=device,
        allow_download=allow_download,
        serves=serves,
    )


def _named(env: "Mapping[str, str]", variable: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
    """One comma-separated setting as the list it names, blanks dropped.

    [LAW:one-source-of-truth] Both list settings are read through here rather
    than each spelling the same split. `CHATTERBOX_SPEAKERS=a,,b` and a trailing
    comma are the same typo, and two copies of this would be free to disagree
    about which of them is empty.
    """
    return tuple(
        name.strip()
        for name in env.get(variable, ",".join(fallback)).split(",")
        if name.strip()
    )


def _fetch(models_dir: Path, allow_download: bool) -> Path:
    """Makes the checkpoint set present, and says which directory holds it.

    `huggingface_hub` owns the download, the resume and the integrity check, so
    this is not the block-copying `_fetch` [`elvenspeak.kokoro`] needs for a plain
    URL: the hub writes through its own blob store and only links a file into
    place once it is complete, which is the same "nothing reaches the final name
    unjudged" guarantee, already implemented.

    Idempotent: a snapshot already present is verified and reused, so a rebuild
    over a warm cache re-fetches nothing.

    [LAW:no-silent-failure] `local_files_only` is how downloading is refused,
    rather than checking for the files first and then asking anyway. The hub
    raises when it cannot satisfy the request offline, which is the loud refusal
    a missing asset should be; a presence check of our own would be a second,
    weaker copy of what the hub already decides, and it would pass on a snapshot
    directory that exists and is short of a file.
    """
    from huggingface_hub import snapshot_download

    models_dir.mkdir(parents=True, exist_ok=True)
    if allow_download:
        _LOGGER.info("fetching chatterbox checkpoints into %s", models_dir)
    return Path(
        snapshot_download(
            repo_id=_REPO,
            revision=_REVISION,
            allow_patterns=list(_ASSETS),
            cache_dir=str(models_dir),
            local_files_only=not allow_download,
        )
    )


def _cloned(
    model: "ChatterboxMultilingualTTS", speaker: str, models_dir: Path
) -> "Conditionals":
    """One speaker's identity, computed once and shared by every voice of theirs.

    [`BUILTIN_SPEAKER`] is the identity the model ships with, already loaded from
    `conds.pt` as part of the checkpoint set. Every other name is a reference
    `.wav` baked beside the model files, and cloning it here — at build and at
    boot, never inside a request — is what makes its voice id stable and what
    keeps [`Engine.voices`]' "now, not eventually" true.

    `prepare_conditionals` returns nothing and writes `model.conds`, so the
    result is read back off the model and the model is left holding whichever
    speaker was cloned last. That is harmless because nothing reads `model.conds`
    except [`ChatterboxEngine._synthesized`], which writes it first under the
    lock — but it is why the value is captured here rather than left where the
    library put it.
    """
    if speaker == BUILTIN_SPEAKER:
        # The one speaker with no reference audio: `from_local` loaded `conds.pt`
        # into `model.conds` already. A checkpoint set without it would leave this
        # None, and the voices built from it would fail on their first request
        # rather than here, so it is refused at the moment it is missing.
        if model.conds is None:
            raise ConfigError(
                [
                    f"the chatterbox checkpoints carry no {BUILTIN_SPEAKER!r} voice "
                    f"(conds.pt is absent from {_REPO}@{_REVISION[:8]}); "
                    f"name reference speakers in {SPEAKERS} instead"
                ]
            )
        return model.conds

    reference = models_dir / REFERENCES_DIR / f"{speaker}.wav"
    if not reference.is_file():
        raise ConfigError(
            [
                f"{SPEAKERS} names {speaker!r}, so {reference} has to be a "
                f"reference recording of that speaker, and it is not there"
            ]
        )
    _LOGGER.info("cloning speaker %r from %s", speaker, reference)
    model.prepare_conditionals(str(reference))
    return model.conds


def _describe(
    speaker: str, language: str, conditionals: "Conditionals", serves: frozenset[str]
) -> _Spoken:
    """One voice as the API surface has to show it, plus what it takes to speak.

    There is no per-voice metadata to read — a cloned speaker is a waveform and a
    language is a two-letter tag — so the id is a total description, which is why
    it is also the thing composed rather than looked up.
    """
    identifier = f"{speaker}-{language}"
    return _Spoken(
        voice=engine.Voice(
            id=identifier,
            name=identifier,
            description=(
                f"Chatterbox {speaker} speaking {language} "
                f"(one speaker across every language offered)"
            ),
            # The Chatterbox facts with no ElevenLabs field of their own. `speaker`
            # is the one worth publishing: it is how a client can tell that two ids
            # are the same person, which is the whole reason this engine exists and
            # is not derivable from any other field.
            labels=(("speaker", speaker), ("engine", "chatterbox")),
            capabilities=_INHERENT,
            models=serves,
            language=language,
        ),
        conditionals=conditionals,
        language=language,
    )
