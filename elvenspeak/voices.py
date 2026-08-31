"""Which voice a request means, and the model that speaks it.

One question — "which voice is this?" — asked by three callers that must agree:
the synthesis endpoints, `GET /v1/voices`, and the alias table. They agree
because they all read this module and this module reads one catalog.

# Why an unknown voice ID substitutes instead of failing

Because clients written against ElevenLabs hold ElevenLabs voice IDs, and a
server that 404s every one of them is not a drop-in for anything. elvenreader-
server — this service's predecessor — resolved unrecognised IDs through a table
with a catch-all, so an unmapped ID came back as a successful response in a
substitute voice. Callers already depend on that: openconv passes Happy's stored
ElevenLabs voice ID straight through precisely because it is guaranteed a
response, and its own comments record that as the reason it keeps no voice table
of its own.

[LAW:no-silent-failure] Substituting is therefore contractual, not a swallowed
error — but it is still a case where the caller did not get what it named, so it
does not happen invisibly. [`Resolution.substituted`] says whether a swap
occurred, and every synthesis response carries an `x-elvenspeak-voice` header naming
what actually spoke. The behaviour a client depends on is preserved; the fact it
happened is not hidden.

# Why only installed voices are served

A Piper voice is a ~60 MB ONNX file. Fetching one on the first request that
names it would make that request pay an unbounded, silent delay, which is the
confusing failure this service already refuses at startup: voices are downloaded
when the process starts, and a voice that is not installed is not offered. So
`GET /v1/voices` lists what can be spoken *now*, not what could be fetched.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cost is real, the symbol is not
    from piper import PiperVoice

_LOGGER = logging.getLogger("elvenspeak.voices")

_ALIASES_FILE = Path(__file__).parent / "aliases.toml"

# `<lang>-<name>-<quality>`, the shape every Piper voice key has. Used to read a
# key's parts back when the downloaded catalog is unavailable — never to decide
# whether a voice exists, which only the catalog can answer.
_KEY_PARTS = 3


@dataclass(frozen=True)
class Voice:
    """A Piper voice this server has on disk and can speak.

    `key` is Piper's own identifier (`en_US-lessac-medium`) and doubles as the
    voice_id a caller can name directly — one identifier, so a client that reads
    `GET /v1/voices` and echoes an id back always names something real.
    """

    key: str
    name: str
    language: str
    quality: str
    model_path: Path
    sample_rate: int
    num_speakers: int

    def as_elevenlabs(self, aliases: tuple[str, ...] = ()) -> dict:
        """This voice in the shape `GET /v1/voices` returns.

        Field names are ElevenLabs', not Piper's, because the whole point is
        that an unmodified client can read the response. `labels` carries the
        Piper facts that have no ElevenLabs equivalent rather than dropping them
        — a caller choosing a voice wants the quality tier, and inventing a
        field for it would be worse than putting it where free-form data goes.
        """
        return {
            "voice_id": self.key,
            "name": self.name,
            "category": "premade",
            "labels": {
                "language": self.language,
                "quality": self.quality,
                "engine": "piper",
            },
            "description": f"Piper {self.name} ({self.language}, {self.quality})",
            "preview_url": None,
            "available_for_tiers": [],
            "high_quality_base_model_ids": [],
            "samples": None,
            "settings": None,
            "sharing": None,
            "fine_tuning": {
                "is_allowed_to_fine_tune": False,
                "state": {},
                "verification_failures": [],
                "verification_attempts_count": 0,
                "manual_verification_requested": False,
            },
            # Round-trips the alias table so a client can discover that the
            # ElevenLabs id it holds will reach this voice, instead of finding
            # out by trying it.
            "aliases": list(aliases),
        }


@dataclass(frozen=True)
class Resolution:
    """The voice that will speak, and whether it is the one that was asked for.

    Two fields rather than a bare `Voice`, because "you got what you named" and
    "you got a substitute" are different facts about the same successful
    response, and a caller that cannot tell them apart cannot report the second.
    """

    voice: Voice
    requested: str
    substituted: bool


class VoiceNotInstalled(LookupError):
    """A voice id that is neither installed nor aliased, with no fallback set.

    Only reachable when substitution is switched off. With a fallback
    configured — the default — resolution always succeeds.
    """

    def __init__(self, requested: str, installed: tuple[str, ...]) -> None:
        super().__init__(
            f"voice {requested!r} is not installed; "
            f"available: {', '.join(installed) or '(none)'}"
        )
        self.requested = requested


class Catalog:
    """The installed voices, and the one table that resolves an id onto them.

    Models are loaded lazily and kept: an ONNX session is tens of megabytes of
    weights and seconds of load time, so the first request for a voice pays for
    it and every later one does not. Loading is not guarded by a lock — under
    the ASGI worker model two concurrent first-requests would each build a
    session and one would win, which wastes a load and is otherwise harmless;
    a lock here would serialize every synthesis behind a dictionary lookup.
    """

    def __init__(
        self,
        voices: dict[str, Voice],
        fallback: str | None,
        include_alignments: bool,
        aliases: dict[str, str] | None = None,
    ) -> None:
        # [LAW:parse-dont-validate] Checked here, where the catalog is made,
        # rather than by whichever caller remembers to. `resolve()` indexes
        # `_voices[_fallback]` on its last branch, so a fallback naming no
        # installed voice turns every unrecognised id — the case the fallback
        # exists for — into a bare KeyError from inside synthesis. Enforcing it
        # at construction means no Catalog that exists can reach that state.
        if fallback is not None and fallback not in voices:
            raise ValueError(
                f"fallback voice {fallback!r} is not among the installed voices: "
                f"{', '.join(sorted(voices)) or '(none)'}"
            )
        self._voices = voices
        self._fallback = fallback
        self._include_alignments = include_alignments
        # Taken as a value, not read from disk. Resolution is pure once it holds
        # its table, so a test can supply one and `load_aliases` can fail at
        # startup where a malformed file is an operator's problem to see.
        table = {} if aliases is None else aliases
        # Held to the same standard as the fallback, and for the same reason:
        # `resolve` indexes `_voices[aliased]` on its alias branch, so a target
        # that is not installed is the same bare KeyError from inside a request.
        # Refused rather than filtered — dropping uninstalled targets is
        # `load_aliases`' job and it says how many it dropped, whereas a
        # constructor that quietly discarded a caller's entry would report
        # nothing at all.
        dangling = sorted(set(table.values()) - set(voices))
        if dangling:
            raise ValueError(
                f"alias targets are not among the installed voices: "
                f"{', '.join(dangling)}"
            )
        self._aliases = table
        self._loaded: dict[str, "PiperVoice"] = {}

    @property
    def installed(self) -> tuple[Voice, ...]:
        """Every voice that can be spoken now, in a stable order."""
        return tuple(self._voices[key] for key in sorted(self._voices))

    def aliases_for(self, key: str) -> tuple[str, ...]:
        """Foreign ids that reach `key`, for `GET /v1/voices` to report."""
        return tuple(sorted(f for f, local in self._aliases.items() if local == key))

    def get(self, key: str) -> Voice | None:
        """The installed voice with this exact key, if there is one."""
        return self._voices.get(key)

    def resolve(self, requested: str) -> Resolution:
        """Decides which installed voice answers for `requested`.

        Three steps, most specific first: the id names an installed voice, the
        id is aliased onto one, or the fallback speaks. Only the third can be
        switched off.
        """
        exact = self._voices.get(requested)
        if exact is not None:
            return Resolution(voice=exact, requested=requested, substituted=False)

        aliased = self._aliases.get(requested)
        if aliased is not None:
            # An alias is a deliberate mapping, not a guess, so the caller did
            # reach the voice this server promised for that id — reported as a
            # substitution anyway, because the voice that speaks is not the one
            # whose id was sent, and a caption or a voice picker needs to know.
            return Resolution(
                voice=self._voices[aliased], requested=requested, substituted=True
            )

        if self._fallback is None:
            raise VoiceNotInstalled(requested, tuple(sorted(self._voices)))

        return Resolution(
            voice=self._voices[self._fallback], requested=requested, substituted=True
        )

    def model(self, voice: Voice) -> "PiperVoice":
        """The loaded ONNX model for `voice`, loading it on first use."""
        cached = self._loaded.get(voice.key)
        if cached is not None:
            return cached

        from piper import PiperVoice

        _LOGGER.info("loading voice %s", voice.key)
        # `include_alignments` patches the graph in memory to expose per-phoneme
        # durations. It is decided once, for the whole process, rather than per
        # request: the patch happens at load time, so a request asking for
        # timestamps against an unpatched session could only fail or lie.
        model = PiperVoice.load(
            str(voice.model_path), include_alignments=self._include_alignments
        )
        self._loaded[voice.key] = model
        return model


def install(
    keys: tuple[str, ...],
    models_dir: Path,
    fallback: str | None,
    include_alignments: bool,
    allow_download: bool,
) -> Catalog:
    """Makes sure every named voice is on disk, and builds the catalog.

    Runs at startup, before the server accepts a request, so that a missing
    model is one clean failure to boot rather than an unbounded delay inside
    somebody's first call.
    """
    from piper.download_voices import download_voice

    models_dir.mkdir(parents=True, exist_ok=True)
    voices: dict[str, Voice] = {}

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
            # FileNotFoundError from `_describe` opening the sidecar, which
            # names the missing file but not the download that failed to make it.
            if not (model_path.exists() and config_path.exists()):
                raise FileNotFoundError(
                    f"downloading voice {key!r} into {models_dir} did not produce "
                    f"both {model_path.name} and {config_path.name}"
                )
        voices[key] = _describe(key, model_path)

    # Read before the catalog is built, so `aliases.toml` is parsed during
    # startup rather than on whichever request first needs an alias. The file is
    # documented as operator-editable, so a malformed edit is a realistic event;
    # reached lazily it surfaced as an uncaught TOMLDecodeError on a synthesis
    # call, which is the opposite of this module's "one clean failure to boot"
    # rule and is invisible to a healthcheck that never touches resolution.
    #
    # The fallback's membership in `voices` is Catalog's own precondition and is
    # enforced in its constructor, so this call is also where that is checked.
    return Catalog(
        voices=voices,
        fallback=fallback,
        include_alignments=include_alignments,
        aliases=load_aliases(voices),
    )


def load_aliases(voices: dict[str, Voice]) -> dict[str, str]:
    """Foreign voice ids mapped onto the installed Piper keys they reach.

    [LAW:effects-at-boundaries] The one place `aliases.toml` is read. Resolution
    itself takes the finished table, so it stays a pure function of its inputs
    and a test can hand it a fixture instead of depending on the shipped file.

    Entries naming a voice that is not installed are dropped rather than kept and
    failed later: the table's job is to answer "which voice is this", and an
    answer that cannot be spoken is not an answer.
    """
    if not _ALIASES_FILE.exists():
        return {}
    with _ALIASES_FILE.open("rb") as handle:
        table = tomllib.load(handle)
    published = table.get("elevenlabs", {})
    mapped = {
        foreign: local for foreign, local in published.items() if local in voices
    }
    dropped = len(published) - len(mapped)
    if dropped:
        _LOGGER.info(
            "%d alias(es) point at voices that are not installed; ignoring them",
            dropped,
        )
    return mapped


def _describe(key: str, model_path: Path) -> Voice:
    """Reads a voice's metadata from the config file beside its model.

    From the `.onnx.json` rather than from the remote catalog, because that file
    is what the loaded model actually runs on: the sample rate here is the rate
    the samples will really have, which is what the encoder needs.
    """
    import json

    with (model_path.parent / f"{key}.onnx.json").open(encoding="utf-8") as handle:
        config = json.load(handle)

    parts = key.split("-")
    # `or {}` rather than a default argument: `.get(key, {})` substitutes only
    # for an absent key, so an explicit null in a hand-edited or half-written
    # sidecar returned None and the chained lookup raised AttributeError —
    # instead of the key-derived fallback this expression already promises.
    language = (config.get("language") or {}).get("code") or (
        parts[0] if len(parts) == _KEY_PARTS else key
    )
    return Voice(
        key=key,
        name=config.get("dataset") or (parts[1] if len(parts) == _KEY_PARTS else key),
        language=language,
        quality=(config.get("audio") or {}).get("quality")
        or (parts[2] if len(parts) == _KEY_PARTS else "medium"),
        model_path=model_path,
        sample_rate=int(config["audio"]["sample_rate"]),
        num_speakers=int(config.get("num_speakers", 1)),
    )
