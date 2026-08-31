"""Which voice a request means.

One question — "which voice is this?" — asked by three callers that must agree:
the synthesis endpoints, `GET /v1/voices`, and the alias table. They agree
because they all read this module and this module reads one catalog.

It is server policy, deliberately kept out of [`elvenspeak.engine`]. An engine
answers about the voices it has and speaks in one it was handed; deciding what an
unrecognised id ought to mean is a compatibility decision this service makes, and
an engine that had to make it would be implementing the same table again.

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
occurred, and every synthesis response carries an `x-elvenspeak-voice` header
naming what actually spoke. The behaviour a client depends on is preserved; the
fact it happened is not hidden.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path

from . import engine

_LOGGER = logging.getLogger("elvenspeak.voices")

_ALIASES_FILE = Path(__file__).parent / "aliases.toml"


@dataclass(frozen=True)
class Resolution:
    """The voice that will speak, and whether it is the one that was asked for.

    Two fields rather than a bare [`engine.Voice`], because "you got what you
    named" and "you got a substitute" are different facts about the same
    successful response, and a caller that cannot tell them apart cannot report
    the second.
    """

    voice: engine.Voice
    requested: str
    substituted: bool


class VoiceNotInstalled(LookupError):
    """A voice id that is neither offered nor aliased, with no fallback set.

    Only reachable when substitution is switched off. With a fallback configured
    — the default — resolution always succeeds.
    """

    def __init__(self, requested: str, available: tuple[str, ...]) -> None:
        super().__init__(
            f"voice {requested!r} is not installed; "
            f"available: {', '.join(available) or '(none)'}"
        )
        self.requested = requested


class Catalog:
    """The voices that can be spoken, and the one table that resolves an id.

    Holds values, not an engine: everything here is a lookup over
    [`engine.Voice`] and the alias table, so it is pure once constructed and a
    test can build one without a model, a network, or an engine at all.
    """

    def __init__(
        self,
        voices: dict[str, engine.Voice],
        fallback: str | None,
        aliases: dict[str, str] | None = None,
    ) -> None:
        # [LAW:parse-dont-validate] Checked here, where the catalog is made,
        # rather than by whichever caller remembers to. `resolve()` indexes
        # `_voices[_fallback]` on its last branch, so a fallback naming no
        # available voice turns every unrecognised id — the case the fallback
        # exists for — into a bare KeyError from inside synthesis. Enforcing it
        # at construction means no Catalog that exists can reach that state.
        if fallback is not None and fallback not in voices:
            raise ValueError(
                f"fallback voice {fallback!r} is not among the installed voices: "
                f"{', '.join(sorted(voices)) or '(none)'}"
            )
        self._voices = voices
        self._fallback = fallback
        # Taken as a value, not read from disk. Resolution is pure once it holds
        # its table, so a test can supply one and `load_aliases` can fail at
        # startup where a malformed file is an operator's problem to see.
        table = {} if aliases is None else aliases
        # Held to the same standard as the fallback, and for the same reason:
        # `resolve` indexes `_voices[aliased]` on its alias branch, so a target
        # that is not available is the same bare KeyError from inside a request.
        # Refused rather than filtered — dropping unavailable targets is
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

    @staticmethod
    def for_engine(source: engine.Engine, fallback: str | None) -> "Catalog":
        """The catalog over everything `source` can speak now.

        [LAW:effects-at-boundaries] Where `aliases.toml` is read, which is why
        this is separate from the constructor: it happens once at startup, so a
        malformed edit to an operator-editable file is a refusal to boot rather
        than an uncaught TOMLDecodeError on whichever synthesis call first needed
        an alias — invisible to a healthcheck that never touches resolution.
        """
        voices = {voice.id: voice for voice in source.voices()}
        return Catalog(
            voices=voices, fallback=fallback, aliases=load_aliases(voices)
        )

    @property
    def installed(self) -> tuple[engine.Voice, ...]:
        """Every voice that can be spoken now, in a stable order."""
        return tuple(self._voices[key] for key in sorted(self._voices))

    def aliases_for(self, key: str) -> tuple[str, ...]:
        """Foreign ids that reach `key`, for `GET /v1/voices` to report."""
        return tuple(sorted(f for f, local in self._aliases.items() if local == key))

    def get(self, key: str) -> engine.Voice | None:
        """The available voice with this exact id, if there is one."""
        return self._voices.get(key)

    def resolve(self, requested: str) -> Resolution:
        """Decides which available voice answers for `requested`.

        Three steps, most specific first: the id names a voice this server has,
        the id is aliased onto one, or the fallback speaks. Only the third can be
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


def load_aliases(voices: dict[str, engine.Voice]) -> dict[str, str]:
    """Foreign voice ids mapped onto the local ids they reach.

    [LAW:effects-at-boundaries] The one place `aliases.toml` is read. Resolution
    itself takes the finished table, so it stays a pure function of its inputs
    and a test can hand it a fixture instead of depending on the shipped file.

    Entries naming a voice that is not available are dropped rather than kept and
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
