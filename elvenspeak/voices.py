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
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from . import declarations, engine
from .provisioning import ConfigError

_LOGGER = logging.getLogger("elvenspeak.voices")


class Substitution(Enum):
    """What answers for an unknown voice id when the operator named no voice.

    [LAW:types-are-the-program] Two answers rather than one absent value. The
    setting has always had three states — name a voice, take the obvious one, or
    switch substitution off — and while the voice list lived beside it in
    [`Settings`], "the obvious one" could be resolved during parsing and the
    third state collapsed into `None`. The list belongs to the engine now, so the
    choice outlives the parse and has to survive in the type; spelled as one
    absent value it would be two meanings sharing a representation, and the
    operator who cleared the variable to disable substitution would silently get
    a voice instead.
    """

    #: Whichever voice the engine offers first. The compatible default: a server
    #: that 404s the ElevenLabs ids its clients hold is a drop-in for nothing.
    FIRST_OFFERED = "the first voice the engine offers"
    #: Nothing — an unrecognised id is a 404. Set by giving the variable an empty
    #: value, and correct only for a deployment whose callers know its ids.
    OFF = "no substitution"


#: A voice id, or how to pick one. What [`Settings.fallback`] holds.
Fallback = str | Substitution


def _chosen(fallback: Fallback, voices: Mapping[str, engine.Voice]) -> str | None:
    """The id that answers for unknown ones, given what the engine actually offers.

    [LAW:dataflow-not-control-flow] The one branch here is on the domain's own
    enum, which is what that enum is for; the result is a single value, so
    nothing downstream re-asks how it was chosen.

    An engine offering no voices leaves nothing to substitute with, and the
    catalog then refuses every id by name — the same answer it gives for
    [`Substitution.OFF`], because it is the same situation and not a failure this
    function could report more usefully.
    """
    if fallback is Substitution.OFF:
        return None
    if fallback is Substitution.FIRST_OFFERED:
        return next(iter(voices), None)
    return fallback


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
        #
        # A [`ConfigError`] rather than a bare `ValueError`, because what is
        # wrong is the operator's environment and it should be reported the way
        # every other bad setting is. Raised from here it reaches
        # `settings.reported_or_exit`, which the entry points wrap around the
        # whole startup — a plain ValueError came out as an unhandled traceback,
        # since this is the one configuration check that cannot run until an
        # engine is open. It subclasses ValueError, so a caller catching that
        # still does.
        # Accumulated rather than raised one at a time, because [`ConfigError`]
        # promises the whole list and an operator with a bad fallback *and* a
        # dangling alias would otherwise fix one, restart, and meet the other.
        # This module was the last one still raising on the first problem.
        problems: list[str] = []
        if fallback is not None and fallback not in voices:
            problems.append(
                f"fallback voice {fallback!r} is not among the installed "
                f"voices: {', '.join(sorted(voices)) or '(none)'}"
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
            problems.append(
                f"alias targets are not among the installed voices: "
                f"{', '.join(dangling)}"
            )

        if problems:
            raise ConfigError(problems)
        self._aliases = table

    @property
    def fallback(self) -> str | None:
        """The id that actually answers for unknown ones, or `None` for none.

        Read rather than [`Settings.fallback`] by anything reporting to an
        operator: the setting may only say "whichever comes first", and this is
        the one place that knows which one that turned out to be.

        Read-only, like [`installed`]. [`resolve`] indexes `_voices` with this on
        its last branch, so a settable attribute would let later code put back
        the state the constructor check above exists to make unreachable — and
        that check runs once, at construction, which is the whole basis on which
        that comment claims no `Catalog` can reach it.
        """
        return self._fallback

    @staticmethod
    def for_engine(name: str, source: engine.Engine, fallback: Fallback) -> "Catalog":
        """The catalog over everything the engine called `name` can speak now.

        `name` is that engine's key in [`elvenspeak.engines.ENGINES`], and it is
        here for one reason: it picks the declaration file. It arrives as an
        argument rather than being asked of `source`, because which engine this
        is was decided when the environment was parsed and is a fact
        [`elvenspeak.settings.Settings`] already holds — asking the engine would
        make every engine answer a question about the server's registry.

        [LAW:effects-at-boundaries] Where the declarations are read, which is why
        this is separate from the constructor: it happens once at startup, so a
        malformed table is a refusal to boot rather than a failure raised on
        whichever synthesis call first needed an alias — invisible to a
        healthcheck that never touches resolution. Which exception that would be
        is not the point and is deliberately not named: [`declarations._read`]
        answers a declaration it cannot open or cannot parse with a
        `ConfigError`, so a type pinned here would be a copy of that decision,
        living in the module least likely to be edited when it changes.

        Also the one place [`Substitution.FIRST_OFFERED`] can be answered, since
        this is where a real voice list first exists. The constructor still takes
        a plain id and still refuses one it does not have, so the membership rule
        keeps its single enforcer ([LAW:single-enforcer]) and gains no second
        spelling for the resolved case.
        """
        voices = {voice.id: voice for voice in source.voices()}
        return Catalog(
            voices=voices,
            fallback=_chosen(fallback, voices),
            aliases=load_aliases(name, voices),
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

    def speaking(self, language: str | None) -> dict[str, engine.Voice]:
        """The voices eligible to answer, narrowed to a language when one is asked.

        [LAW:dataflow-not-control-flow] The narrowing is a value, so `resolve`
        runs its same three steps over a smaller table rather than growing a
        language branch through each of them. `None` — no language asked for —
        admits every voice, which is why this is a filter and not a special case.

        Falling back to the whole table when nothing speaks the language is the
        rule `model_id` already set: an id this deployment does not map steers
        nothing and is reported in `x-elvenspeak-ignored` rather than refused. A
        language no baked voice speaks is the same kind of unanswerable ask, and
        the alternative is worse than it looks — an empty table would send every
        such request to a 404 for a voice that is installed and was never the
        problem.
        """
        wanted = {
            key: voice
            for key, voice in self._voices.items()
            if language in (None, voice.language)
        }
        return wanted or self._voices

    def resolve(self, requested: str, language: str | None = None) -> Resolution:
        """Decides which available voice answers for `requested`.

        Three steps, most specific first: the id names a voice this server has,
        the id is aliased onto one, or the fallback speaks. Only the third can be
        switched off.

        `language` narrows which voices those steps may choose from. It outranks
        the id because our voices are monolingual, so the two cannot both be
        honoured and one of them has to give: a voice is a preference a substitute
        can satisfy — that is what this whole method is for — while reading
        Spanish text with English phonemes is not a lesser rendering of the
        request, it is a different and wrong one, and it is inaudible as a
        failure. The substitution is reported like every other, so a caller is
        never left guessing which voice spoke.

        Narrowing is gated on there being a fallback, and gated once for all three
        steps, because without one there is nothing to catch an id that narrowing
        removed: the request cannot fall through to a substitute, so it falls
        through to [`VoiceNotInstalled`] — whose message then lists as *available*
        the very voice that would have answered. Ungated, an exact id did exactly
        that. Applying it to the alias step alone would do it again, for aliases.
        So such a deployment gets the id it named and hears `language_code`
        reported in `x-elvenspeak-ignored`, which is the true answer where a 404
        was not.
        """
        speaking = self.speaking(language if self._fallback is not None else None)

        exact = speaking.get(requested)
        if exact is not None:
            return Resolution(voice=exact, requested=requested, substituted=False)

        aliased = self._aliases.get(requested)
        if aliased in speaking:
            # An alias is a deliberate mapping, not a guess, so the caller did
            # reach the voice this server promised for that id — reported as a
            # substitution anyway, because the voice that speaks is not the one
            # whose id was sent, and a caption or a voice picker needs to know.
            return Resolution(
                voice=speaking[aliased], requested=requested, substituted=True
            )

        if self._fallback is None:
            raise VoiceNotInstalled(requested, tuple(sorted(self._voices)))

        # The configured fallback where it speaks the language, and otherwise the
        # first voice that does. Offer order is already what a deployment naming
        # no fallback answers unknown ids with, so leaning on it here says the
        # same thing in the same voice rather than inventing a second rule.
        chosen = self._fallback if self._fallback in speaking else next(iter(speaking))
        return Resolution(
            voice=speaking[chosen], requested=requested, substituted=True
        )


def load_aliases(name: str, voices: dict[str, engine.Voice]) -> dict[str, str]:
    """Foreign voice ids mapped onto the local ids they reach, for one engine.

    Where the engine's declaration ([`elvenspeak.declarations`]) meets the voices
    it actually has. Resolution itself takes the finished table, so it stays a
    pure function of its inputs and a test can hand it a fixture instead of
    depending on the shipped file.

    [LAW:one-source-of-truth] The table is scoped to the engine because its
    values are that engine's voice ids and mean nothing anywhere else. One
    shared table was measured against the images that actually shipped and
    resolved to nothing in either of them: it named Piper voices, so the Kokoro
    image dropped all nine as unspeakable, and it named Piper voices the Piper
    image did not bake, so that image dropped all nine too. A table that can
    name a voice no image carries is a table free to be wrong everywhere at
    once; named after the engine, each one can only be wrong about voices its
    own engine was supposed to have.

    An engine with no declarations has no file and gets an empty table, which is
    the honest answer rather than a failure — it answers for the ids it owns and
    for nothing else. Nothing is registered centrally to add a table: an engine's
    declarations are a file named after it.

    Entries naming a voice that is not available are dropped rather than kept and
    failed later: the table's job is to answer "which voice is this", and an
    answer that cannot be spoken is not an answer.
    """
    published = declarations.voice_aliases(name)
    mapped = {
        foreign: local for foreign, local in published.items() if local in voices
    }
    dropped = len(published) - len(mapped)
    if dropped:
        _LOGGER.info(
            "%d of %s's alias(es) point at voices that are not installed; "
            "ignoring them",
            dropped,
            name,
        )
    return mapped
