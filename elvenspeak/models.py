"""Which engine a request's `model_id` names.

ElevenLabs splits synthesis into two independent axes — `model_id` chooses the
engine, `voice_id` chooses the voice — and this module answers the first exactly
as [`elvenspeak.voices`] answers the second: from what this process actually
has, through a table the engine declares, never from a roster written beside it.

# Why an unknown model is not an unknown voice

The two questions look alike and their answers to "what does an id I do not
recognise mean" are opposite, which is why they are two modules.

An unrecognised *voice* substitutes: clients hold ElevenLabs voice ids and a
server that 404s all of them replaces nothing. An unrecognised *model* resolves
to nothing at all — there is no fallback engine, because this deployment runs
the one engine it was built with, and answering in it would be inventing a
routing rule nobody wrote. What happens instead is what already happened when
`model_id` was omitted entirely: the voice decides, and `x-elvenspeak-ignored`
names the field so the caller is told the id did not steer anything.

# The three things a model id can be

[LAW:types-are-the-program] Three, not two, and the third is the whole ticket.
"Served here" and "means nothing here" leave no way to say *this names another
engine* — and without that state a deployment asked for kokoro can only shrug
and answer in piper, which is the silent wrong-engine answer this exists to
refuse. The set of engines this build knows is what makes the distinction
representable, so it is carried rather than guessed at from the shape of the id.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from . import declarations
from .provisioning import ConfigError


class Reach(Enum):
    """How far a `model_id` gets in this deployment."""

    #: This deployment's engine answers for it — by its own name, or as one of
    #: the foreign ids it declares.
    SERVED = "served here"
    #: It names an engine this build has and this deployment is not running.
    #: Refused: the alternative is answering in the engine the caller did not
    #: ask for, which is indistinguishable from success from the outside.
    ELSEWHERE = "an engine this deployment is not running"
    #: It names no engine here — a model id from the real ElevenLabs that
    #: nothing maps, or nothing at all when the request omitted the field.
    #: Reported as ignored rather than refused, because every stock ElevenLabs
    #: client sends a `model_id` and a 422 would reject most first requests.
    UNKNOWN = "no engine here"


@dataclass(frozen=True)
class Directory:
    """The model ids that reach this deployment, and the engines that do not."""

    #: The engine this deployment runs, under the name that selects it. Also a
    #: legal `model_id`: asking for `piper` by name on a piper deployment is the
    #: one request that cannot be ambiguous.
    engine: str
    #: Foreign model ids this engine declares, as declared. Order carries no
    #: meaning — nothing resolves by precedence here, which is the whole reason
    #: two engines claiming one id is refused rather than ranked.
    foreign: tuple[str, ...]
    #: Engine names this build has and this deployment is not serving. The
    #: difference between [`Reach.ELSEWHERE`] and [`Reach.UNKNOWN`], and the only
    #: reason a roster is held at all — it names engines, never their voices or
    #: their addresses, both of which remain the engines' own to advertise.
    absent: frozenset[str]

    @staticmethod
    def for_engine(name: str, known: Iterable[str]) -> "Directory":
        """The directory for a deployment running `name`, out of a build with `known`.

        [LAW:effects-at-boundaries] Where the declaration file is read, which is
        why it is separate from the constructor: once at startup, so a malformed
        table refuses to boot rather than surfacing on whichever request first
        named a model.

        `known` is the registry's keys, carried down from the one lookup that
        chose the engine ([`elvenspeak.settings.Settings.known_engines`]) rather
        than imported. This module must stay importable without any engine's
        third-party library, exactly as the rest of the surface is.
        """
        declared = declarations.model_ids(name)
        absent = frozenset(known) - {name}
        # [LAW:parse-dont-validate] An engine claiming another engine's name as
        # one of its foreign ids would make `reach` answer SERVED for a request
        # that plainly named someone else — the one contradiction this type
        # cannot express its way out of, since both readings are in the table.
        # Refused here, at construction, so no Directory that exists holds one.
        contested = sorted(set(declared) & absent)
        if contested:
            raise ConfigError(
                [
                    f"{name} declares model id(s) naming another engine: "
                    f"{', '.join(contested)}"
                ]
            )
        return Directory(engine=name, foreign=declared, absent=absent)

    @property
    def answers_for(self) -> frozenset[str]:
        """Every `model_id` this deployment serves."""
        return frozenset((self.engine, *self.foreign))

    def listed(self) -> tuple[str, ...]:
        """Every served `model_id` for `GET /v1/models`, this engine's name first.

        The engine leads because it is the id a caller can rely on: the foreign
        ones are a compatibility mapping this deployment happens to declare, and
        an operator reading the listing wants to know what it is before what it
        will answer to.
        """
        return (self.engine, *sorted(set(self.foreign)))

    def reach(self, model_id: str | None) -> Reach:
        """How far `model_id` gets, `None` being a request that named no model.

        `None` is [`Reach.UNKNOWN`] and not a state of its own: a request that
        named no engine and a request that named one nobody here has are the
        same situation — the voice decides — and giving them two spellings would
        put the same branch in every caller.
        """
        if model_id in self.answers_for:
            return Reach.SERVED
        if model_id in self.absent:
            return Reach.ELSEWHERE
        return Reach.UNKNOWN
