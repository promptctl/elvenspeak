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

# Why the served set is on the voice and not here

[`Directory`] answers what reaches the *deployment*; which ids the server about
to speak answers to is [`elvenspeak.engine.Voice.models`], and [`reach`] takes
both. They are the same set for a single-engine deployment and cannot be for a
routed one, which is the reason capabilities moved onto the voice in
`piper-routing-7e2.4` and the reason this followed in `piper-routing-7e2.17`.

Deriving the deployment's set from its own engine name was what that ticket
found: a router asked `declarations.model_ids("router")`, was told nothing, and
so advertised itself as the only engine it served while fronting two. A router
ships no declaration file and never will — the ids it answers for are its
backends' — so the union over what the voices carry is the only answer that can
be right, and it is computed rather than held ([LAW:one-source-of-truth]).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from . import declarations
from .provisioning import ConfigError


class Reach(Enum):
    """How far a `model_id` gets in this deployment."""

    #: The server about to speak answers for it — by its own name, or as one of
    #: the foreign ids it declares.
    SERVED = "served here"
    #: It names an engine that will not be speaking this request: one this build
    #: has and this deployment is not running, or — behind a router — one running
    #: here that does not own the resolved voice. Refused: the alternative is
    #: answering in the engine the caller did not ask for, which is
    #: indistinguishable from success from the outside.
    ELSEWHERE = "an engine that is not speaking this request"
    #: It names no engine here — a model id from the real ElevenLabs that
    #: nothing maps, or nothing at all when the request omitted the field.
    #: Reported as ignored rather than refused, because every stock ElevenLabs
    #: client sends a `model_id` and a 422 would reject most first requests.
    UNKNOWN = "no engine here"


def declared_by(name: str, known: Iterable[str]) -> frozenset[str]:
    """Every `model_id` a server running `name` answers to, out of a build with `known`.

    What one deployment stamps onto every voice it offers
    ([`elvenspeak.engine.Voice.models`]). A router stamps nothing with this: its
    voices arrive carrying their own backend's answer, which is the whole reason
    the field is on the voice.

    [LAW:effects-at-boundaries] Where the declaration file is read — once at
    startup, so a malformed table refuses to boot rather than surfacing on
    whichever request first named a model.

    `known` is the registry's keys, carried down from the one lookup that chose
    the engine ([`elvenspeak.settings`]) rather than imported. This module must
    stay importable without any engine's third-party library, exactly as the rest
    of the surface is.
    """
    declared = declarations.model_ids(name)
    # [LAW:parse-dont-validate] A declared id may name no engine at all — that is
    # what the table is for — but naming one is a contradiction no later code can
    # express its way out of, in either direction. Another engine's name would
    # make `reach` answer SERVED for a request that plainly named someone else;
    # this engine's own name is already served, so declaring it again duplicates
    # it in the one endpoint whose job is saying which ids are legal. Refused
    # here, so no served set that exists holds either.
    contested = sorted(set(declared) & set(known))
    if contested:
        raise ConfigError(
            [f"{name} declares model id(s) naming an engine: {', '.join(contested)}"]
        )
    return frozenset((name, *declared))


@dataclass(frozen=True)
class Directory:
    """Every `model_id` that reaches this deployment at all, against its build.

    Holds no engine name and reads no declaration. Both were how this used to be
    built, and both are what `piper-routing-7e2.17` found wrong: a router has a
    name that declares nothing and fronts engines that declare plenty, so a
    directory derived from the name advertised `["router"]` while routing to two
    engines it refused to name.
    """

    #: Every `model_id` some voice on offer answers to — the union over
    #: [`elvenspeak.engine.Voice.models`], derived where the voices are and never
    #: written beside them ([LAW:one-source-of-truth]).
    served: frozenset[str]
    #: The registry's keys: every engine name this *build* has, running here or
    #: not. What makes [`Reach.ELSEWHERE`] representable, and the only roster held
    #: anywhere near a router — it names engines, never their voices or their
    #: addresses, both of which remain the engines' own to advertise.
    known: frozenset[str]

    @staticmethod
    def over(offered: Iterable[frozenset[str]], known: Iterable[str]) -> "Directory":
        """The directory for a deployment whose voices answer to `offered`.

        Takes the served sets rather than the voices themselves so this module
        keeps naming no type it does not own.
        """
        return Directory(
            served=frozenset[str]().union(*offered), known=frozenset(known)
        )

    def listed(self) -> tuple[str, ...]:
        """Every served `model_id` for `GET /v1/models`, engine names first.

        The engines lead because they are the ids a caller can rely on: the rest
        are a compatibility mapping some backend happens to declare, and an
        operator reading the listing wants to know what is running before what it
        will answer to.
        """
        engines = self.served & self.known
        return (*sorted(engines), *sorted(self.served - engines))

    def reach(self, model_id: str | None, here: frozenset[str]) -> Reach:
        """How far `model_id` gets for a request the voice in `here` will speak.

        `here` is that voice's own [`elvenspeak.engine.Voice.models`]. Passing it
        per request is what makes a router honest in both directions: naming
        piper reaches piper's voices and is *refused* for kokoro's, rather than
        being honoured by whichever engine the voice happened to route to. Behind
        a single engine every voice carries the same set and this collapses to
        the deployment-wide question it used to be.

        `None` is [`Reach.UNKNOWN`] and not a state of its own: a request that
        named no engine and a request that named one nobody here has are the same
        situation — the voice decides — and giving them two spellings would put
        the same branch in every caller.
        """
        if model_id in here:
            return Reach.SERVED
        # Served by some *other* voice here, or by an engine this build has and
        # this deployment is not running: two ways of naming an engine that will
        # not be speaking, and one refusal, because the caller's mistake and the
        # answer they would otherwise get are identical in both.
        if model_id in self.served or model_id in self.known:
            return Reach.ELSEWHERE
        return Reach.UNKNOWN
