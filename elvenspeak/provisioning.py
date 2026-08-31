"""How a deployment obtains an engine.

[`elvenspeak.engine`] says what the *endpoints* need from a source of speech.
This module says what the *entry points* need to get one, which is a separate
question with a separate audience: an engine author reads `engine` and never has
to read this, and everything here was derived from a line in `main` or `bake`
rather than from any endpoint.

    main.py                 "give me a ready engine"     -> [`Prepared.open`]
    python -m elvenspeak.bake  "put its assets on disk"  -> [`Prepared.acquire`]
    a bad environment       "say so at startup, once"    -> [`Configure`]

# Why choosing an engine is a value and not a line of code

Both entry points used to name a module and call it. That is two maps of one
fact — which engine this deployment runs — with nothing making them agree, so
the day a second engine exists the image can bake one engine's voices and boot
another's. Both now read a single [`Prepared`] that [`elvenspeak.settings`]
parsed once, and the disagreement stops being expressible.

Across processes it cannot be closed here, and does not need to be: the build
and the boot read their environments separately, but an engine asked to open
assets that were never baked fails loudly, because fetching at boot is off in
the image. The harmless direction — assets baked for an engine nobody runs — is
wasted bytes rather than wrong audio.

# Why this module names no engine

[LAW:one-way-deps] The roster of real engines lives in [`elvenspeak.engines`],
which the entry points import and nothing in the ElevenLabs surface does. If the
lookup lived behind `settings` instead, `api` would reach every engine's library
through its own configuration import, and the reusable half would stop being
importable without them — a claim `tests/test_encoding.py` checks rather than
describes, for both `settings` and this module.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from .engine import Engine, Voice


class ConfigError(ValueError):
    """Everything wrong with the environment, reported in one pass.

    A list rather than the first problem, because an operator bringing the
    service up for the first time should get the whole list, not one item per
    restart.

    Lives here rather than in [`elvenspeak.settings`] so that an engine can raise
    it while parsing its own configuration without importing the module that
    parses the server's. [LAW:one-way-deps] Nothing an engine needs may point
    back up at the entry point that chose it.
    """

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


class Prepared(Protocol):
    """One deployment's engine: configured, checked, and not yet built.

    [LAW:parse-dont-validate] The type a [`Configure`] returns and could not have
    returned before it read the environment. Both methods take no arguments,
    which is the whole point — there is no way to hand either of them a value
    that has not already been through the one checkpoint, because there is no way
    to hand them anything.

    Constructing one does no I/O. It describes what to do; the entry points do
    it ([LAW:effects-at-boundaries]), which is what lets a whole environment be
    parsed and rejected before a single model is fetched or opened.
    """

    def acquire(self) -> tuple[Voice, ...]:
        """Makes this engine's assets present, and says what they turned out to be.

        Run once at image build time, never while serving. Fetching is what this
        method *is* rather than a flag on it: the build is the moment a download
        is the right answer, and [`open`] is the moment it is not, so the two
        lifecycle points are told apart by which method the caller reached for
        rather than by a boolean both of them read.

        Idempotent, because a rebuild over a warm cache must not re-fetch what is
        already there.

        Returns the voices, rather than nothing, because that return is the
        guarantee worth having: an engine that cannot describe what it installed
        has not installed it, and the build is the last moment that failure is
        cheap. An engine with nothing to install — a remote API — returns no
        voices, which is the honest answer and not a failure; an engine with
        something to install refuses an empty voice list in [`Configure`], so the
        two cases never arrive at the same value.
        """
        ...

    def open(self) -> Engine:
        """Builds the engine, ready to speak in every voice it will offer.

        Everything slow or failable happens here rather than inside the first
        request: [`Engine.voices`] promises its voices can be spoken *now*, and
        the only way to keep that promise is to have finished before the port is
        bound. A deployment problem is therefore a refusal to boot rather than an
        unbounded silent delay charged to whoever called first.
        """
        ...


#: Turns an environment into a [`Prepared`], or raises [`ConfigError`] naming
#: every problem it found — all of them, not the first, so that an engine's
#: complaints join the server's in the single list an operator reads at startup.
Configure = Callable[[Mapping[str, str]], Prepared]

#: The engines a deployment may choose between, by the name that selects one.
#:
#: [LAW:one-source-of-truth] A mapping rather than a sequence of records that
#: each carry a name: a `name` field inside a value stored under that same name
#: is two copies free to disagree. The key is the name.
#:
#: Order is meaning. The first entry is the default, so "the default is one of
#: the engines" is true by construction rather than by a second setting that
#: could name a missing one — the same convention [`Settings.voices`] already
#: uses for the fallback voice.
Registry = Mapping[str, Configure]
