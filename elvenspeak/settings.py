"""The one place this process reads its environment.

[LAW:parse-dont-validate] Everything downstream of [`Settings.from_env`] runs on
values already known to be well-formed, so no handler asks whether a port was a
number. A bad environment stops the process at startup, naming every problem at
once, rather than surfacing as a 500 on somebody's first call.

# What is here and what is the engine's

Only what is true whichever engine is running: which engine that is, which voice
answers for an id this server does not know, whether a key is required, and
where to listen. Where model files live and whether they may be fetched are
Piper's business and are parsed by Piper — held here, they would be fields every
other engine is handed and ignores.

The engine's problems still arrive in this module's list, at this module's one
moment, because [`Settings.from_env`] splices them in. Separating the settings
did not separate the report.

[LAW:one-way-deps] The registry of real engines is taken as an argument rather
than imported. Reaching for it here would make `api`, which imports this module
for its API key, transitively import every engine's third-party library — and
the reusable half of this package would stop being importable without them.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

from .provisioning import ConfigError, Prepared, Registry
from .voices import Fallback, Substitution


@dataclass(frozen=True)
class Settings:
    """Everything the server needs, with no absent values left in it."""

    #: The chosen engine, configured and checked but not yet built. `main.py`
    #: opens it; the image's bake step acquires its assets. Both read this one
    #: value, which is what stops the build and the boot naming different
    #: engines.
    engine: Prepared
    #: Which voice answers for an id this server does not know. A name, or one of
    #: [`Substitution`]'s two answers for callers who named neither a voice nor
    #: nothing. Switching substitution off makes unknown ids 404 — correct for a
    #: closed deployment, wrong for anything replacing ElevenLabs, which is why
    #: it is not the default.
    fallback: Fallback
    #: The value callers must present in `xi-api-key`. `None` accepts every
    #: request, which is the right default for a service on a private network
    #: and the wrong one anywhere else.
    api_key: str | None
    host: str
    port: int

    @staticmethod
    def from_env(
        engines: Registry, environ: Mapping[str, str] | None = None
    ) -> "Settings":
        env = os.environ if environ is None else environ
        problems: list[str] = []

        # Stripped like the values it is compared against downstream. A trailing
        # space in a .env file once made a plainly-present voice report missing.
        fallback_text = env.get("ELVENSPEAK_FALLBACK_VOICE")
        fallback: Fallback = (
            Substitution.FIRST_OFFERED
            if fallback_text is None
            else (fallback_text.strip() or Substitution.OFF)
        )

        port_text = env.get("PORT", "5001")
        try:
            port = int(port_text)
        except ValueError:
            problems.append(f"PORT={port_text!r} is not a number")
            port = 0
        else:
            # Parsing is not validating: -1 and 99999 are integers and neither is
            # a port. Caught here so it joins the list this module exists to
            # produce, instead of failing later inside uvicorn with a worse
            # message.
            if not 1 <= port <= 65535:
                problems.append(f"PORT={port} is outside 1-65535")

        try:
            prepared = _prepare(engines, env)
        except ConfigError as error:
            # Spliced rather than replacing: a bad port and a bad voice name are
            # both true at once, and an operator should see both on the first
            # run rather than one per restart.
            raise ConfigError(problems + error.problems) from None

        if problems:
            raise ConfigError(problems)

        return Settings(
            engine=prepared,
            fallback=fallback,
            api_key=env.get("ELVENSPEAK_API_KEY") or None,
            host=env.get("HOST", "0.0.0.0"),
            port=port,
        )

@contextmanager
def reported_or_exit() -> Iterator[None]:
    """Runs a startup, turning any configuration problem into a clean exit 2.

    [LAW:single-enforcer] Every entry point comes through here — `uv run
    main.py`, the factory behind `uvicorn main:build --factory`, and the image's
    `python -m elvenspeak.bake` step — so a misconfiguration is reported one way
    whichever one is running. This was `main.py`'s private helper, which is the
    shape that lets the next entry point answer a bad environment with a raw
    traceback: the divergence gets written by omission, in the module that never
    knew the helper existed.

    A block rather than a wrapper around the parse, because not every
    configuration problem is discoverable while parsing. Whether the fallback
    voice names one the engine offers cannot be answered until the engine is
    open, so [`Catalog`] is where that check lives — and a reporter that spanned
    only [`Settings.from_env`] left exactly that problem coming out as an
    unhandled traceback, which is the one failure mode this whole module exists
    to prevent. The span is the startup, not the parse.
    """
    try:
        yield
    except ConfigError as error:
        # Every problem at once, on stderr, with a non-zero exit: an operator
        # bringing this up for the first time should not discover their
        # configuration one restart at a time.
        for problem in error.problems:
            print(f"config error: {problem}", file=sys.stderr)
        raise SystemExit(2) from None


def _prepare(engines: Registry, env: Mapping[str, str]) -> Prepared:
    """The named engine, configured from `env`, or a [`ConfigError`] saying why not.

    [LAW:parse-dont-validate] Its own unit, returning a type that could not exist
    before the name was checked, and failing loudly rather than returning an
    unconfigured stand-in. Nothing downstream re-asks whether the engine is real,
    because nothing downstream holds a name.

    An unnamed engine is the registry's first entry rather than a literal here.
    A default spelled in this module is a second answer to "which engines exist",
    free to name one that does not ([LAW:one-source-of-truth]).

    Named-but-blank is not unnamed. `ELVENSPEAK_ENGINE=` is a present key, which
    an operator reaches by interpolating an unset variable into a compose file —
    and taking it for "no preference" would boot the default engine, silently,
    for someone whose whole intent was to run a different one.
    """
    if not engines:
        # Before `next(iter(engines))`, which would otherwise be a bare
        # StopIteration — the one way out of this module that is not a
        # ConfigError, and so the one `reported_or_exit` cannot turn into a
        # clean exit. Nothing in this repository registers an empty one, but
        # `Registry` is a plain mapping a caller supplies and the type cannot
        # say it is non-empty.
        raise ConfigError(["no engines registered"])

    named = env.get("ELVENSPEAK_ENGINE", "").strip()
    if "ELVENSPEAK_ENGINE" in env and not named:
        raise ConfigError(
            ["ELVENSPEAK_ENGINE is empty; name an engine or unset it"]
        )
    chosen = named or next(iter(engines))
    configure = engines.get(chosen)
    if configure is None:
        raise ConfigError(
            [f"ELVENSPEAK_ENGINE={chosen!r} is not one of: {', '.join(engines)}"]
        )
    return configure(env)
