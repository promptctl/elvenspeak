"""The one place this process reads its environment.

[LAW:parse-dont-validate] Everything downstream of [`Settings.from_env`] runs on
values already known to be well-formed, so no handler asks whether a voice list
was empty or a port was a number. A bad environment stops the process at
startup, naming every problem at once, rather than surfacing as a 500 on
somebody's first call.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

#: The voice installed when none is named. `en_US-lessac-medium` because it is
#: the one Piper voice that is MIT at the repository level, so a default install
#: carries no licence surprise — see README, "Voice licensing".
DEFAULT_VOICE = "en_US-lessac-medium"


@dataclass(frozen=True)
class Settings:
    """Everything the server needs, with no absent values left in it."""

    #: Voices to install at startup, in the order they were named. The first is
    #: the fallback unless `fallback` says otherwise.
    voices: tuple[str, ...]
    #: Which installed voice answers for an id this server does not know.
    #: `None` switches substitution off, and unknown ids become 404 — correct
    #: for a closed deployment, wrong for anything replacing ElevenLabs, which
    #: is why it is not the default.
    fallback: str | None
    models_dir: Path
    #: Whether a voice missing from `models_dir` may be fetched at startup.
    #: Off in an image that bakes its voices in, so a broken mount fails loudly
    #: instead of quietly re-downloading 60 MB on every boot.
    allow_download: bool
    #: The value callers must present in `xi-api-key`. `None` accepts every
    #: request, which is the right default for a service on a private network
    #: and the wrong one anywhere else.
    api_key: str | None
    #: Patches voice models at load time to expose phoneme durations. Costs
    #: memory and load time for every voice, so it is off unless the timestamp
    #: endpoints are actually wanted.
    timestamps: bool
    host: str
    port: int

    @staticmethod
    def from_env(environ: dict[str, str] | None = None) -> "Settings":
        env = os.environ if environ is None else environ
        problems: list[str] = []

        voices = tuple(
            name.strip()
            for name in env.get("PIPER_VOICES", DEFAULT_VOICE).split(",")
            if name.strip()
        )
        if not voices:
            problems.append("PIPER_VOICES is empty; name at least one voice")

        # Stripped like the entries it is compared against. Comparing a raw value
        # to trimmed ones made a trailing space in a .env file report that a
        # plainly-present voice was missing.
        fallback = env.get("PIPER_FALLBACK_VOICE", voices[0] if voices else None)
        if fallback is not None:
            fallback = fallback.strip()
        if fallback == "":
            fallback = None
        # No `voices and` here: an empty PIPER_VOICES is its own problem, and
        # suppressing this one until that is fixed is how an operator discovers
        # the second misconfiguration only after restarting for the first.
        elif fallback is not None and fallback not in voices:
            problems.append(
                f"PIPER_FALLBACK_VOICE={fallback!r} is not in PIPER_VOICES "
                f"({', '.join(voices)})"
            )

        # Stripped and checked like everything else here. `PIPER_MODELS_DIR=` is
        # a present key, so `get` returns "" rather than the default, `Path("")`
        # is the working directory, and `mkdir` on it succeeds — the server then
        # reads and writes 60 MB models wherever it happened to be launched from,
        # having reported nothing. An unset variable interpolated into a compose
        # file is an ordinary way to arrive there.
        models_text = env.get("PIPER_MODELS_DIR", "").strip()
        if "PIPER_MODELS_DIR" in env and not models_text:
            problems.append("PIPER_MODELS_DIR is empty; name a directory or unset it")
        models_dir = Path(models_text or str(Path(__file__).parent.parent / "models"))

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

        flags = {}
        for name, default in (
            ("PIPER_ALLOW_DOWNLOAD", True),
            ("ELVENSPEAK_TIMESTAMPS", True),
        ):
            try:
                flags[name] = _flag(env, name, default=default)
            except ValueError as error:
                problems.append(str(error))
                flags[name] = default

        if problems:
            raise ConfigError(problems)

        return Settings(
            voices=voices,
            fallback=fallback,
            models_dir=models_dir,
            allow_download=flags["PIPER_ALLOW_DOWNLOAD"],
            api_key=env.get("ELVENSPEAK_API_KEY") or None,
            timestamps=flags["ELVENSPEAK_TIMESTAMPS"],
            host=env.get("HOST", "0.0.0.0"),
            port=port,
        )


    @staticmethod
    def from_env_or_exit() -> "Settings":
        """The environment, or a process already stopped over what is wrong with it.

        [LAW:single-enforcer] Every entry point comes through here — `uv run
        main.py`, the factory behind `uvicorn main:build --factory`, and the
        image's `python -m elvenspeak.bake` step — so a misconfiguration is
        reported one way whichever one is running. This was `main.py`'s private
        helper, which is the shape that lets the next entry point answer a bad
        environment with a raw traceback: the divergence gets written by
        omission, in the module that never knew the helper existed.
        """
        try:
            return Settings.from_env()
        except ConfigError as error:
            # Every problem at once, on stderr, with a non-zero exit: an operator
            # bringing this up for the first time should not discover their
            # configuration one restart at a time.
            for problem in error.problems:
                print(f"config error: {problem}", file=sys.stderr)
            raise SystemExit(2) from None


class ConfigError(ValueError):
    """Everything wrong with the environment, reported in one pass.

    A list rather than the first problem, because an operator bringing the
    service up for the first time should get the whole list, not one item per
    restart.
    """

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def _flag(env, name: str, default: bool) -> bool:
    """Reads a boolean setting, refusing anything that is not clearly one.

    [LAW:no-silent-failure] The obvious implementation — true if the value is in
    a true-set, false otherwise — makes `PIPER_ALLOW_DOWNLOAD=tru` mean "off",
    silently, in the one module whose stated job is catching configuration
    mistakes at startup. A typo in a boolean is exactly as much a mistake as a
    typo in a port, and is reported the same way.
    """
    raw = env.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(
        f"{name}={raw!r} is not a boolean; "
        f"use one of {', '.join(sorted(_TRUE))} or {', '.join(sorted(_FALSE))}"
    )
