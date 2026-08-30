"""The one place this process reads its environment.

[LAW:parse-dont-validate] Everything downstream of [`Settings.from_env`] runs on
values already known to be well-formed, so no handler asks whether a voice list
was empty or a port was a number. A bad environment stops the process at
startup, naming every problem at once, rather than surfacing as a 500 on
somebody's first call.
"""

from __future__ import annotations

import os
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

        fallback = env.get("PIPER_FALLBACK_VOICE", voices[0] if voices else None)
        if fallback == "":
            fallback = None
        elif fallback is not None and voices and fallback not in voices:
            problems.append(
                f"PIPER_FALLBACK_VOICE={fallback!r} is not in PIPER_VOICES "
                f"({', '.join(voices)})"
            )

        port_text = env.get("PORT", "5001")
        try:
            port = int(port_text)
        except ValueError:
            problems.append(f"PORT={port_text!r} is not a number")
            port = 0

        if problems:
            raise ConfigError(problems)

        return Settings(
            voices=voices,
            fallback=fallback,
            models_dir=Path(
                env.get("PIPER_MODELS_DIR", str(Path(__file__).parent.parent / "models"))
            ),
            allow_download=_flag(env, "PIPER_ALLOW_DOWNLOAD", default=True),
            api_key=env.get("ELVENSPEAK_API_KEY") or None,
            timestamps=_flag(env, "ELVENSPEAK_TIMESTAMPS", default=True),
            host=env.get("HOST", "0.0.0.0"),
            port=port,
        )


class ConfigError(ValueError):
    """Everything wrong with the environment, reported in one pass.

    A list rather than the first problem, because an operator bringing the
    service up for the first time should get the whole list, not one item per
    restart.
    """

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


def _flag(env, name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
