"""Puts the voices an image will serve onto its disk, at build time.

Run as `python -m elvenspeak.bake` by the Dockerfile, once, before the image is
sealed. The container ships with its voices already present because a ~60 MB
download on every restart is a slow start that looks like a hang, and because a
service that must reach Hugging Face to answer is a service that stops answering
when Hugging Face does.

# Why this is a file

It replaces a `python -c` string in the Dockerfile that did the same work. That
string was executable and load-bearing — the image's voices came from it — and
invisible to every tool that reads this repository: not imported, not linted,
not type-checked, not covered, and run by nothing but a real image build, which
by this repository's rules happens in CI after a merge. It shipped two escaped
defects in two consecutive pull requests, both past a fully green suite: a
symbol it went on importing after that symbol was moved, and a guarantee that
weakened underneath it without changing its call. Static checking of a string
catches the first shape and can never catch the second, because a function that
still exists and still takes those arguments can always mean less than it did.
As an ordinary module none of that is special: a test calls [`bake`] directly.

[LAW:one-way-deps] Names Piper, as `main` does, and for the same reason —
choosing an engine is what an entry point is for. Nothing in the ElevenLabs
surface imports this module, so the seam `elvenspeak.engine` draws is untouched
and `test_encoding.py`'s import-graph check has nothing new to say about it.
"""

from __future__ import annotations

import logging

from . import engine, piper
from .settings import Settings

_LOGGER = logging.getLogger("elvenspeak.bake")

#: The build fetches; the runtime does not. `PIPER_ALLOW_DOWNLOAD` is off in the
#: image so a missing model at boot fails the deploy instead of quietly
#: re-downloading, which leaves this step as the one moment a fetch is the right
#: answer. [LAW:one-source-of-truth] Stated here rather than passed in by the
#: Dockerfile, because a fact spelled in the Dockerfile is a fact no test reads.
_ALLOW_DOWNLOAD = True


def bake(settings: Settings) -> tuple[engine.Voice, ...]:
    """Installs every voice `settings` names, and says what they turned out to be.

    [LAW:parse-dont-validate] Returns the described voices rather than nothing,
    because that return is the guarantee worth having: `install` cannot report a
    voice whose `.onnx.json` it failed to read, so a value here is proof the
    image can describe what it serves. A truncated sidecar therefore fails the
    build, which is the last moment that failure is cheap — the alternative is
    a green image that dies at every container start.
    """
    ready = piper.install(
        keys=settings.voices,
        models_dir=settings.models_dir,
        allow_download=_ALLOW_DOWNLOAD,
    )
    return tuple(ready[key].voice for key in settings.voices)


def main() -> None:
    """`python -m elvenspeak.bake`: the environment in, voices on disk, or exit."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env_or_exit()
    for voice in bake(settings):
        _LOGGER.info("baked %s into %s", voice.id, settings.models_dir)


if __name__ == "__main__":
    main()
