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

# Why it names no engine

[LAW:one-source-of-truth] It used to name Piper, as `main` did, and that was two
answers to "which engine does this deployment run" with nothing keeping them the
same — an image whose bake step and whose boot chose differently was one edit
away, and nothing in the repository could have noticed. Both entry points read
the same [`Settings.engine`] now, so for any one environment the question has one
answer.

The build fetches and the runtime does not, which used to be a constant here
overriding the deployment's own `PIPER_ALLOW_DOWNLOAD`. It is
[`Prepared.acquire`] rather than [`Prepared.open`] now: the lifecycle moment is
carried by which method this module calls, so there is no setting to override
and no second statement of the rule to keep in step.
"""

from __future__ import annotations

import logging

from . import engine
from .engines import ENGINES
from .settings import Settings, reported_or_exit

_LOGGER = logging.getLogger("elvenspeak.bake")


def bake(settings: Settings) -> tuple[engine.Voice, ...]:
    """Installs every voice the chosen engine needs, and says what they are.

    [LAW:parse-dont-validate] Returns the described voices rather than nothing,
    because that return is the guarantee worth having: an engine cannot report a
    voice it failed to describe, so a value here is proof the image can describe
    what it serves. A truncated sidecar therefore fails the build, which is the
    last moment that failure is cheap — the alternative is a green image that
    dies at every container start.
    """
    return settings.engine.acquire()


def main() -> None:
    """`python -m elvenspeak.bake`: the environment in, voices on disk, or exit."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    with reported_or_exit():
        settings = Settings.from_env(ENGINES)
    for voice in bake(settings):
        _LOGGER.info("baked %s", voice.id)


if __name__ == "__main__":
    main()
