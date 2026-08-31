"""Entry point: reads the environment, builds the app, serves it.

Kept at the repository root, and kept this thin, because `uv run main.py` is
what the README has always told people to type and what the container image
runs. Everything it does beyond wiring belongs in the package.
"""

from __future__ import annotations

import logging

from elvenspeak import create_app
from elvenspeak.engines import ENGINES
from elvenspeak.settings import Settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _app(settings: Settings):
    """The composition root: this is where the chosen engine is built.

    [LAW:one-way-deps] Names no engine. Everything above it — the endpoints, the
    formats, the alignment, the voice table — depends on [`elvenspeak.engine`]
    and cannot tell which one is behind it; this line cannot tell either, because
    the choice was made when the environment was parsed and arrives here as a
    value. A second engine is an entry in [`elvenspeak.engines`], not a change
    here and not a change to the API surface.

    Built before the server starts rather than in a lifespan hook, so a voice
    that cannot be fetched or opened is a refusal to boot with a non-zero exit,
    not a process that binds a port and then answers 500 to everything.
    """
    return create_app(settings, settings.engine.open())


def build():
    """The ASGI application, for `uvicorn main:build --factory` and for tests."""
    return _app(Settings.from_env_or_exit(ENGINES))


if __name__ == "__main__":
    import uvicorn

    settings = Settings.from_env_or_exit(ENGINES)
    uvicorn.run(_app(settings), host=settings.host, port=settings.port)
