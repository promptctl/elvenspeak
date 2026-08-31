"""Entry point: reads the environment, builds the app, serves it.

Kept at the repository root, and kept this thin, because `uv run … main.py` is
what the README has always told people to type and what the container image
runs. Everything it does beyond wiring belongs in the package.
"""

from __future__ import annotations

import logging

from elvenspeak import create_app
from elvenspeak.engines import ENGINES
from elvenspeak.settings import Settings, reported_or_exit

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
    with reported_or_exit():
        return _app(Settings.from_env(ENGINES))


if __name__ == "__main__":
    import uvicorn

    # The block spans the build, not just the parse: a fallback voice that names
    # nothing the engine offers is only discoverable once the engine is open, and
    # it is as much a misconfiguration as a bad port.
    with reported_or_exit():
        settings = Settings.from_env(ENGINES)
        app = _app(settings)

    uvicorn.run(app, host=settings.host, port=settings.port)
