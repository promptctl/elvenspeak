"""Entry point: reads the environment, builds the app, serves it.

Kept at the repository root, and kept this thin, because `uv run main.py` is
what the README has always told people to type and what the container image
runs. Everything it does beyond wiring belongs in the package.
"""

from __future__ import annotations

import logging

from elvenspeak import create_app, piper
from elvenspeak.settings import Settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _app(settings: Settings):
    """The composition root: this is where the engine is chosen.

    [LAW:one-way-deps] The only place in the service that names a concrete
    engine. Everything above it — the endpoints, the formats, the alignment, the
    voice table — depends on [`elvenspeak.engine`] and cannot tell which one is
    behind it, so a second engine is a change to this line rather than to the
    API surface.

    Built before the server starts rather than in a lifespan hook, so a voice
    that cannot be fetched or opened is a refusal to boot with a non-zero exit,
    not a process that binds a port and then answers 500 to everything.
    """
    return create_app(
        settings,
        piper.load(
            keys=settings.voices,
            models_dir=settings.models_dir,
            allow_download=settings.allow_download,
            timings=settings.timestamps,
        ),
    )


def build():
    """The ASGI application, for `uvicorn main:build --factory` and for tests."""
    return _app(Settings.from_env_or_exit())


if __name__ == "__main__":
    import uvicorn

    settings = Settings.from_env_or_exit()
    uvicorn.run(_app(settings), host=settings.host, port=settings.port)
