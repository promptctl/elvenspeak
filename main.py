"""Entry point: reads the environment, builds the app, serves it.

Kept at the repository root, and kept this thin, because `uv run main.py` is
what the README has always told people to type and what the container image
runs. Everything it does beyond wiring belongs in the package.
"""

from __future__ import annotations

import logging
import sys

from elvenspeak import create_app
from elvenspeak.settings import ConfigError, Settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _settings() -> Settings:
    """Reads the environment, or exits naming every problem with it.

    [LAW:single-enforcer] Both entry points come through here, so a
    misconfiguration is reported the same way whichever one is used. Previously
    only the script path caught `ConfigError`, and `uvicorn main:build --factory`
    — a documented, supported way to start this service — answered a bad
    environment with a raw traceback carrying every problem joined onto one line.
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


def build():
    """The ASGI application, for `uvicorn main:build --factory` and for tests."""
    return create_app(_settings())


if __name__ == "__main__":
    import uvicorn

    settings = _settings()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
