# syntax=docker/dockerfile:1

# ffmpeg is the dependency `pyproject.toml` structurally cannot declare: it is an
# executable, not a Python package, and every synthesis request shells out to it.
# A previous incident in this homelab was an Atlantis container missing `jq`, and
# the failure mode is identical — the image builds, the service starts, and every
# request fails at the moment it matters. It is installed here, and the build
# proves it is present rather than assuming.

FROM python:3.12-slim AS base

RUN apt-get update \
 && apt-get install --no-install-recommends -y ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 && ffmpeg -hide_banner -encoders | grep -q libmp3lame \
 && ffmpeg -hide_banner -encoders | grep -q libopus

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies before source, so editing a handler does not re-resolve the
# environment. `--frozen` makes the lockfile authoritative: a build that would
# need to change it fails instead of silently resolving something else.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY piper_server/ ./piper_server/
COPY main.py ./
RUN uv sync --frozen --no-dev

# Voices are baked in rather than fetched at boot. A ~60 MB download on every
# restart is a slow start that looks like a hang, and it makes the container
# depend on Hugging Face being reachable to serve traffic it could otherwise
# serve offline. PIPER_ALLOW_DOWNLOAD stays off for the same reason: a missing
# model should fail the deploy, not quietly re-download.
ARG PIPER_VOICES=en_US-lessac-medium
ENV PIPER_MODELS_DIR=/app/models
RUN uv run python -c "\
import os, pathlib; \
from piper.download_voices import download_voice; \
d = pathlib.Path(os.environ['PIPER_MODELS_DIR']); d.mkdir(parents=True, exist_ok=True); \
[download_voice(v, d) for v in '${PIPER_VOICES}'.split(',')]"

ENV PIPER_VOICES=${PIPER_VOICES} \
    PIPER_ALLOW_DOWNLOAD=0 \
    ELVENSPEAK_TIMESTAMPS=1 \
    PORT=5001

EXPOSE 5001

# Exercises a real synthesis path rather than just process liveness: the
# endpoint reports which voices actually loaded, so a container whose models
# failed to load reads as unhealthy instead of as up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,json,sys; \
b=json.load(urllib.request.urlopen('http://127.0.0.1:5001/health')); \
sys.exit(0 if b['voices'] else 1)"

CMD ["uv", "run", "--no-dev", "main.py"]
