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

# [LAW:no-ambient-temporal-coupling] Pinned, not `:latest`. An unpinned tag makes
# two builds of this same commit different artifacts, and a breaking uv release
# then surfaces as a build failure at a time nobody chose.
COPY --from=ghcr.io/astral-sh/uv:0.9.10 /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies before source, so editing a handler does not re-resolve the
# environment. `--frozen` makes the lockfile authoritative: a build that would
# need to change it fails instead of silently resolving something else.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY elvenspeak/ ./elvenspeak/
COPY main.py ./
RUN uv sync --frozen --no-dev

# Voices are baked in rather than fetched at boot. A ~60 MB download on every
# restart is a slow start that looks like a hang, and it makes the container
# depend on Hugging Face being reachable to serve traffic it could otherwise
# serve offline. PIPER_ALLOW_DOWNLOAD stays off for the same reason: a missing
# model should fail the deploy, not quietly re-download.
#
# [LAW:one-source-of-truth] ELVENSPEAK_ENGINE is set once, above the bake and
# inherited by the runtime, so the image boots the engine whose assets it baked.
# Naming it only on `docker run` would leave the two halves free to differ — and
# an engine asked to open assets nobody installed fails at boot, which is loud
# but is a worse way to learn it than not being able to say it.
ARG ELVENSPEAK_ENGINE=piper
ARG PIPER_VOICES=en_US-lessac-medium
ENV ELVENSPEAK_ENGINE="${ELVENSPEAK_ENGINE}" \
    PIPER_MODELS_DIR=/app/models \
    PIPER_VOICES="${PIPER_VOICES}" \
    PIPER_ALLOW_DOWNLOAD=0 \
    ELVENSPEAK_TIMESTAMPS=1 \
    PORT=5001

# A module in the package, not Python written into this file. The build arg
# reaches it through the environment above, which is also why it cannot be
# injected into: interpolating `'${PIPER_VOICES}'.split(',')` once put a
# caller-supplied string inside a Python literal inside a shell command.
#
# What this step guarantees, and why the failure is worth having here, is
# documented where the code is — `elvenspeak/bake.py`. A `python -c` string was
# code no linter, type checker or test could see, and it shipped two escaped
# defects in two consecutive pull requests before it was given a file.
RUN uv run python -m elvenspeak.bake

# [LAW:effects-at-boundaries] Nothing after this point needs root. ffmpeg and the
# ONNX runtime both process caller-influenced input, so a compromise anywhere in
# that path lands in an unprivileged account rather than owning the container.
RUN useradd --system --create-home --home-dir /home/elvenspeak elvenspeak \
 && chown -R elvenspeak:elvenspeak /app
USER elvenspeak

# [LAW:one-source-of-truth] Expanded from the ENV above rather than repeated, so
# the exposed port cannot drift from the one the server binds. Docker resolves
# this at build time, so `run -P -e PORT=...` still publishes the value baked in
# here — no Dockerfile construct tracks a runtime override, and the drift this
# removes is the one that is actually representable.
EXPOSE ${PORT}

# Voices are loaded during startup, so a successful /health means their ONNX
# sessions were built — not merely that the process is alive. It does not
# synthesize: inference every 30s would compete with real requests for the CPU
# this service is bound by.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request,json,os,sys; \
b=json.load(urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ['PORT'])); \
sys.exit(0 if b['voices'] else 1)"

CMD ["uv", "run", "--no-dev", "main.py"]
