# syntax=docker/dockerfile:1

# The two dependencies `pyproject.toml` structurally cannot declare: they are
# native, not Python packages. Every synthesis request shells out to ffmpeg, and
# Kokoro phonemizes through espeak-ng. A previous incident in this homelab was an
# Atlantis container missing `jq`, and the failure mode is identical — the image
# builds, the service starts, and every request fails at the moment it matters.
# Both are installed here, and the build proves each is present rather than
# assuming.
#
# espeak-ng comes from apt rather than from the `espeakng-loader` wheel that
# `kokoro-onnx` would otherwise reach for. That wheel ships a library which
# ignores the data path it is initialized with and aborts the process on a path
# from the machine it was built on — verified against the library directly on
# macOS. It aborts rather than failing to load, so phonemizer's own fallback to a
# system-wide espeak never fires, and the only reliable fix is to name the
# working library outright. Hence the symlink below: the apt path is
# architecture-dependent (`x86_64-linux-gnu`, `aarch64-linux-gnu`) and `ENV`
# cannot glob, so the build resolves it once to a fixed name.

FROM python:3.12-slim AS base

RUN apt-get update \
 && apt-get install --no-install-recommends -y ffmpeg espeak-ng \
 && rm -rf /var/lib/apt/lists/* \
 && ffmpeg -hide_banner -encoders | grep -q libmp3lame \
 && ffmpeg -hide_banner -encoders | grep -q libopus \
 && lib="$(ls /usr/lib/*/libespeak-ng.so.1 | head -1)" \
 && test -n "$lib" \
 && ln -s "$lib" /usr/local/lib/libespeak-ng.so.1

ENV PHONEMIZER_ESPEAK_LIBRARY=/usr/local/lib/libespeak-ng.so.1

# [LAW:no-ambient-temporal-coupling] Pinned, not `:latest`. An unpinned tag makes
# two builds of this same commit different artifacts, and a breaking uv release
# then surfaces as a build failure at a time nobody chose.
COPY --from=ghcr.io/astral-sh/uv:0.9.10 /uv /usr/local/bin/uv

WORKDIR /app

# Declared here rather than beside the other build args, because the engine's
# name now decides two things that must never disagree: which libraries go into
# the image, and which engine boots out of it.
#
# [LAW:one-source-of-truth] `ELVENSPEAK_ENGINE=x` and the extra `elvenspeak[x]`
# are one word — the engines each own an extra named after their registry key,
# and `tests/test_packaging.py` holds the two lists together. So this image
# carries exactly the engine it runs: the piper image has no ONNX runtime for
# Kokoro, the kokoro image has no piper-tts, and neither inherits the other's
# version pins. A name that is not an engine fails here, at the install, rather
# than at the boot two hundred megabytes later.
ARG ELVENSPEAK_ENGINE=piper

# Dependencies before source, so editing a handler does not re-resolve the
# environment. `--frozen` makes the lockfile authoritative: a build that would
# need to change it fails instead of silently resolving something else.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --extra "${ELVENSPEAK_ENGINE}"

COPY elvenspeak/ ./elvenspeak/
COPY main.py ./
RUN uv sync --frozen --no-dev --extra "${ELVENSPEAK_ENGINE}"

# [LAW:no-ambient-temporal-coupling] The two syncs above are the only owners of
# this environment. `uv run` syncs before running unless told not to, so the bake
# step below and the CMD at the end would each be a second owner — and neither
# names the extra, so a sync that pruned to the core dependencies would remove
# the engine this image was built for, after it was installed and before it was
# used. Today's uv does not prune, which is a fact about a version rather than a
# guarantee, and this image should not rest on it.
ENV UV_NO_SYNC=1

# Voices are baked in rather than fetched at boot. A ~60 MB download on every
# restart is a slow start that looks like a hang, and it makes the container
# depend on Hugging Face being reachable to serve traffic it could otherwise
# serve offline. PIPER_ALLOW_DOWNLOAD stays off for the same reason: a missing
# model should fail the deploy, not quietly re-download.
#
# [LAW:one-source-of-truth] ELVENSPEAK_ENGINE, declared once above the install
# and read here, is now the whole of what makes an image internally consistent:
# the libraries installed, the assets baked, and the engine that boots all come
# from that one word. Naming it only on `docker run` would leave those free to
# differ — and an engine asked to open assets nobody installed fails at boot,
# which is loud but is a worse way to learn it than not being able to say it.
# Each engine's own variables, set whichever engine is chosen. They are inert
# for the engine that is not running — `piper.configure` has never heard of
# `KOKORO_MODEL` and vice versa — which is the point of an engine parsing its own
# environment: adding the second engine's settings here costs the first engine
# nothing and changes no shared type.
# Three English voices rather than one, because one voice makes the ElevenLabs
# alias table unanswerable: it maps nine foreign ids onto voices of two
# registers, and an alias whose target is not baked is dropped at startup — so
# the image that baked a single voice resolved none of the nine. Two female and
# one male is the smallest set the table can be honest about, at ~278 MB.
#
# Then two Spanish voices, ~120 MB more, so that a Spanish reply can be spoken
# with Spanish phonemes instead of read aloud by an English voice. That failure
# is not audible as a failure — espeak renders the same sentence
# `ˈola, kˈomo estˈas?` under `es` and `ˈoʊlæ, kəmˈoʊ ɛstˈɑːz?` under `en-us`,
# which plays perfectly and is nonsense. One of each register again, and the
# regions differ because Piper publishes no single-speaker Castilian female
# above `low`: es_ES-sharvard-medium is the only other Castilian at tier and it
# carries two speakers.
#
# REGISTER WAS MEASURED, NOT READ OFF THE NAME — the upstream index carries
# quality, language and speaker count but no gender. Same method and the same
# control voice as `elvenspeak/aliases/piper.toml` records, re-measured on this
# table's own line: pitch is a property of what was said as well as of who said
# it, so the control reads 115 Hz here on the Spanish sentence and 121 Hz there
# on the English one. Two numbers for one voice is the method working, not a
# disagreement between the tables.
#
#   es_MX-claude-high      200 Hz   female — the name reads male to an English
#                                   eye, which is the whole reason for measuring
#   es_AR-daniela-high     185 Hz   female — not baked: 111 MB alone, nearly
#                                   double the others, for a register claude
#                                   already covers
#   es_ES-davefx-medium    122 Hz   male
#   en_US-hfc_male-medium  115 Hz   the control, and where a male speaker belongs
#
# KOKORO BAKES NO SPANISH VOICE, and that is measured too. Its Spanish voices
# reach the phonemizer for free (`kokoro.py` maps the id prefix `e` to `es`),
# but they fail the short utterances openconv's turn loop is made of — the same
# zero-sample defect that keeps bf_emma out of the alias table, an order worse:
# ef_dora 15/16, em_alex 14/16, em_santa 12/16 one- and two-word Spanish lines
# returned no samples, against 0/16 for both Piper voices above. Longer text is
# fine in all of them, which is exactly what makes it dangerous to bake.
#
# [LAW:one-source-of-truth] The first name here is `piper.DEFAULT_VOICE`, and
# `tests/test_dockerfile.py` fails if it is not. First is the one that matters:
# a deployment naming no fallback speaks unknown ids in whichever voice the
# engine offers first, so reordering this line changes what every such
# deployment says.
ARG PIPER_VOICES=en_US-lessac-high,en_US-ljspeech-high,en_US-hfc_male-medium,es_ES-davefx-medium,es_MX-claude-high
ARG KOKORO_VOICES=af_heart,am_michael,bf_emma,bm_george
ARG KOKORO_MODEL=kokoro-v1.0.int8.onnx

# The one setting here that exists only to get through the build. The bake below
# parses the whole environment before it can call `acquire`, and the router's
# parse requires somewhere to ask — so without a value the router image fails to
# build, even though a router acquires nothing and never contacts Consul here.
#
# An ARG and deliberately not an ENV: a default that survived into the image
# would give a router deployed without `ROUTER_CONSUL_URL` a bogus address
# instead of the refusal to boot that setting exists to produce. This one is
# spent during the build and is gone from the artifact. The RFC 2606 `.invalid`
# TLD can never resolve, so the value cannot quietly become somebody's endpoint.
ARG ROUTER_CONSUL_URL=http://built-without-a-fleet.invalid:8500
ENV ELVENSPEAK_ENGINE="${ELVENSPEAK_ENGINE}" \
    PIPER_MODELS_DIR=/app/models \
    PIPER_VOICES="${PIPER_VOICES}" \
    PIPER_ALLOW_DOWNLOAD=0 \
    KOKORO_MODELS_DIR=/app/models \
    KOKORO_VOICES="${KOKORO_VOICES}" \
    KOKORO_MODEL="${KOKORO_MODEL}" \
    KOKORO_ALLOW_DOWNLOAD=0 \
    PORT=5001

# [LAW:one-source-of-truth] The same ARG that installs the extra and bakes the
# assets, made readable on the artifact. Not a second copy of the fact: there is
# no way to change which engine this image carries without changing what this
# label says, because both come from the one word above.
#
# It lives here rather than in the CI workflow's `--label` list because it is a
# fact about the image, not about the build that produced it — an image built by
# hand carries it too, and the question it answers is asked by someone holding a
# container and nothing else. The published tag is `elvenspeak-<engine>:<date>`,
# but a tag is a pointer the registry owns and can be moved; this travels with
# the bytes and still answers after a pull by digest.
#
# CI reads it back out of the *published* image and fails the run unless it names
# the engine that run was asked to build. That is the check that catches a build
# arg which never arrived: the image would build, boot, pass its healthcheck and
# serve fluent audio while rejecting every voice id its callers were told to use.
LABEL gdn.sanctuary.elvenspeak.engine="${ELVENSPEAK_ENGINE}"

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
#
# [LAW:single-enforcer] Reads the status line and nothing else. This used to
# parse the body and decide for itself that an empty `voices` list was a
# failure, which made it a second opinion about fitness — right, but private to
# this file, and Nomad's docker driver never runs it. `/health` owns the verdict
# now, so this and the cluster's own check reach the same answer by reading the
# same thing.
#
# `piper-build-82n`: still a string, and the one `python -c` this file keeps.
# What made the bake step dangerous was that its source named symbols in this
# package, so a rename broke code no import ever reached; this names none — two
# stdlib calls and one environment variable — so that failure is not available
# to it. Giving it an `elvenspeak/health.py` would add a module whose body is
# those same two calls, then reach it only through the venv interpreter and the
# package `__init__`. Measured at 0.09s bare against 0.31s once `elvenspeak` is
# imported: small either way against a 30s interval, so cost is not the argument.
# The argument is that the conversion buys nothing.
#
# What was actually missing is that nothing held this against the endpoint it
# reads. `tests/test_dockerfile.py` now lifts this command out of this file and
# runs it against a real server whose voices are open and one that can speak
# nothing, under an interpreter with no site-packages. So `/health` cannot stop
# distinguishing the two while the image goes on reporting healthy, and this
# cannot quietly acquire an import the base image's `python` could not resolve.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request,os; \
urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ['PORT'])"

CMD ["uv", "run", "--no-dev", "main.py"]
