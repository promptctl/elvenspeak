# piper-server

Text-to-speech over HTTP, backed by [Piper](https://github.com/rhasspy/piper) running
locally. Exists so openconv never again loses every call's audio because a third party
disabled an upstream account — see `openconv-openconv-bwy.17` in openconv's `lit`
backlog for the incident this replaces.

## Why this shape

openconv's `Synthesizer` (in `crates/openconv-agent/src/tts.rs`) is an HTTP client
against whatever `OPENCONV_TTS_URL` names, and decodes the response as MPEG audio. It
was originally elvenreader-server, itself a proxy in front of ElevenLabs'
`/v1/text-to-speech/{voice_id}/stream`. This service answers the identical shape —
same path, same request body, same MP3 response — so pointing `OPENCONV_TTS_URL` here
instead is a config change, not a code change, on the openconv side.

Every `voice_id` gets the one voice this server loaded at startup. elvenreader-server's
own voice table has a catch-all for IDs it doesn't recognise rather than erroring, so
callers already expect a substitute voice for an unmapped ID — this just always
substitutes.

## Running it

```
uv run main.py
```

Downloads the configured voice into `models/` on first start (a one-time fetch from
Hugging Face; after that, nothing this service does needs the network). Listens on
`:5001` by default.

```
PIPER_VOICE=en_US-lessac-medium    # <language>-<name>-<quality>, see Piper's VOICES.md
PIPER_MODELS_DIR=./models          # where voice files are cached
PORT=5001
```

Point openconv at it:

```
OPENCONV_TTS_URL=http://127.0.0.1:5001
```

## Voice licensing

Piper voice models are licensed individually, not all the same way as Piper's code
(MIT). The default here, `en_US-lessac-medium`, ships from the `rhasspy/piper-voices`
Hugging Face repository, which is tagged MIT at the repo level. Check a voice's own
license before switching to it — some are CC-BY or otherwise restricted, which matters
if this is ever used somewhere with commercial terms attached.

## What this deliberately doesn't do

No GPU path — Piper runs comfortably on CPU, which is the point: it's what let this
service replace both a paid account and a GPU-bound alternative (CosyVoice) with
something that just needs a machine, not a budget. No multi-voice table — see above.
No deployment manifest — this is a working local service; putting it in the homelab's
Nomad cluster is separate infrastructure work, tracked in `home-infra`.
