# elvenspeak

Text-to-speech that speaks ElevenLabs' HTTP API, backed by
[Piper](https://github.com/rhasspy/piper) voices running on the machine you
start it on. Point a client's base URL at this instead of `api.elevenlabs.io`
and it keeps working — no account, no key, no network after the first start.

It exists because the alternative failed in the way third-party dependencies
fail: ElevenLabs disabled the account this project's predecessor proxied, and
every voice call in the homelab went silent at once.

## What "ElevenLabs-compatible" means here

Compatible cannot mean "returns 200 whatever you send". Three rules, and each is
checked by a test:

1. **A parameter that can be honoured, is.** `output_format` selects any of the
   28 published formats. `voice_id` selects a real voice. `voice_settings.speed`
   changes the speech rate.
2. **A parameter that cannot be honoured is named back to you.** Nothing here
   has an equivalent of `stability` or `seed`, so those are dropped — and the
   response carries `x-elvenspeak-ignored: model_id, seed, voice_settings.stability`
   so you learn it from the response instead of from the audio. The list is
   worked out per request from the engine behind the server, not written down
   anywhere: an engine that cannot vary its speaking rate adds
   `voice_settings.speed` to it, and one that could reproduce a `seed` would
   drop it, with no edit here.
3. **A request that cannot be served is refused.** An unknown `output_format` is
   a `422` quoting the value you sent, not a quiet substitution. So is `text`
   that is empty, whitespace-only, or longer than **5000 characters** — the cap
   exists because `ELVENSPEAK_API_KEY` is unset by default, and without a bound
   one caller can hold a CPU core for as long as it likes.

### Endpoints

| Endpoint | Notes |
|---|---|
| `POST /v1/text-to-speech/{voice_id}` | Whole response in one piece. |
| `POST /v1/text-to-speech/{voice_id}/stream` | Audio arrives as it is synthesized. |
| `POST /v1/text-to-speech/{voice_id}/with-timestamps` | Audio plus character timings. |
| `POST /v1/text-to-speech/{voice_id}/stream/with-timestamps` | One JSON object per sentence. |
| `GET /v1/voices` | The voices installed here, in ElevenLabs' shape. |
| `GET /v1/voices/{voice_id}` | One voice. 404 if it is not installed. |
| `GET /v1/voices/settings/default` | ElevenLabs' documented defaults. |
| `GET /v1/voices/{voice_id}/settings` | Same, per voice. |
| `GET /health` | Status and which voices actually loaded. Never requires a key. |

### Output formats

All 28: `mp3_{22050_32,24000_48,44100_32,44100_64,44100_96,44100_128,44100_192}`,
`opus_48000_{32,64,96,128,192}`, `pcm_{8000,16000,22050,24000,32000,44100,48000}`,
`wav_{…same rates…}`, `ulaw_8000`, `alaw_8000`. Default `mp3_44100_128`, as
ElevenLabs'.

Piper emits 16-bit mono PCM at its voice's own rate; one `ffmpeg` pass resamples
and encodes to whatever you asked for. MP3 comes back with no ID3 tag, starting
at a frame sync, matching the real API byte for byte at the front.

### Voice IDs you did not get from here

An id this server does not know still gets audio, in the fallback voice, because
clients hold ElevenLabs voice ids in saved settings and a server that 404s all
of them replaces nothing.

`aliases.toml` maps the nine original ElevenLabs voices onto comparable Piper
ones — comparable in register, **not** in likeness. **An alias only takes effect
if its target voice is installed**: the table is filtered to `PIPER_VOICES` at
startup, since an alias pointing at a voice that cannot speak is not an answer.
So under the default single-voice setup the table is inert and all nine ids land
on the fallback. To make them live, install their targets:

```
PIPER_VOICES=en_US-lessac-medium,en_US-hfc_female-medium,en_US-kristin-medium,en_US-amy-medium,en_US-kathleen-low,en_US-hfc_male-medium,en_US-joe-medium,en_US-bryce-medium,en_US-john-medium,en_US-danny-low
```

`GET /v1/voices` reports each voice's live aliases, so what actually resolves is
readable from the server rather than inferred from this file.

Substitution is never invisible: every synthesis response carries
`x-elvenspeak-voice` naming what actually spoke, and `x-elvenspeak-voice-requested` when
that differs from what you asked for. `GET /v1/voices/{id}` does **not**
substitute — discovery must only report what is really installed.

### Timestamps, and what is measured

Piper reports per-*phoneme* durations; ElevenLabs publishes per-*character*
timings. These do not correspond one to one, so the mapping is explicit about
its resolution:

- **Word boundaries are measured.** espeak emits a separator between words, so
  each word's start and end come from the model.
- **Characters within a word are interpolated** across that word's measured
  span.
- **When they cannot be, the result says so.** Timings are `interpolated` when
  the phonemizer's word count disagrees with the text's — an expanded number, an
  abbreviation — or when audio arrives that no phoneme accounts for.
  `/with-timestamps` reports it as `x-elvenspeak-alignment`;
  `/stream/with-timestamps` reports it per object as `alignment_fidelity`, since
  fidelity is decided per sentence.

## Running it

Needs **ffmpeg** on `PATH` — it is an executable, so `pyproject.toml` cannot
declare it. `brew install ffmpeg` or `apt install ffmpeg`; the Dockerfile
installs it and fails the build if the codecs are missing.

```
uv run main.py
```

The configured voices download into `models/` on first start. After that
nothing here touches the network.

```
# The server's own, true whichever engine runs:
ELVENSPEAK_ENGINE=piper            # the only one so far; any other name refuses to start
ELVENSPEAK_FALLBACK_VOICE=…        # default: the first voice the engine offers.
                                   # Empty string turns substitution off (404s).
ELVENSPEAK_API_KEY=                # unset accepts every request
PORT=5001
HOST=0.0.0.0

# The Piper engine's own, read only when it is the engine:
PIPER_VOICES=en_US-lessac-medium   # comma-separated; all installed at startup
PIPER_MODELS_DIR=./models
PIPER_ALLOW_DOWNLOAD=1             # 0 to require models be present already
ELVENSPEAK_TIMESTAMPS=1            # 0 saves memory; timestamp endpoints 501
```

`ELVENSPEAK_TIMESTAMPS` is in the second group despite its name — the Piper
engine is what reads it, and another engine would answer the timestamp
endpoints on its own terms.

Voices come from Piper's catalog of 175 — `en_US-amy-medium`,
`en_GB-alba-medium`, `de_DE-thorsten-medium` and so on. Naming several installs
several, and `GET /v1/voices` lists exactly what loaded.

### With Docker

```
docker build -t elvenspeak --build-arg PIPER_VOICES=en_US-lessac-medium .
docker run -p 5001:5001 elvenspeak
```

Voices are baked into the image, and `PIPER_ALLOW_DOWNLOAD` is off inside it: a
missing model fails the deploy instead of quietly re-downloading 60 MB on every
restart.

### Pointing openconv at it

```
OPENCONV_TTS_URL=http://127.0.0.1:5001
```

No code change on that side — `crates/openconv-agent/src/tts.rs` is an HTTP
client against whatever that URL names and has no opinion about what answers.

## Tests

```
uv run --extra dev pytest
```

Only the API tests need a voice in `models/`, and they skip without one.
Everything else runs anywhere.

Two of them are worth knowing about, because they encode findings rather than
expectations. `test_pcm_length_matches_its_declared_rate` runs against the
encoder rather than over HTTP, because **Piper is not deterministic** — the same
sentence synthesized three times gave 36352, 37888 and 37376 samples, since VITS
samples from a noise distribution. That variance also explains why `seed`, which
exists on the real API precisely to make output reproducible, is in the ignored
list rather than implemented. And `test_word_count_mismatch_is_reported_not_hidden`
pins the alignment fallback's honesty, not its numbers.

## Voice licensing

Piper's code is MIT; its voices are licensed individually. The default,
`en_US-lessac-medium`, ships from the `rhasspy/piper-voices` Hugging Face
repository, tagged MIT at the repository level. Check a voice's own licence
before switching — some are CC-BY or otherwise restricted, which matters if this
ends up somewhere with commercial terms attached.

## What this deliberately does not do

**No GPU path.** Piper runs comfortably on CPU, which is the point: it replaced
a paid account with something that needs a machine, not a budget. Measured on
four cores, it synthesizes about 30 seconds of speech per second of compute.

**No voice cloning, no voice library, no `DELETE /v1/voices/{id}`.** Those
endpoints manage a hosted account's voices. There is no account here, and a
stub that pretended to accept them would be a lie in the shape of an API.

**No `model_id`.** A Piper voice *is* its model. Sending one is reported in
`x-elvenspeak-ignored` rather than silently accepted.
