# elvenspeak

Text-to-speech that speaks ElevenLabs' HTTP API, backed by
[Piper](https://github.com/rhasspy/piper) or
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) voices running on the
machine you start it on. Point a client's base URL at this instead of `api.elevenlabs.io`
and it keeps working — no account, no key, no network after the first start.

It exists because the alternative failed in the way third-party dependencies
fail: ElevenLabs disabled the account this project's predecessor proxied, and
every voice call in the homelab went silent at once.

## What "ElevenLabs-compatible" means here

Compatible cannot mean "returns 200 whatever you send". Three rules, and each is
checked by a test:

1. **A parameter that can be honoured, is.** `output_format` selects any of the
   28 published formats. `voice_id` selects a real voice. `voice_settings.speed`
   changes the speech rate. `model_id` selects the engine, when it names the one
   this deployment runs.
2. **A parameter that cannot be honoured is named back to you.** Nothing here
   has an equivalent of `stability` or `seed`, so those are dropped — and the
   response carries `x-elvenspeak-ignored: seed, voice_settings.stability`
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
| `GET /v1/models` | Every `model_id` this deployment accepts — the engine's own name, then the ElevenLabs ids that reach it — and what it will honour. A bare array, as ElevenLabs returns. |
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

`elvenspeak/aliases/` holds one table per engine — `piper.toml`, `kokoro.toml` —
each named after that engine's key in the registry, and an engine reads only the
file named after itself. Every table maps the nine original ElevenLabs voices
onto that engine's own voices, comparable in register, **not** in likeness. The
scoping is what keeps them honest: one shared table can only name one engine's
voices, and a Piper voice name is meaningless inside the Kokoro image. An engine
with no file of its own gets an empty table rather than an error, so an engine
you supplied can have aliases or go without them, and neither has to be
registered anywhere central.

**An alias only takes effect if its target voice is installed**: the table is
filtered at startup to the voices the engine actually loaded, since an alias
pointing at a voice that cannot speak is not an answer. Both images bake every
voice their table names, so all nine ids resolve to real speech out of the box.

`GET /v1/voices` reports each voice's live aliases, so what actually resolves is
readable from the server rather than inferred from a file.

Substitution is never invisible: every synthesis response carries
`x-elvenspeak-voice` naming what actually spoke, and `x-elvenspeak-voice-requested` when
that differs from what you asked for. `GET /v1/voices/{id}` does **not**
substitute — discovery must only report what is really installed.

### Model IDs, and which engine they reach

On the real API `model_id` picks the synthesis model, so here it picks the
engine — and an image holds exactly one engine, so there are three answers
depending on what you named.

- **It names the engine this deployment runs**, either by that engine's own
  name (`piper`, `kokoro`) or by one of the ElevenLabs model ids that engine
  declares. The request is served, and `model_id` is not in
  `x-elvenspeak-ignored`.
- **It names an engine this build has but this deployment is not running** —
  `model_id: "kokoro"` sent to a piper deployment. That is a `422` quoting the
  value and listing what this deployment does serve. It is never quietly
  answered by the engine that is running: that would be a caller hearing a
  different engine than the one they asked for, with nothing in the response
  saying so.
- **It names no engine here** — `eleven_turbo_v2`, or any other value. The
  request is served, the voice decides as it does when `model_id` is omitted,
  and `model_id` comes back in `x-elvenspeak-ignored`. Deliberately not a
  `422`: every stock ElevenLabs client sends a `model_id`, and refusing the
  unrecognised ones would turn most real callers away on their first request.

Which ElevenLabs ids reach an engine is declared by that engine, in the same
`elvenspeak/aliases/<engine>.toml` file that holds its voice aliases, under the
top-level `elevenlabs_models` key. Piper declares `eleven_flash_v2_5` and
`eleven_turbo_v2_5` — ElevenLabs' fast half, which is what piper is here — and
kokoro declares `eleven_multilingual_v2`, the quality half. No id may be claimed
by two engines; a test refuses that before any image is built.

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

Whether Kokoro can answer the timestamp endpoints at all depends on the export
file it opened, not on the engine: the library reports support by checking
whether the ONNX graph has a `duration` output. The published
`model-files-v1.1` exports have it — including the default — so timings come
back. The older `model-files-v1.0` exports do not, and a deployment running one
of those gets a `501` from the timestamp endpoints instead of invented numbers.
The engine reads this off the session it opened, so it can never claim timings
it has no way to produce.

## Running it

Needs **ffmpeg** on `PATH` — it is an executable, so `pyproject.toml` cannot
declare it. `brew install ffmpeg` or `apt install ffmpeg`; the Dockerfile
installs it and fails the build if the codecs are missing.

The kokoro engine phonemizes through **espeak-ng**, which is a native library
and an executable and so cannot be declared either: `brew install espeak-ng` or
`apt install espeak-ng`, and the Dockerfile installs it and fails the build if
it is missing. It has to be the system one — the `espeakng-loader` wheel that
`kokoro-onnx` would otherwise use ships a library that on macOS ignores the data
path it is handed and aborts the process, so the Dockerfile points
`PHONEMIZER_ESPEAK_LIBRARY` at the apt-installed one. A piper-only deployment
needs none of this.

Installing espeak-ng is not enough on its own, because the bundled library is
the one tried first and it aborts before phonemizer's system-wide fallback can
run. Outside Docker, running the kokoro engine means naming the working library
too:

```
PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/lib/libespeak-ng.dylib          # macOS, arm64
PHONEMIZER_ESPEAK_LIBRARY=/usr/local/lib/libespeak-ng.dylib             # macOS, x86_64
PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1   # Linux, x86_64
PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1  # Linux, arm64
```

The engine does not go looking for one itself. A broken wheel on one platform is
a fact about a development machine, and an engine that quietly hunted for a
library that works would be a silent fallback inside the component whose job is
to fail loudly.

```
uv run --extra piper main.py            # or --extra kokoro, to run the other one
```

The extra is not optional in practice, it is the engine. Each engine's libraries
live in an extra named after it, so the name that selects an engine is the name
that installs it — see [Supplying your own engine](#supplying-your-own-engine)
for why they are not core dependencies. Leaving the extra off installs the API
surface with no engine behind it, and the engine you selected fails to open with
a `ModuleNotFoundError` naming the library it wanted. That is deliberate: a
missing wheel is an installation fault, not a configuration one, and reporting it
as a config error would also disguise a genuinely broken library build.

The configured voices download into `models/` on first start. After that
nothing here touches the network.

```
# The server's own, true whichever engine runs:
ELVENSPEAK_ENGINE=piper            # or kokoro; any other name refuses to start,
                                   # and says which names are real
ELVENSPEAK_FALLBACK_VOICE=…        # default: the first voice the engine offers.
                                   # Empty string turns substitution off (404s).
ELVENSPEAK_API_KEY=                # unset accepts every request
ELVENSPEAK_WITHHOLD=               # comma-separated capabilities to switch off:
                                   # timestamps, speed. Naming one the engine
                                   # never had is fine; a name that is not a
                                   # capability refuses to start.
PORT=5001
HOST=0.0.0.0

# The Piper engine's own, read only when it is the engine:
PIPER_VOICES=en_US-lessac-high     # comma-separated; all installed at startup.
                                   # The published image bakes three, so that
                                   # its alias table can answer in both
                                   # registers; one voice is enough to develop
                                   # against and is a third of the download.
PIPER_MODELS_DIR=./models
PIPER_ALLOW_DOWNLOAD=1             # 0 to require models be present already

# The Kokoro engine's own, read only when it is the engine:
KOKORO_VOICES=af_heart,am_michael,bf_emma,bm_george
                                   # comma-separated, and the order matters: the
                                   # first is what ELVENSPEAK_FALLBACK_VOICE defaults to
KOKORO_MODELS_DIR=./models
KOKORO_MODEL=kokoro-v1.0.int8.onnx # which published ONNX export to open
KOKORO_ALLOW_DOWNLOAD=1            # 0 to require assets be present already
```

`ELVENSPEAK_WITHHOLD` is in the server's group and stays there whichever engine
runs. It names capabilities rather than features — the same closed vocabulary the
engines declare against — so `ELVENSPEAK_WITHHOLD=timestamps` means the timestamp
endpoints answer 501 whether Piper, Kokoro or something you wrote is behind them.
The server subtracts what you withheld from whatever the engine declared, so no
engine can disagree with you by forgetting to read a setting.

An engine is also *told* what you withheld, which is where the saving comes from:
Piper patches its ONNX graph at load time to expose durations, and withholding
`timestamps` means it never opens a patched session at all. An engine with no
such economy to make ignores the message and costs you nothing.

It was `ELVENSPEAK_TIMESTAMPS`, and only Piper read it — so switching timestamps
off and then running Kokoro got you timestamps anyway, silently, having used the
documented name. The old name is refused at startup rather than ignored, with the
new spelling in the message.

Piper is the default because it is the first entry in the registry, and because
of what it costs: it runs at an RTF of about 0.03, against Kokoro's 0.77 on the
same class of machine — roughly twenty-five times the compute for the same
second of speech. Kokoro sounds considerably better. That is a trade a
deployment should make on purpose rather than inherit, so it is opt-in.

Voices come from Piper's catalog of 175 — `en_US-amy-medium`,
`en_GB-alba-medium`, `de_DE-thorsten-medium` and so on. Naming several installs
several, and `GET /v1/voices` lists exactly what loaded.

Kokoro's ids are an unrelated namespace: `af_heart`, `am_michael`, `bf_emma`,
`zf_xiaoni`. The first character is the language — `a` American English, `b`
British English, `e` Spanish, `f` French, `h` Hindi, `i` Italian, `j` Japanese,
`p` Brazilian Portuguese, `z` Mandarin — and the second is the gender, `f` or
`m`. All 54 voices ship in a single style pack of about 28 MB, so offering more
of them costs no extra disk or memory; all 54 are selectable through
`KOKORO_VOICES`, and four are offered by default.

The aliases send ElevenLabs ids only to the American pair, `af_heart` and
`am_michael`. The two British voices return zero samples for short utterances —
across sixteen one- and two-word lines, `bf_emma` failed twelve and `bm_george`
seven, against `af_heart`'s none — and a caller sees that as a 500. That is a
defect with its own ticket rather than a property of the aliases; both voices
stay reachable by their own ids. What the table decides is only where a caller
who named no voice of this server's is sent, and sending them to a voice that
fails one short line in two would be choosing that failure for them.

### With Docker

```
docker build -t elvenspeak \
  --build-arg PIPER_VOICES=en_US-lessac-high,en_US-ljspeech-high,en_US-hfc_male-medium .
docker run -p 5001:5001 elvenspeak
```

```
docker build -t elvenspeak-kokoro \
  --build-arg ELVENSPEAK_ENGINE=kokoro \
  --build-arg KOKORO_VOICES=af_heart,am_michael \
  --build-arg KOKORO_MODEL=kokoro-v1.0.int8.onnx .
```

The bake step installs the assets of whichever engine `ELVENSPEAK_ENGINE` names,
and that engine's `*_ALLOW_DOWNLOAD` is off inside the image: a missing model
fails the deploy instead of quietly re-downloading 60 MB on every restart. It
also decides which engine's libraries get installed, so the piper image carries
no Kokoro and the kokoro image carries no piper-tts — one word, `ELVENSPEAK_ENGINE`,
picks the dependencies, the baked assets and the running engine together.

### Pointing openconv at it

```
OPENCONV_TTS_URL=http://127.0.0.1:5001
```

No code change on that side — `crates/openconv-agent/src/tts.rs` is an HTTP
client against whatever that URL names and has no opinion about what answers.

## Supplying your own engine

There are two ways to consume this. Run it as a service and point a base URL at
it, which is what openconv does above. Or install it as a library, register an
engine of your own, and get the whole ElevenLabs surface over it — 28 output
formats, four synthesis endpoints, voice discovery, aliases, character timings,
the ignored-parameter reporting. That is a few thousand lines of compatibility
work you do not have to write badly for the fourth time, and there is nothing to
fork: the registry of engines is an argument, not an import.

```
pip install elvenspeak            # the API surface, and no engine at all
pip install elvenspeak[piper]     # plus the engine that ships here
pip install elvenspeak[kokoro]    # plus the other one
```

The engines are extras because they are not the reusable part. A bare install is
about 21 MB; adding both engines takes it to about 277 MB of ONNX runtimes,
phonemizers and model tooling that your engine will never call and whose pins
are free to conflict with your own. Each extra is named after the value
`ELVENSPEAK_ENGINE` takes, so `elvenspeak[kokoro]` and `ELVENSPEAK_ENGINE=kokoro`
are one word rather than two that can drift.

### What you implement

Two protocols, both in `elvenspeak`, and everything you need is exported from the
package root — you should never have to import a submodule of this package.

`Engine` is four methods: `voices()`, `capabilities()`, `speak()` and
`speak_timed()`. Implement that and construct it yourself, and `create_app` gives
you the server. `Prepared` and `Configure` are the second, optional half: a
`Configure` turns an environment and a set of withheld capabilities into a
`Prepared`, and a `Prepared` has `acquire()` to install assets at build time and
`open()` to build the engine at boot. Implement those too and a deployment can
select your engine by name.

The withheld set is the one thing your engine is told rather than left to read
for itself, and it is `ELVENSPEAK_WITHHOLD` already parsed. Use it to skip work —
build no aligner if `TIMESTAMPS` was withheld — and nothing more: the server
subtracts the same set from whatever you declare, so ignoring it costs you an
economy and never a wrong answer. Do not invent your own name for it. A setting
in your engine's dialect means a deployment that switches something off gets it
back the day it runs a different engine, which is the defect this argument exists
to close.

Registration is a dict:

```python
from elvenspeak import Settings, create_app

settings = Settings.from_env({"tone": configure})   # your engine, your name
settings.engine.acquire()                           # your image's bake step
app = create_app(settings, settings.engine.open())
```

`tests/test_supplied_engine.py` is a complete, working, executed-on-every-test-run
example of everything on this page: an engine unlike either of the ones that ship
here, its own configuration, its own installable assets, and the real API driven
over it. Read that rather than this. Its last test asserts that the whole file
was written against `import elvenspeak` alone, so if the example compiles, the
public surface is sufficient.

### What the protocols cannot tell you

**Your engine owns its native dependencies, and must fail loudly.** `pip install`
is not necessarily the whole obligation: the kokoro engine here needs espeak-ng
present as a system library and needs `PHONEMIZER_ESPEAK_LIBRARY` pointing at a
working copy, and it does not go hunting for one. An engine that quietly probed
for a library that works, or skipped the voices it could not load, would be a
silent fallback inside the component whose whole job is to fail loudly. Raise
from `acquire()` or `open()`. Those are the build and the boot — the two moments
where failing is cheap and visible. A request is neither.

**Capabilities may be a fact about the deployment, not about your engine.** They
are read once at startup and must be constant for the process, but they need not
be constant for the code: the kokoro engine reads `TIMESTAMPS` off the ONNX
session it actually opened, because one published export has a duration output
and another does not. So answer `capabilities()` from what you really opened,
never from the configuration that asked for it. Getting this wrong is
undetectable rather than obviously wrong — the server will report a capability as
honoured and the audio will disagree.

**`Capability` is a closed enum, and stays one.** An engine cannot declare
something the enum does not name, which reads like a limit on outside engines and
is not one. A capability is not a description of what your engine can do; it is
the list of things *this server behaves differently about* — refuse an endpoint,
name a parameter in the `x-elvenspeak-ignored` header. An engine that does
something no endpoint asks for has nothing to declare it to, because no answer
the server gives would change. Widening the enum to strings would let an engine
declare a capability with no consequence, which is a lie shaped like a feature,
and it would make the reported vocabulary engine-supplied — an engine learning
the name of a request body field, which is the coupling the whole seam exists to
prevent. If an endpoint should start honouring something new, that is a change to
the ElevenLabs surface, and adding an enum member is part of making it, not a tax
on having made it.

## Tests

```
uv run --all-extras pytest
```

`--all-extras` rather than the engines named one by one, because the suite runs
the conformance contract against every registered engine and so needs all of
them — and a list of extras spelled here would be one more copy free to forget
the next engine.

The tests that need real models fetch them instead of skipping, so the first run
downloads the Piper voice and the Kokoro assets into `models/` — a few hundred
MB, once, and nothing after that — and needs espeak-ng installed. The skip
markers came out because a skip is indistinguishable from a pass in a summary,
which meant the suite's most expensive claims silently stopped being made on
exactly the machines least likely to have run them.

Two of them are worth knowing about, because they encode findings rather than
expectations. `test_pcm_length_matches_its_declared_rate` runs against the
encoder rather than over HTTP, because **Piper is not deterministic** — the same
sentence synthesized three times gave 36352, 37888 and 37376 samples, since VITS
samples from a noise distribution. That variance also explains why `seed`, which
exists on the real API precisely to make output reproducible, is in the ignored
list rather than implemented. And `test_word_count_mismatch_is_reported_not_hidden`
pins the alignment fallback's honesty, not its numbers.

CI runs that command with `--locked` on every pull request and on pushes to
master, so a drifted `uv.lock` fails the run rather than quietly resolving
something else, and master's branch protection requires that check — named
`pytest` — to pass on a branch up to date with master, so a commit that breaks
a test cannot merge. The workflow installs ffmpeg and espeak-ng from apt, which
no Python metadata can declare; it leaves the espeak library path unnamed,
because `tests/conftest.py` already finds it.

Nothing retries. Roughly one full-suite run in five has aborted at interpreter
teardown from ONNX Runtime after every test passed (piper-tests-ona). A check
that reran until green would hide the only signal that the bug is still there,
and a step that cannot fail cannot block a merge. `tests/test_merge_gate.py`
refuses two edits: narrowing what pytest collects, and silencing what it
returns.

## Voice licensing

The two engines differ in whether this needs checking per voice.

Piper's code is MIT; its voices are licensed individually. The default,
`en_US-lessac-high`, ships from the `rhasspy/piper-voices` Hugging Face
repository, tagged MIT at the repository level. Check a voice's own licence
before switching — some are CC-BY or otherwise restricted, which matters if this
ends up somewhere with commercial terms attached.

Kokoro is uniform, so there is no per-voice check to do: the model
(`hexgrad/Kokoro-82M`) is Apache-2.0, and all 54 voices ship inside that one
release's single style pack under the same terms. The `kokoro-onnx` wrapper the
exports come from is MIT.

## What this deliberately does not do

**No GPU path.** Piper runs comfortably on CPU, which is the point: it replaced
a paid account with something that needs a machine, not a budget. Measured on
four cores, it synthesizes about 30 seconds of speech per second of compute.

**No voice cloning, no voice library, no `DELETE /v1/voices/{id}`.** Those
endpoints manage a hosted account's voices. There is no account here, and a
stub that pretended to accept them would be a lie in the shape of an API.
