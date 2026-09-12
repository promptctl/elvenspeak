# What a prober can check, and what a verdict means

Fifty-seven of this service's documented promises can be checked over HTTP against a
running deployment. Fifty-two of them can be *falsified* — decided against
something outside the deployment's own account of itself — and five can only be
checked for agreement with what the deployment said about itself elsewhere.

Those five are not scattered. `HEALTH-1`, `MOD-2`, `MOD-7`, `CAP-5` and `SUB-3` are
every claim about a value the deployment *derives* from its own catalogue — the
union over its voices, the agreement between a status line and the body beside it.
That derivation is exactly how a router advertises the fleet behind it
(`piper-routing-7e2.17` moved the served set onto the voice so it could). So the
epic's stated failure — a prober that returns green on a lying deployment — has an
address: a router misreporting what it fronts passes all five, and nothing else in
this file would notice. The report has to show the split, not a single count.

A second class of promise is missing from the tables altogether, and its absence
matters as much. Every guarantee `elvenspeak/engine.py` makes about the voice
listing — best-first, stable order, ids stable across restarts, capabilities fixed
while offered, no id twice — is unobservable over HTTP, because the server asks the
engine for its voices once at startup and serves that snapshot forever. One of them
cannot be caught at any time by anything: a duplicate id is dropped before an
endpoint exists to be asked, which is `piper-voices-1n0`.

This file fixes three things the rest of `piper-conformance-e16` builds against: the
vocabulary a probe may return, the exit code a run produces, and the identifier each
later issue names when it says which claim it probes.

The sources are `README.md` and `elvenspeak/engine.py`, as the ticket scopes them,
with `elvenspeak/api.py` cited where it is the only place a promise is written
precisely enough to check. Every "observed" note below was read off the real
`create_app` on 2026-09-11, not off the source.

## Verdicts: held, broken, unasked

A probe returns one of three values, and there is deliberately no fourth.

| Verdict | Meaning |
|---|---|
| `held` | The claim was asked and it holds. |
| `broken` | The claim was asked and it does not hold. |
| `unasked` | A precondition failed, so the claim could not be asked. Carries what blocked it. |

**There is no `skipped`.** This repository has already decided that question twice —
`CLAUDE.md` refuses CI path filters because "a skip is indistinguishable from a
pass", and the pytest suite dropped its skip markers because a skip "meant the
suite's most expensive claims silently stopped being made on exactly the machines
least likely to have run them" (`README.md`, "Tests"). A prober is the same shape of
thing: it exists to stand between a defect and a deployment somebody trusts, and a
vocabulary that lets "I did not find out" render like "I found out and it was fine"
defeats it. `unasked` is a result: it prints like any other verdict and it moves the
exit code off 0.

What looks like "not applicable" mostly is not. A voice that declares no
`timestamps` does not make the timestamp claim inapplicable — it selects the other
arm of a two-armed claim, and that arm (`501`, with the message naming the
capability) is asked instead. Both arms are written as separate claims below with
separate ids, so the prober reads which one applies out of the deployment's own
`capabilities` list rather than branching on which engine it thinks it is talking
to. That is the shape `speaks.py` already uses, and its reason is worth repeating:
"an engine no one anticipated is conformant here if it answers, and needs no entry
in any table to be asked."

So `unasked` means a precondition genuinely failed — the voice listing did not
parse, so no per-voice claim could be attempted — and not that a claim turned out
not to apply.

Nor is a deployment refusing a request a precondition failing. Two different things
can stop a claim reading the audio it needed, and the line between them is where the
deployment's own answer begins. A draw — one request a probe composes and sends —
that could never be composed, or that never got an answer at all, leaves the claim
genuinely unasked: the listing gave no voice to address, the request never
completed, and the prober is reporting on itself. A draw that was composed, sent
and answered with a non-200 is the deployment speaking, and for a claim that
promised the request would be served, the refusal *is* the promise not being kept
— which is why `CAP-1`, `CAP-2`, `CAP-3`, `MOD-3`, `MOD-6` and `SUB-5` read one as
`broken`.
Read it the other way and a voice that publishes `timestamps` in its own
`capabilities` list and then refuses both timestamp endpoints is reported as
something the prober could not look at, rather than as the capability lie it is —
and nothing else catches that one, because `CAP-4`'s subject is the voices that
*omit* the capability. The division to carry away is that `unasked` is about what
the prober could not do and `broken` about what the deployment did, and a refusal
is always something the deployment did.

## Exit codes

| Code | Condition |
|---|---|
| `0` | Every claim was asked, and every claim held. |
| `1` | At least one claim is `broken`. |
| `2` | The prober could not run at all: bad arguments, or a base URL that never answered. |
| `3` | Nothing is `broken`, but at least one claim is `unasked`. |

Precedence is `2`, then `1`, then `3`, then `0`: a run that cannot start reports
that rather than a verdict, and a broken claim outranks an unasked one because it is
the stronger finding.

`2` is the code this project already spends on a fault that stops a process before
it does its work — `main.py` exits 2 on a configuration fault, and the
`ELVENSPEAK_*` misspelling refusal does the same (`README.md`, "Running it"). A
prober handed an unreachable URL is in that state, not in a state that says anything
about conformance.

`3` is separate from `0` because `0` has to mean *everything was asked and
everything held*, and separate from `1` because an operator reading a red run needs
to know whether the deployment is wrong or whether the prober could not find out.
Those call for different next actions, and one exit code cannot ask for both.

Exit codes are the contract; how the report is printed is `piper-conformance-e16.3`.
One constraint on that report belongs here, because it follows from the vocabulary:
**every claim must appear in the report with its verdict, `held` ones included.** A
report that prints only failures cannot be read for coverage, and coverage is the
number that tells you whether a green run meant anything.

`piper-conformance-e16.3` implemented `AUTH-1` exactly as its `Evidence` column
reads and `HEALTH-3` more broadly, and the asymmetry is worth recording. The
prober discovers whether a deployment is guarded by asking `GET /v1/voices` with
no key: a 401 or a 403 means a key is configured, those being the two statuses
that mean refused for authentication rather than broken. Any other non-2xx is a
listing defect and `DISC-1` reports it as one. `HEALTH-3` — `/health` answers
without a key even where everything else needs one — is then fully decided
against any guarded deployment whether or not the prober holds a key, so it is
asked there. What decides it is the guard and not the key: against an unguarded
deployment there is no guard for `/health` to answer from outside of, and no key
makes it askable. `AUTH-1` is not decidable that way. Showing a guard is
*closed* rather than merely broken takes a missing key refused, a wrong key
refused, and the real key admitted, and without a key that last control cannot
be established — a refusal from an endpoint that refuses everyone is
indistinguishable from a correctly closed one.

When the prober holds a key the deployment refuses, `AUTH-1` reports `unasked`,
not `broken`. From outside, "the operator supplied the wrong key" and "this
endpoint refuses everyone" are the same observation, and the first is
overwhelmingly likelier; reporting `broken` would blame the deployment for the
invocation, which is how a prober earns a reputation for crying wolf and stops
being run. The blocker names `--key`, sending the reader to their own command
line rather than to the server.

## Evidence: falsifiable, or only self-consistent

Each claim below is marked with what decides it.

**`falsifiable`** — something outside the deployment's account of itself decides:
the bytes that came back, an HTTP status, arithmetic on a sample count, or a
constant published by ElevenLabs. A deployment cannot pass one of these by lying
consistently.

**`self-consistent`** — the claim is checked only against what the deployment said
about itself elsewhere. `GET /v1/models` listing the union of every voice's `models`
is real and worth checking, but a deployment wrong in both places passes it. These
are not weak claims — four of the five are the only way a router's account of the
fleet behind it is observable at all. They are simply not proof on their own, and
because there are only five of them a report that mixes them into one count hides
the entire router question inside a 57-claim green.

Many claims are mixed: the prober picks its input self-consistently (take a
`model_id` the voice does not list) and then judges the outcome falsifiably (it must
be a 422 of a stated shape). Those are marked `falsifiable` — what makes a claim
falsifiable is what decides the verdict, not what chose the input.

## The claims

`Probed by` names the issue that owns the claim. `piper-conformance-e16.8` is absent
from that column because it owns all of them: it runs the finished prober against
the live routed deployment, and the report it commits is the whole table with a
verdict against every row. Where a claim is already asked of a built image by
`speaks.py`, that is noted; see "Overlap with speaks.py" below.

### Health and authentication

| Id | Claim | Source | Evidence | Probed by |
|---|---|---|---|---|
| `HEALTH-1` | `GET /health` answers 200 with a `voices` array of one or more non-empty strings, or 503 with an empty one — status and body agree | README:51, api.py:793 | self-consistent | e16.3 |
| `HEALTH-2` | Every id in `/health`'s `voices` appears in `GET /v1/voices` and can be spoken | api.py:819 | falsifiable | e16.3 |
| `HEALTH-3` | `/health` answers without a key even when one is configured | README:51, api.py:793 | falsifiable (only against a guarded deployment) | e16.3 |
| `AUTH-1` | With a key configured, a guarded endpoint answers 401 `{"detail":"invalid xi-api-key"}` to a missing or wrong `xi-api-key` | api.py:630 | falsifiable (only when the prober holds a key) | e16.3 |
| `AUTH-2` | With no key configured, every endpoint answers without one | README:240 | falsifiable | e16.3 |

Observed: a wrong key and an absent key produce the same 401 and the same body;
`/health` answered 200 in both cases.

**`AUTH-2` and `CAP-4` demanded opposite answers to the same request, and
`piper-conformance-e16.5` settled it by exempting one published sentence.** `AUTH-2`
judged every non-2xx status as a keyless caller turned away, exempting only 404 on
the grounds that a documented path this build does not route belongs to another
claim. But `CAP-4` below *requires* a 501 carrying `{"detail":"this service cannot
report how long each part of an utterance took"}` from a voice that does not declare
`timestamps`, so a deployment doing exactly what one claim in this file asks of it
broke another. The deployment that triggers it is legitimate rather than
hypothetical: the prober's own `_PUBLISHED_FIELDS` note says "an engine may honour
nothing beyond plain speech", and such a deployment was reported broken for
conforming. `AUTH-2` now exempts that 501 the way it exempts 404 — but keyed on the
published sentence, not on the bare status, so a deployment guarding its endpoints
behind an undocumented 501 still breaks the claim. Exempting the status alone would
have turned the fix into a hole. This is the third defect of one family found in this
prober; the other two are recorded under "Per-voice capability honesty" and
"Substitution" below, and all three have the same shape — a conformant deployment
reported broken.

### Voice discovery

| Id | Claim | Source | Evidence | Probed by |
|---|---|---|---|---|
| `DISC-1` | `GET /v1/voices` returns `{"voices":[…]}`, each entry carrying ElevenLabs' fields plus `aliases`, `capabilities`, `models` and `language` | README:47, api.py:1167 | falsifiable | e16.2, e16.3 |
| `DISC-2` | `GET /v1/voices/{id}` returns that voice for an installed id and 404s an id that is not — discovery never substitutes | README:117, api.py:1099 | falsifiable | e16.5 |
| `DISC-3` | `GET /v1/voices/settings/default` returns ElevenLabs' documented defaults: `stability` 0.5, `similarity_boost` 0.75, `style` 0.0, `use_speaker_boost` true, `speed` 1.0 | README:49, api.py:1085 | falsifiable | e16.2 |
| `DISC-4` | `GET /v1/voices/{id}/settings` returns that same object, and 404s an unknown id | README:50, api.py:1109 | falsifiable | e16.2 |
| `DISC-5` | Each voice's `language` is an ISO 639-1 family — the vocabulary `language_code` is compared in | engine.py:202 | falsifiable | e16.5 |

**Every promise `engine.py` makes about the listing itself is missing from this
table, and the absence is the finding.** An engine's voices are promised best-first
(engine.py:406), in an order stable across calls (engine.py:406), under ids stable
across restarts (engine.py:125), with capabilities fixed while the voice is offered
(engine.py:162). None of it is observable over HTTP, because the server never asks
the engine twice: `api.py:498` builds the `Catalog` once at startup and every
discovery response is a projection of that one snapshot. Two `GET /v1/voices` calls
are byte-identical for the life of the process whatever the engine does.

The ordering promise is worse than unverifiable. `voices.py:235` serves
`sorted(self._voices)` — alphabetically by id — while `_chosen` picks the fallback
with `next(iter(voices))`, the engine's own insertion order (`voices.py:85`). So the
engine's best-first ordering is not what `GET /v1/voices` shows, and the first voice
in that listing is not generally the voice an unconfigured deployment substitutes
to. A prober reading the listing to check either one would be reading the wrong
sequence, not merely an unfalsifiable one.

All of these are under "Needs an oracle" below.

### Model ids

| Id | Claim | Source | Evidence | Probed by |
|---|---|---|---|---|
| `MOD-1` | `GET /v1/models` is a bare array, not an object with a `models` key | README:46, api.py:1038 | falsifiable | e16.2 |
| `MOD-2` | The listing is the union of every offered voice's `models`, and never contains `router` | README:145, api.py:1019 | self-consistent | e16.5 |
| `MOD-3` | A `model_id` the resolved voice lists is served, and `model_id` is absent from `x-elvenspeak-ignored` | README:128, api.py:102 | falsifiable | e16.5 |
| `MOD-4` | A `model_id` that names an engine here but not the one speaking the resolved voice is a 422 quoting the value, naming the voice, and listing `served` | README:132, api.py:728 | falsifiable | e16.5 |
| `MOD-5` | A `model_id` naming no engine here is served, and comes back in `x-elvenspeak-ignored` | README:139, api.py:713 | falsifiable | e16.5 |
| `MOD-6` | `voice_id` decides who speaks and `model_id` is read against that — `model_id` never overrides `voice_id` | README:123 | falsifiable | e16.5 |
| `MOD-7` | Each model entry's `languages` are the languages of the voices that id actually reaches | api.py:1279 | self-consistent | e16.5 |

Observed: `model_id: "eleven_turbo_v2"` against a deployment not declaring it
returned 200 with `x-elvenspeak-ignored: model_id`. `MOD-4` and `MOD-5` are one
decision read two ways, so a prober that asks only one of them cannot tell a server
that refuses everything from one that ignores everything.

**`MOD-4` is the only claim that moves between a direct engine and a router**, which
`piper-conformance-e16.5` established by running the prober against both. Against a
direct piper engine it reported 38 `held`, 0 `broken` and 7 `unasked`; against a
router fronting piper and kokoro, 39 `held`, 0 `broken` and 6 `unasked`. Exit code 3
in both cases. The seven unasked against the direct engine are `HEALTH-3`, `AUTH-1`,
`MOD-4`, `CAP-2`, `CAP-4`, `SUB-4` and `SUB-5`, and `MOD-4` is the one that changes
when a second engine appears behind the base URL — the one claim only a multi-engine
deployment can be asked. From outside a single-engine deployment it is not merely
unasked but unaskable, and that is the honest verdict rather than a gap in the
prober: the server answers *elsewhere* for a `model_id` in `served` union `known`,
and `known` is published nowhere, so a prober cannot construct the request without
guessing engine names. Guessing is what would make the verdict a lie.

### Per-voice capability honesty

| Id | Claim | Source | Evidence | Probed by |
|---|---|---|---|---|
| `CAP-1` | A voice declaring `speed` really changes the rate — 2.0 is faster audio | README:100, engine.py:78 | falsifiable | e16.5 — `speaks.py` |
| `CAP-2` | A voice not declaring `speed` names `voice_settings.speed` in `x-elvenspeak-ignored` | README:107 | falsifiable | e16.5 — `speaks.py` |
| `CAP-3` | A voice declaring `timestamps` answers both timestamp endpoints 200, with a non-empty alignment | README:106 | falsifiable | e16.5 — `speaks.py` |
| `CAP-4` | A voice not declaring `timestamps` answers both 501, with `{"detail":"this service cannot report how long each part of an utterance took"}` | README:106, api.py:647 | falsifiable | e16.5 — `speaks.py` |
| `CAP-5` | `GET /v1/models`' `capabilities` is the union over the offered voices | README:111, api.py:1246 | self-consistent | e16.5 |
| `CAP-6` | A parameter the server cannot honour is named in `x-elvenspeak-ignored` rather than dropped — including a body field this build has never heard of | README:22, api.py:267 | falsifiable | e16.5 |
| `CAP-7` | `x-elvenspeak-ignored` is absent, not empty, when everything asked for was honoured | api.py:776 | falsifiable | e16.5 |
| `CAP-8` | A `language_code` the resolved voice does not speak is named in `x-elvenspeak-ignored`; one it does speak is not | api.py:773 | falsifiable | e16.5 |
| `CAP-9` | A blank, whitespace-only or absent `language_code` expresses no preference, and is never reported ignored | api.py:206 | falsifiable | e16.5 |
| `CAP-10` | `en-GB`, `es_MX` and `ES` are the same request — a tag is reduced to its ISO 639-1 family before it is compared | engine.py:89, api.py:220 | falsifiable | e16.5 |

Observed: an invented body field (`invented_2027`) came back named in
`x-elvenspeak-ignored`, so `CAP-6` holds for fields added after this build — which
is the half of rule 2 a fixed list of parameter names would never have caught.

`CAP-9`, `CAP-10` and `REF-6` are the three answers a `language_code` can get, and
they are worth probing together because two of them look alike from outside. A tag
naming *no* language — `""`, `"   "`, absent — is absence, and the caller is told
nothing. A tag naming a *different* language is a preference that could not be met,
and is named back. A value that is not a string at all is refused. Observed: `""`,
`"  "`, `null` and an omitted field all returned 200 with no `x-elvenspeak-ignored`;
`"fr"` returned 200 naming `language_code`; `5` returned 422. `en-GB` was honoured
against an `en` voice, which is `CAP-10`.

**`probe_cap_7` and `probe_cap_9` raised `KeyError` on every run**, which is the
first of two defects `piper-conformance-e16.5` found in the prober it inherited. Both
read the `PLAIN` draw — the request asking for nothing but the text — off the
deployment-wide draw table, and that table never contained a `PLAIN` row, so a run
died before any report could print. It was fixed at the seam rather than in the two
probes: the deployment-wide draws now join the first voice's own table, so there is
exactly one table to look in, and `once` became a derived name for that voice rather
than a second table standing beside it. One consequence is worth recording because it
is easy to undo — three readers that previously swept `(*voices, once)` now sweep the
voices only. They had been double-counting the first voice.

### Substitution

| Id | Claim | Source | Evidence | Probed by |
|---|---|---|---|---|
| `SUB-1` | An unknown `voice_id` on a synthesis endpoint returns audio, with `x-elvenspeak-voice` naming what spoke and `x-elvenspeak-voice-requested` naming what was asked for | README:115, README:64 | falsifiable | e16.5 |
| `SUB-2` | Every synthesis 200 carries `x-elvenspeak-voice`; `x-elvenspeak-voice-requested` appears only when the two differ | README:115, api.py:762 | falsifiable | e16.5 |
| `SUB-3` | `x-elvenspeak-voice` names a voice that `GET /v1/voices` offers | api.py:763 | self-consistent | e16.5 |
| `SUB-4` | With substitution off, an unknown `voice_id` is a 404 rather than audio | README:238 | falsifiable — the other arm of `SUB-1` | e16.5 |
| `SUB-5` | An alias listed on a voice really reaches that voice | README:95 | falsifiable | e16.5 |
| `SUB-6` | A voice id whose bytes are not latin-1 substitutes and is escaped into the header rather than answering 500 | api.py:1118 | falsifiable | e16.5 |

`SUB-1` and `SUB-4` are two arms of one claim, and the prober reports which arm it
got rather than requiring one.

`SUB-5` has no subject behind a router on purpose: the router's alias table is
empty and aliases do not resolve through it (`piper-routing-7e2.15`), so the
listing offers nothing to try. **That is `unasked`, not a claim that holds
trivially** — this paragraph said "holds trivially" until `piper-conformance-e16.5`
implemented it, and the sentence was contradicting the rest of this file. Zero
aliases all resolving correctly is a check that cannot fail, which is the same
defect the "Verdicts" section rejects two pages above and which
`HEALTH-2` had already settled the other way in the same prober: a deployment
offering no voices blocks that claim rather than passing it on an empty set. The
half of the sentence that was right stands: the report shows the empty alias set,
in the blocker, rather than dropping the claim — so a routed and a direct
deployment differ visibly here, which is the difference worth seeing.

**`SUB-2` reported a conformant server broken**, which is the second of the two
defects `piper-conformance-e16.5` found in the prober it inherited. `probe_sub_2`
required `x-elvenspeak-voice-requested` to equal the addressed voice id byte for
byte, but the server escapes everything outside the printable ASCII range, plus
backslash and comma — so the voice id `vøîce-ñ` comes back as the literal text
`v\xf8\xeece-\xf1` and the comparison failed against a server keeping its promise.
The fix compares only where a header can hold the id as it was sent, and checks
presence alone where it cannot: which escape a build picks is that build's own
business, and holding the prober to one spelling would go red on a version skew that
is not a defect. The first attempted fix was wrong, and the reason is instructive
enough to record — it tested latin-1 encodability, and `vøîce-ñ` *is* latin-1. The
rule is the printable ASCII range, not the latin-1 range.

### Output formats

| Id | Claim | Source | Evidence | Probed by |
|---|---|---|---|---|
| `FMT-1` | All 28 published formats are accepted | README:55, formats.py:222 | falsifiable | e16.4 |
| `FMT-2` | Each response's `Content-Type` is that codec's | api.py:858, formats.py:84 | falsifiable | e16.4 |
| `FMT-3` | Each response's bytes carry that format's real signature — RIFF for `wav_*`, an Ogg page for `opus_*`, a frame sync for `mp3_*`, no container at all for `pcm_*`, `ulaw_8000` and `alaw_8000` | README:60 | falsifiable | e16.4 |
| `FMT-4` | A `pcm_*` response's byte count divided by two and by the rate named in the format is the utterance's real duration | README:56, speaks.py:63 | falsifiable | e16.4 |
| `FMT-5` | A `wav_*` response's RIFF header states the rate the format named | README:56 | falsifiable | e16.4 |
| `FMT-6` | An `mp3_*` response starts at a frame sync with no ID3 tag | README:62 | falsifiable | e16.4 |
| `FMT-7` | Omitting `output_format` gives `mp3_44100_128` | README:57, formats.py:147 | falsifiable | e16.4 |
| `FMT-8` | The 422's `supported` array lists exactly the 28 | api.py:691 | falsifiable | e16.4 |

`pcm_*` is the family the prober should ask in whenever it needs a length, and
`speaks.py:63` says why: raw signed 16-bit samples at a rate the name states make a
byte count a sample count, where "every codec that is not PCM makes the length a
fact about the encoder instead of about the utterance." Verifying that an `mp3_*` or
`opus_*` response really runs at its named rate needs a decoder, and is listed under
"Needs an oracle".

### Refusals

| Id | Claim | Source | Evidence | Probed by |
|---|---|---|---|---|
| `REF-1` | An unknown `output_format` is a 422 quoting the value, with the supported set beside it | README:32, api.py:687 | falsifiable | e16.4 |
| `REF-2` | Empty `text` is a 422 | README:33, api.py:161 | falsifiable | e16.4 |
| `REF-3` | Whitespace-only `text` is a 422 | README:33, api.py:163 | falsifiable | e16.4 |
| `REF-4` | `text` longer than 5000 characters is a 422 | README:34, api.py:142 | falsifiable | e16.4 |
| `REF-5` | A refusal carries no `x-elvenspeak-*` headers — the bound on README:115's "every synthesis response" | README:115, observed | falsifiable | e16.4 |
| `REF-6` | A `language_code` that is not a string is a 422 | api.py:224 | falsifiable | e16.4 |
| `REF-7` | An unmodelled body field is kept and reported, never a 422 | api.py:148 | falsifiable | e16.4 |

**The two refusal bodies have different shapes, and e16.4 decided about it: the
prober accepts both and reads neither.** `README.md:32` promises "a `422` quoting
the value you sent" for all of them, and all of them do quote it — but under
different keys:

    REF-1  {"detail": {"message": "unsupported output_format: 'mp3_9999'",
                       "supported": [...]}}
    MOD-4  {"detail": {"message": "model_id ... does not name the engine ...",
                       "served": [...]}}
    REF-2  {"detail": [{"type": "string_too_short", "loc": ["body","text"],
                        "msg": "...", "input": "", "ctx": {"min_length": 1}}]}

The first two are this service's own refusals and are objects; the last is
pydantic's, an array of error records, and it quotes the value under `input`. Both
satisfy the README as written, so `elvenspeak/prober.py` requires neither: no probe
there reads `detail` as an object, and none reads it as an array. Demanding the
object shape would report `broken` against a deployment keeping its documented
promise, and the only way to make a deployment satisfy that demand would be to
rewrite pydantic's refusal into a body no other FastAPI-shaped server sends —
worse compatibility, which is the whole product.

**What replaces the shape is not this file's recommendation, which cannot be taken
literally.** "The offending value appears *somewhere* in the body" is unfalsifiable
for the two refusals that matter most: `REF-2` sends `""` and `REF-3` sends
`"   "`, and a check for those as substrings passes against every 422 ever
written. That is the shape of a check that cannot fail, which is the defect this
epic exists to remove rather than one it may inherit.

The rule the prober enforces instead is that **the body must name what was
wrong** — the offending value where it is distinctive enough to be named back
(`REF-1`'s `mp3_9999`, plus the word `supported`), and the offending *field* where
it is not (`text` for `REF-2`, `REF-3` and `REF-4`; `language_code` for `REF-6`).
It is matched on a word boundary anywhere in the body, which is what lets one rule
over a table serve both shapes rather than a shape read five ways: this service's
object names the field in prose, and pydantic's array names it in `loc`, and a
word boundary finds it in either. The boundary is not decoration — a raw substring
is satisfied by `context` for `text` and by `unsupported` for `supported`, so a
deployment naming nothing would hold `REF-1` through `REF-4`, which is the same
check-that-cannot-fail this section rejects two paragraphs above.
`Refusal.named` carries it, and `tests/test_prober.py` asks the same five claims of
two stand-ins that agree on every fact and share no byte of structure, requiring
all ten verdicts to be `held`.

The rule is stated over refusals generally rather than over these five rows, so
`MOD-4` above inherits it when `piper-conformance-e16.5` lands its model claims.

### Timestamps

| Id | Claim | Source | Evidence | Probed by |
|---|---|---|---|---|
| `TIME-1` | `/with-timestamps` returns `audio_base64`, `alignment`, `normalized_alignment` and `alignment_fidelity` | api.py:1299 | falsifiable | e16.2 |
| `TIME-2` | `/with-timestamps` carries `x-elvenspeak-alignment`, whose value is `word-exact` or `interpolated` | README:187, alignment.py:51 | falsifiable | e16.enf |
| `TIME-3` | `/stream/with-timestamps` emits one JSON object per line, each with its own `alignment_fidelity`, and carries no `x-elvenspeak-alignment` header | README:188, api.py:998 | falsifiable | e16.enf |
| `TIME-4` | Character end times ascend and the last one accounts for the whole utterance — every sample is covered | engine.py:370, speaks.py:694 | falsifiable against `pcm_*` | e16.enf — `speaks.py` |
| `TIME-5` | `alignment` and `normalized_alignment` are the same object | api.py:1303 | falsifiable | e16.enf |
| `TIME-6` | A streamed run's objects lay end to end — each sentence starts where the last one ended | api.py:979 | falsifiable | e16.enf |

Observed: a voice declaring `timestamps` answered `x-elvenspeak-alignment:
word-exact`; the same request against a voice without it answered 501 on both
endpoints. `e16.5` observed the other half of `TIME-3` on its way past:
`/stream/with-timestamps` really does carry no `x-elvenspeak-alignment` header.

These five began as `piper-conformance-e16.5`'s and were moved to a sibling of it
rather than dropped, because they are the one group in that issue whose subject is
not a voice. `e16.5` asks what each voice's own declaration promises; these ask
what one endpoint's body looks like, which is the same question however many
voices a deployment offers. The move costs nothing to ask: `_voice_draws` already
puts both timestamp endpoints and a `pcm_22050` draw to every voice declaring
`timestamps`, so all five are readers over draws that have already been made.

### Routes this server does not serve

| Id | Claim | Source | Evidence | Probed by |
|---|---|---|---|---|
| `ROUTE-1` | An unknown path answers 404 `{"detail":"Not Found"}` | api.py registers no catch-all | falsifiable | e16.6 |
| `ROUTE-2` | A registered path with the wrong method answers 405 `{"detail":"Method Not Allowed"}` | Starlette's default | falsifiable | e16.6 |
| `ROUTE-3` | There is no WebSocket surface at any path | absent from the tree | falsifiable | e16.7 |

These three record what the server does today rather than a promise it makes, which
is the point: they are the epic's two first-contact questions, and both are now
settled by evidence. `ROUTE-1` and `ROUTE-2` answer the unknown-route question,
`ROUTE-3` the WebSocket one.

## What the first probe settled

Both questions the epic said would be answered "by evidence from the SDK run rather
than by opinion" are answerable without the SDK, and were, by driving the real
`create_app`.

**There is no WebSocket surface, anywhere.** `websocket` and `stream-input` do not
occur in any `.py`, `.md`, `.toml` or `.yaml` in the tree. A connect attempt against
`/v1/text-to-speech/{voice}/stream-input` is closed rather than upgraded, and the
same path over POST is a plain 404. So `piper-conformance-e16.7` is deciding whether
to *add* a surface, not whether an existing one conforms — and the README's "What
this deliberately does not do" is where the answer belongs if the decision is no.

**An unknown route falls through to FastAPI's bare `{"detail":"Not Found"}`,** and a
wrong method to Starlette's bare `{"detail":"Method Not Allowed"}`. The second is
not in the epic's description and matters more than it looks: `DELETE
/v1/voices/{id}` is a *documented non-goal* — the README's last line says a stub
pretending to accept it "would be a lie in the shape of an API" — and it currently
answers 405 with a body naming nothing. A caller reaching for a voice-management
endpoint learns only that the verb is wrong, which reads like a client bug rather
than like a server that does not do this. So `piper-conformance-e16.6` has two
fall-throughs to answer, not one, and the 405 is the one that misleads.

Observed, verbatim:

    GET    /                                      -> 404  {"detail":"Not Found"}
    GET    /v1/user/subscription                  -> 404  {"detail":"Not Found"}
    POST   /v1/text-to-speech/{id}/stream-input   -> 404  {"detail":"Not Found"}
    GET    /v1/text-to-speech/{id}                -> 405  {"detail":"Method Not Allowed"}
    DELETE /v1/voices/{id}                        -> 405  {"detail":"Method Not Allowed"}

## Needs an oracle this deployment cannot provide

These are promises about the HTTP surface that a prober cannot decide from the
responses alone. They are not "unasked" — they are outside the prober's reach, and
listing them here is what keeps a later issue from filing one as a gap.

- **An `mp3_*` or `opus_*` response really runs at its named rate.** Needs a
  decoder. `FMT-4` and `FMT-5` cover the two families where the rate is readable
  without one.
- **The audio is mono at exactly the rate it declares.** `speaks.py:36` states this
  exclusion for itself, and the reason carries: the byte count proves a duration
  given a rate, not the rate.
- **The audio is in the right voice.** Also `speaks.py`'s own exclusion (line 43) — an
  utterance in the wrong speaker's voice is fluent, the right length, and identical
  under every arithmetic available here. Separating them needs a speaker embedding,
  which is a model.
- **Word boundaries are measured rather than interpolated** (README:180). The
  prober can check that the timeline accounts for the audio (`TIME-4`) and that
  fidelity is reported (`TIME-2`), but not that a reported boundary is where the
  word really ends. That needs a phonemizer, and it is what `alignment_fidelity`
  exists to tell a caller instead.
- **MP3 matches the real API byte for byte at the front** (README:62). Needs
  ElevenLabs. `FMT-6` — a frame sync and no ID3 — is the checkable remainder.
- **A silent engine answers 502 carrying `x-elvenspeak-silence`** (engine.py:332,
  api.py:604). A prober cannot make an engine go mute, so this is unreachable from
  outside. `tests/test_silence.py` owns it.

Five more are unreachable for one shared reason — the server asks the engine for its
voices once, at startup (`api.py:498`), and every discovery response projects that
snapshot. A prober sees the cache, never the engine.

- **The listing is stable across calls** (engine.py:406) and **a voice's
  capabilities do not change while it is offered** (engine.py:162). Both are
  structurally guaranteed inside one process: the responses are built from the same
  frozen `Voice` objects every time, so no two calls can disagree however the engine
  behaves. Only a restart could expose a violation, and the prober cannot force one.
- **Voice ids are stable across restarts** (engine.py:125). Needs the restart itself.
- **Voices come back best-first** (engine.py:406). `voices.py:235` re-sorts the
  listing alphabetically, so the engine's order never reaches a caller at all.
- **No voice id is offered twice.** This one is not merely out of reach, it is
  permanently masked: `voices.py:225` keys the catalogue by id, so a duplicate is
  dropped last-wins with no error and no log, before any endpoint exists to be
  asked. No probe at any time can catch a violation. `speaks.py:507-513` asks a live
  server this question today and cannot ever fail it. Filed as `piper-voices-1n0`,
  which proposes refusing the boot instead — it is a `[LAW:no-silent-failure]`
  problem in the server, not only an unprobeable one.

The fallback promise is a casualty of the same two lines. `engine.py:406` says a
deployment naming no fallback answers unknown ids in whichever voice its engine
lists first — but `_chosen` reads that from the engine's insertion order
(`voices.py:85`) while the listing is alphabetical, so the first voice a prober can
see is not generally the fallback. It could not be checked even if the prober could
see whether `ELVENSPEAK_FALLBACK_VOICE` was set.

## Not checkable over HTTP at all

Promises in the two sources that describe the build, the boot or the machine rather
than the surface. None of these belong in the prober. Two of them already have a
checker: `smoke.py` runs the image, so a container that refuses its own environment
fails the smoke in seconds (README:446), and one that reached a 200 must have had
its assets baked.

"No network after the first start" (README:232); the RTF and
throughput figures under "Running it" and "What this deliberately does not do";
which engine's libraries and baked assets are in an image (README:394); the
`ELVENSPEAK_*` misspelling refusal and its exit 2 (README:314); Piper's
non-determinism (README:615); voice licensing (README:637); and that an engine
raises from `acquire()` or `open()` rather than mid-request (README:557).

## Overlap with speaks.py

`speaks.py` already asks `CAP-1`, `CAP-2`, `CAP-3`, `CAP-4` and `TIME-4` — plus a
concurrency property no claim here states — of a *built image*, in CI, before it is
pushed. It is not the prober and should not become one: its subject is the artifact
about to be published, it is standard-library-only so it can run before any install,
and its verdict is deliberately binary because a red one stops a publish.

Two checks it makes are not on that list because they cannot fail. `speaks.py:500`
compares two voice listings and `speaks.py:507` looks for a duplicate id, and the
catalogue described above guarantees both pass. They are the reason this file
separates "asked and held" from "could not be asked": a green `speaks.py` run has
always included two questions nothing could have answered wrongly.

The prober's subject is a *deployed base URL*, and its value is the claims
`speaks.py` cannot reach from CI — the 28 formats, the refusal bodies, the
substitution contract, `model_id`'s three answers read per voice, and the property
the epic calls the most worth testing, that a router and a direct engine answer
identically.

Where the two overlap, the claim id is the shared name. If a check moves between
them, it keeps its id, and neither file becomes the place the other's coverage is
inferred from.
