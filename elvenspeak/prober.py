"""Every documented promise this service makes, asked of a deployed base URL.

# Why this ships in the package

`speaks.py` asks a *built image* a fixed set of questions in CI and answers yes
or no, because a red answer stops a publish. This asks a *running deployment*
the documented contract and answers claim by claim, because an operator pointing
it at a URL needs to know which promises hold — not whether all of them do. It
ships inside `elvenspeak` rather than in the test tree so a project that
registered its own engine gets it by installing the package, which is the
difference between a promise and a warranty.

Standard library only, and it imports nothing else from this package. The
deployment under test is routinely a *different build* than the one running this
file, so a check comparing the server's answer against this checkout's own
constants would be asserting the implementation against itself and would go red
on a version skew that is not a defect. What a foreign client can observe is the
whole of what may be asked here ([LAW:behavior-not-structure]).

# What "it parsed" is worth, which is nothing

`piper-conformance-e16.2` measured the official SDK's response models and found
them far looser than this server's promises: `Voice` requires only `voice_id`
and holds twenty-five fields optional, `name` and `category` among them. A
server answering `{"voices": [{"voice_id": "x"}]}` satisfies every one of them.
Deleting `name` from the voice payload was verified to parse cleanly with no
`ValidationError` at all.

So a parse is the floor and never the verdict. Every probe below names the
fields this service promises and judges them past the parse; a `held` that rests
on a check which very nearly cannot fail is the defect this whole epic exists to
remove, not a shortcut it tolerates ([LAW:verifiable-goals] — done needs a shape
that failure can miss).

# The vocabulary, and why there is no fourth word

`docs/conformance-claims.md` settles this and is the authority: a probe returns
`held`, `broken` or `unasked`, and there is deliberately no `skipped`. This
repository has refused that word twice already — `CLAUDE.md` refuses CI path
filters because "a skip is indistinguishable from a pass", and the pytest suite
dropped its skip markers for the same reason. A prober stands between a defect
and a deployment somebody trusts, and a vocabulary letting "I did not find out"
render like "I found out and it was fine" defeats it.

That refusal is structural here rather than promised: [`CLAIMS`] is probed in
full on every run, and a failed precondition becomes an [`Unasked`] *value*
carrying what blocked it — never a probe that quietly did not run
([LAW:dataflow-not-control-flow]). Every claim therefore appears in the report
with a verdict, which is what makes a green run readable for coverage.

`unasked` means a precondition genuinely failed. It does not mean a claim turned
out not to apply: what looks like "not applicable" is usually the other arm of a
two-armed claim, and that arm is asked instead.

# The authority for the table

[`CLAIMS`] carries only ids and evidence classes that `docs/conformance-claims.md`
publishes, and `tests/test_prober.py` holds the two equal in both directions
([LAW:one-source-of-truth]). A claim id invented here, or an evidence class
disagreeing with the document, is a test failure rather than a report nobody can
trace back to a promise.

Discovery reads `GET /v1/voices` — for the catalogue and, keyless, for whether
anything is guarded. `GET /v1/models` is deliberately not read yet: no claim
this file owns needs it, and a discovered field with no reader is a map nothing
keeps true. `piper-conformance-e16.5` owns every model claim and adds it beside
[`Deployment.listing`], in the same shape.

Discovery also *speaks*, once per published format, and [`Spoken`] says why that
belongs there rather than in the five claims that read it.

# The two shapes a 422 arrives in, decided here

`piper-conformance-e16.4` was left this open deliberately, so it is settled in
prose before it is settled in code. This service refuses in two different body
shapes, and both keep the promise `README.md:32` actually makes:

    its own       {"detail": {"message": "unsupported output_format: 'mp3_9999'",
                              "supported": [...]}}
    pydantic's    {"detail": [{"type": "string_too_short", "loc": ["body","text"],
                               "input": "", ...}]}

**Both are accepted, and neither is required.** Demanding the object shape would
report `broken` against a deployment keeping its documented promise, and the only
way to make a deployment satisfy it would be to rewrite pydantic's refusal into a
body no other FastAPI-shaped server sends — worse compatibility, which is the
whole product. So no probe here reads `detail` as an object, and none reads it as
an array.

What replaces the shape is a rule the ticket's recommendation does not reach.
"The value you sent appears somewhere in the body" is unfalsifiable for the
refusals that matter most: `REF-2` sends `""` and `REF-3` sends `"   "`, and a
check for those as substrings passes against every 422 ever written — the shape
of a check that cannot fail, which is the defect this epic exists to remove. The
rule is therefore **the body must name what was wrong**: the offending value
where the value is distinctive enough to be named back (`REF-1`'s `mp3_9999`),
and the offending *field* where it is not (`text`, `language_code`). That is
[`Refusal.named`], and it is one rule over a table rather than a shape read five
ways ([LAW:dataflow-not-control-flow]).

# Running it

    python -m elvenspeak.prober http://localhost:8000
    python -m elvenspeak.prober https://tts.example --key "$ELVENSPEAK_API_KEY"

One invocation and one report shape for a local server and a deployed URL, which
is the property that makes a routed deployment comparable against a direct one.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from email.message import Message
from functools import partial
from typing import Any

#: Spoken wherever a claim needs proof that a voice really answers. Short enough
#: that asking it of every voice a fleet offers stays affordable — the slowest
#: engine here spends better than a minute on one utterance — and made of
#: ordinary words so no engine needs a pronunciation dictionary to say it.
PROBE_TEXT = "One two three."

#: Raw signed 16-bit samples at a rate the name states, so a byte count is a
#: sample count and an answer can be audited without a decoder. `speaks.py` asks
#: in this family for the same reason: every codec that is not PCM makes the
#: length a fact about the encoder instead of about the utterance.
PCM_FORMAT = "pcm_22050"
BYTES_PER_SAMPLE = 2

#: How long one request may take. Sized for a synthesis on the slowest engine
#: this project ships rather than for a listing, because one bound covering both
#: has to cover the worse: `speaks.py` records chatterbox answering four
#: characters in 48s on four cpus under Rosetta. A deployment that is merely slow
#: should read as slow, not as broken.
DEFAULT_TIMEOUT_SECONDS = 180.0

#: What this service answers a request carrying no usable key, verbatim. Quoted
#: rather than matched loosely because the claim is that a caller can *tell* an
#: authentication refusal from every other 401 a proxy in front of the deployment
#: might produce.
INVALID_KEY_DETAIL = "invalid xi-api-key"

#: The two statuses that mean *refused for authentication*. Both, because a
#: foreign deployment — or the gateway in front of one — shuts a keyless caller
#: out with either, and reading 401 alone reports a 403-guarded deployment as
#: open. Every other non-2xx stays an answer to be judged: a 500 or a 404 on
#: `/v1/voices` is a listing owed, not a guard ([LAW:single-enforcer] — which
#: status is a closed door is decided here and nowhere else).
AUTH_REFUSALS = (401, 403)

#: The read endpoints `AUTH-2` asks without a key, `{voice_id}` filled from the
#: catalogue: every gettable path README publishes but `/health`, which is
#: `HEALTH-3`'s subject. It cannot be derived from `elvenspeak.api` — that is
#: this checkout's route table, not the remote build's ([LAW:one-source-of-truth]
#: reaches only as far as one program).
DOCUMENTED_READS = (
    "/v1/voices",
    "/v1/models",
    "/v1/voices/settings/default",
    "/v1/voices/{voice_id}",
    "/v1/voices/{voice_id}/settings",
)

#: The synthesis endpoints `AUTH-2` asks without a key, filled the same way:
#: every postable path README publishes, because a deployment gating the
#: streaming ones alone — the costlier half, and the plausible thing to gate —
#: would answer a check of the plain one and read as open. Only the status is
#: judged, so no stream and no timestamp body is ever decoded here.
DOCUMENTED_SYNTHESES = (
    "/v1/text-to-speech/{voice_id}",
    "/v1/text-to-speech/{voice_id}/stream",
    "/v1/text-to-speech/{voice_id}/with-timestamps",
    "/v1/text-to-speech/{voice_id}/stream/with-timestamps",
)


class Blocked(Exception):
    """A precondition failed, so the claim asking for it could not be asked.

    Raised rather than returned so a probe inherits every precondition it touches
    without declaring one. Reaching for the catalogue when the listing did not
    parse, or for a guarded endpoint holding no key, raises here and [`_finding`]
    turns it into the [`Unasked`] verdict carrying this message — so a probe
    author cannot forget to handle a precondition, and a precondition cannot
    silently become a skip ([LAW:no-silent-failure]).
    """


# --------------------------------------------------------------------- verdicts


@dataclass(frozen=True)
class Held:
    """The claim was asked and it holds.

    `evidence` says what was observed, and is required for the reason the other
    two carry a reason: a report of bare verdicts cannot be audited. "3 of 3
    health voices are listed and spoke" is a reader's only way to tell a claim
    that was really asked from one that found nothing to ask about.
    """

    evidence: str
    word = "held"


@dataclass(frozen=True)
class Broken:
    """The claim was asked and it does not hold.

    [LAW:types-are-the-program] `why` is a required positional argument, so a
    broken verdict with no account of what broke is unrepresentable rather than
    discouraged.
    """

    why: str
    word = "broken"


@dataclass(frozen=True)
class Unasked:
    """A precondition failed, so the claim could not be asked.

    Carries what blocked it, and prints like any other verdict. It is a result,
    not an absence: it moves the exit code off `0`.
    """

    blocker: str
    word = "unasked"


Verdict = Held | Broken | Unasked


# -------------------------------------------------------------------- transport


@dataclass(frozen=True)
class Reply:
    """One exchange that ran to the end of its body, whatever its status said.

    [LAW:parse-dont-validate] Built only by [`_ask`], so holding one is proof the
    transport finished and nothing below catches a socket error a second time. A
    refusal is an answer: a 401 or a 422 is what several claims here are *about*,
    so the status comes back to be judged rather than raised.
    """

    status: int
    headers: Message
    body: bytes


@dataclass(frozen=True)
class Target:
    """Where the prober is pointed, and what it holds to get in with."""

    base_url: str
    #: `None` when the operator gave no key, which is a different state from an
    #: empty one: an empty string is a key this deployment will refuse.
    key: str | None
    timeout: float


def _ask(
    target: Target,
    path: str,
    key: str | None,
    *,
    method: str = "GET",
    body: Mapping[str, Any] | None = None,
) -> Reply:
    """`path` answered to the end, or [`Blocked`] naming what went unanswered.

    [LAW:single-enforcer] The one place this file meets the transport, so every
    request is held to one account of what counts as unanswered. The member easy
    to leave out is `http.client.IncompleteRead` — not an `OSError` — which is
    how a stream whose 200 went out before its encoder failed arrives.

    `key` is a parameter and never read from `target`, because the authentication
    claims are experiments *on* the key: `AUTH-1` asks one endpoint with none,
    with a wrong one and with the real one, and a function that reached for an
    ambient key could not express two of those three
    ([LAW:no-shared-mutable-globals] — an ambient key is a secret parameter every
    signature here would be lying about).
    """
    headers = {"content-type": "application/json"} if body is not None else {}
    if key is not None:
        headers["xi-api-key"] = key
    request = urllib.request.Request(
        f"{target.base_url}{path}",
        data=None if body is None else json.dumps(dict(body)).encode(),
        headers=headers,
        method=method,
    )
    try:
        try:
            answer = urllib.request.urlopen(request, timeout=target.timeout)
        except urllib.error.HTTPError as refused:
            answer = refused
        with answer:
            return Reply(answer.status, answer.headers, answer.read())
    except (OSError, http.client.HTTPException) as unfinished:
        raise Blocked(
            f"{method} {path} got no complete answer: {unfinished!r}"
        ) from unfinished


def _posted(
    target: Target,
    voice_id: str,
    key: str | None,
    wire_name: str | None,
    body: Mapping[str, Any],
) -> Reply:
    """`body` posted to `voice_id`'s synthesis endpoint in `wire_name`.

    [LAW:single-enforcer] The one place this file addresses a synthesis, so the
    format a request names and the body it carries are values crossing one
    boundary rather than a function per combination
    ([LAW:dataflow-not-control-flow]). Twenty-eight format draws, five refusals
    and an unmodelled-field request differ in what is passed here and in nothing
    else.

    `wire_name` of `None` names no format at all, which is not the absence of an
    argument but one of the requests this file has to be able to make: `FMT-7`'s
    whole subject is what a deployment answers when nobody chose.
    """
    quoted = urllib.parse.quote(voice_id, safe="")
    named = "" if wire_name is None else f"?output_format={wire_name}"
    return _ask(
        target,
        f"/v1/text-to-speech/{quoted}{named}",
        key,
        method="POST",
        body=body,
    )


def _spoken(target: Target, voice_id: str, key: str | None, wire_name: str | None) -> Reply:
    """[`PROBE_TEXT`] synthesized in `voice_id`, in `wire_name`."""
    return _posted(target, voice_id, key, wire_name, {"text": PROBE_TEXT})


# -------------------------------------------------------------------- discovery


@dataclass(frozen=True)
class ProbedVoice:
    """One voice as the deployment published it, stamped only where it must be.

    [LAW:parse-dont-validate] The stamp is deliberately thin: a JSON object under
    a non-empty string `voice_id`, which is everything a later claim needs to
    address it. The richer fields stay unstamped in `published` on purpose —
    `DISC-1` is the claim that they are all present and well-typed, and a parser
    strict enough to refuse a voice missing its `name` would turn that claim into
    a crash before any report could carry the verdict.
    """

    id: str
    published: Mapping[str, Any]


def _parsed_voices(reply: Reply) -> tuple[ProbedVoice, ...] | str:
    """`reply`'s voices, or the complaint saying why it published none.

    Returns the complaint rather than raising it, because the same fact means two
    different things to two readers and only the reader knows which. To `DISC-1`,
    whose whole claim is that this endpoint returns `{"voices": [...]}`, a
    malformed listing is the promise *broken*. To a claim that merely needed the
    listing to get at a voice, it is a precondition that failed and the claim goes
    `unasked`. One parser, two readings — which is why the verdict is not decided
    here ([LAW:decomposition]: parsing and judging are two jobs).

    The decode happens here rather than in the caller so a body which is not
    JSON at all takes the same route as one that is JSON and wrong. It used to
    raise past every reader instead, which handed `DISC-1` an `unasked` about a
    deployment it had in fact caught red-handed ([LAW:single-enforcer] — what
    counts as an unusable listing is decided in this one function).
    """
    try:
        listing = json.loads(reply.body)
    except ValueError:
        return f"answered a body that is not JSON: {reply.body[:200]!r}"
    try:
        entries = listing["voices"]
    except (KeyError, TypeError):
        return f"answered without a `voices` list: {listing!r:.200}"
    if not isinstance(entries, list):
        return f"published a `voices` that is not a list: {entries!r:.200}"
    voices = []
    for entry in entries:
        if not isinstance(entry, dict):
            return f"published a voice that is not an object: {entry!r:.200}"
        identifier = entry.get("voice_id")
        if not isinstance(identifier, str) or not identifier:
            return (
                f"published a voice with no usable `voice_id`: {entry!r:.200} — "
                "nothing can address it"
            )
        voices.append(ProbedVoice(id=identifier, published=entry))
    return tuple(voices)


def _listed(listing: Reply | Blocked) -> Reply:
    """`listing` as an answer, or the [`Blocked`] that stopped it being one.

    A free function rather than a method because [`discover`] needs the same
    reading *while it is still building* the deployment that would carry it —
    the format draws below cannot be made without a voice id, and a voice id
    cannot be had without this ([LAW:one-source-of-truth]: one reading of the
    listing, whoever is asking).
    """
    if isinstance(listing, Blocked):
        raise listing
    return listing


def _voices_of(listing: Reply | Blocked) -> tuple[ProbedVoice, ...]:
    """The voices `listing` published, or [`Blocked`] saying why it published none."""
    reply = _listed(listing)
    if reply.status != 200:
        raise Blocked(f"GET /v1/voices answered {reply.status}: {reply.body[:200]!r}")
    parsed = _parsed_voices(reply)
    if isinstance(parsed, str):
        raise Blocked(f"GET /v1/voices {parsed}")
    return parsed


def _first_voice(listing: Reply | Blocked, wanted: str) -> str:
    """The id of the first voice offered, or [`Blocked`] if none is.

    [`_voices_of`]'s empty case, named once because three claims need a voice to
    address and an empty catalogue is a precondition failing for all of them.
    `wanted` says what the caller was going to do with it, so the blocker sends a
    reader to the claim rather than to this function.
    """
    voices = _voices_of(listing)
    if not voices:
        raise Blocked(f"this deployment lists no voice, so {wanted}")
    return voices[0].id


def _key_for(target: Target, guarded: bool) -> str | None:
    """The key to send a guarded endpoint, or [`Blocked`] if there is none.

    `None` is a complete answer for an open deployment — it is the key that works
    there — so this returns rather than refuses whenever nothing is guarded.
    """
    if guarded and target.key is None:
        raise Blocked(
            "this deployment guards /v1/voices and the prober was given no "
            "--key, so it cannot reach anything behind the guard"
        )
    return target.key


@dataclass(frozen=True)
class Deployment:
    """A target that answered, and everything it said about itself.

    Discovered once, before any claim is probed, so two claims reading the same
    endpoint cannot be told two different things ([LAW:one-source-of-truth]).

    `listing` holds either the answer to `GET /v1/voices` or the [`Blocked`] that
    stopped the request being made at all — one field with two shapes rather than
    an optional value beside a reason, which could hold an answer and a blocker at
    once and leave a reader asking which is lying. Everything about the catalogue
    is derived from that one answer rather than stored beside it, so no second
    copy exists to drift from it.
    """

    target: Target
    #: `/health`, fetched with no key whatever the operator holds. That is not a
    #: convenience: it is `HEALTH-3`'s experiment, and running it once here is
    #: what lets `HEALTH-1` and `HEALTH-3` judge one answer instead of two that
    #: could disagree.
    health: Reply
    #: Whether `/v1/voices` refused a keyless request. Read off the deployment's
    #: own answer rather than inferred from whether the operator supplied a key,
    #: so the two authentication arms are selected by what is true of the
    #: deployment and not by how it was invoked.
    guarded: bool
    listing: Reply | Blocked
    #: Every published format asked of one voice, or the [`Blocked`] that stopped
    #: them being asked. Drawn in [`discover`] for the reason `listing` is, and
    #: the same shape: one field with two shapes rather than an answer beside a
    #: reason that could both be present at once.
    drawn: "Spoken | Blocked"

    def listed(self) -> Reply:
        """`GET /v1/voices`' answer, or [`Blocked`] if it could not be asked.

        No [`AUTH_REFUSALS`] status arrives here: [`discover`] turns the ones
        meaning "the prober could not get in" into the blocker, so every reply
        this hands back is one the deployment really composed and every claim
        below is free to judge it ([LAW:single-enforcer] — which status is a
        closed door is decided in one place).
        """
        return _listed(self.listing)

    @property
    def voices(self) -> tuple[ProbedVoice, ...]:
        """The offered voices, or [`Blocked`] if the listing could not be read.

        Derived on every read rather than cached beside `listing`, which costs a
        parse of a small document and buys the guarantee that the voices a claim
        sees are the ones this deployment actually published.
        """
        return _voices_of(self.listing)

    def key_or_blocked(self) -> str | None:
        """The key to send a guarded endpoint, or [`Blocked`] if there is none."""
        return _key_for(self.target, self.guarded)

    def spoken(self) -> "Spoken":
        """The format draws, or [`Blocked`] saying why none were made."""
        if isinstance(self.drawn, Blocked):
            raise self.drawn
        return self.drawn


def discover(target: Target) -> Deployment:
    """Ask `target` what it is, or raise [`Blocked`] if it never answered.

    The first two requests are the ones a run cannot proceed without, and a
    transport failure in either stops it: a base URL that will not carry
    `/health` and `/v1/voices` says nothing about conformance, and
    `docs/conformance-claims.md` spends exit code `2` on exactly that state. A
    *refusal* is not that — every status either of them can answer is carried
    forward to be judged, and only a dead socket reaches [`main`] as a `2`.

    Everything after them is the opposite: the listing behind a guard and the
    format draws are things a run reports *about*, so each is caught here and
    carried as a value. A deployment that answered `/health` and refused
    everything else still produces a full report of `unasked` claims, which is
    the report an operator can act on.
    """
    health = _ask(target, "/health", key=None)
    keyless = _ask(target, "/v1/voices", key=None)
    guarded = keyless.status in AUTH_REFUSALS
    # The keyless answer is reused when nothing is guarded, so an open deployment
    # is asked for its listing once rather than twice.
    try:
        listing: Reply | Blocked = keyless if not guarded else _keyed_listing(target)
    except Blocked as shut_out:
        listing = shut_out
    try:
        drawn: Spoken | Blocked = _draw(target, listing, guarded)
    except Blocked as undrawn:
        drawn = undrawn
    return Deployment(
        target=target, health=health, guarded=guarded, listing=listing, drawn=drawn
    )


def _keyed_listing(target: Target) -> Reply:
    """`GET /v1/voices` behind a guard, or [`Blocked`] saying why it stayed shut.

    The two ways in are missing and being refused, and both are the prober's
    problem rather than the deployment's: a server that correctly refuses a key
    it was never given, or was given wrongly, is keeping its promise. Naming
    which of the two happened is the whole value of this function — the blocker
    it raises is what every claim needing the catalogue will report, and "the key
    you gave me was refused" sends an operator to their own invocation while
    "answered without a `voices` list" sends them to the server
    ([LAW:no-silent-failure] — the message has to say where to look).
    """
    if target.key is None:
        raise Blocked(
            "this deployment guards /v1/voices and the prober was given no --key, "
            "so its catalogue could not be read"
        )
    answer = _ask(target, "/v1/voices", key=target.key)
    if answer.status in AUTH_REFUSALS:
        raise Blocked(
            f"GET /v1/voices answered {answer.status} to the xi-api-key the "
            "prober was given, so this deployment's catalogue could not be read "
            "— check --key before reading anything below as a fault of the "
            "deployment"
        )
    return answer


# ---------------------------------------------------------------- the 28 formats


#: ElevenLabs' twenty-eight published `output_format` values. Written out rather
#: than composed from the codecs and rates that occur in them: the set is not a
#: product — `pcm_48000` and `opus_48000_96` are published while `pcm_96` and
#: `opus_8000_128` are not — so anything generating it would offer names nobody
#: publishes.
#:
#: This is the one table here held against this build's own code, and the reason
#: is the opposite of [`DOCUMENTED_READS`]'. Which paths a build routes is that
#: build's own decision, so a remote one may differ without being wrong; the
#: format set belongs to ElevenLabs, both tuples are transcriptions of one
#: published list, and a deployment answering a different 28 is non-conformant by
#: definition. `tests/test_prober.py` holds the two equal, which is what catches
#: a typo in either transcription ([LAW:one-source-of-truth]).
PUBLISHED_FORMATS = (
    "mp3_22050_32",
    "mp3_24000_48",
    "mp3_44100_32",
    "mp3_44100_64",
    "mp3_44100_96",
    "mp3_44100_128",
    "mp3_44100_192",
    "opus_48000_32",
    "opus_48000_64",
    "opus_48000_96",
    "opus_48000_128",
    "opus_48000_192",
    "pcm_8000",
    "pcm_16000",
    "pcm_22050",
    "pcm_24000",
    "pcm_32000",
    "pcm_44100",
    "pcm_48000",
    "wav_8000",
    "wav_16000",
    "wav_22050",
    "wav_24000",
    "wav_32000",
    "wav_44100",
    "wav_48000",
    "ulaw_8000",
    "alaw_8000",
)

#: What a request naming no `output_format` at all must come back as, which is
#: `FMT-7`.
DEFAULT_FORMAT = "mp3_44100_128"

#: A format nobody publishes, shaped exactly like one that is: a real codec and a
#: plausible four-digit rate. A deployment pattern-matching the name instead of
#: looking it up in its own table accepts this, so `REF-1` catches that rather
#: than the easy 422 a malformed word would earn.
UNKNOWN_FORMAT = "mp3_9999"

#: README:34's bound on one request's `text`. `REF-4` sends one character more.
MAX_TEXT_CHARACTERS = 5000

#: How far apart two readings of one utterance's length may be before they are
#: describing different audio. Encoders pad to frame boundaries, so identical
#: durations are not to be expected to the byte: the seven `pcm_*` answers of one
#: real build were observed to agree within 8µs, while the substitution `FMT-4`
#: exists to catch spreads them by 0.21s. Ten milliseconds sits two orders of
#: magnitude from each, which is what makes it a threshold rather than a guess.
_LENGTH_SPREAD_SECONDS = 0.010

#: A published format's name: a codec, a sample rate, and — for the two lossy
#: families — a bitrate.
_WIRE_NAME = re.compile(r"^(mp3|opus|pcm|wav|ulaw|alaw)_(\d{4,5})(?:_\d{2,3})?$")


@dataclass(frozen=True)
class PublishedFormat:
    """One `output_format`, parsed into the facts a claim can check.

    [LAW:parse-dont-validate] Built only by [`_published`], so every value here
    came through [`_WIRE_NAME`] and nothing below asks again whether a name is
    well-formed. `FMT-2` reads `codec`; `FMT-4` and `FMT-5` read `sample_rate`.

    There is deliberately no bitrate field, though five of the names carry one.
    Nothing short of a decoder can read a bitrate off an answer, so a field
    holding it would be a map with no reader — the same reason `/v1/models` stays
    unread in [`discover`] ([FRAMING:representation]).
    """

    wire_name: str
    codec: str
    sample_rate: int


def _published(wire_name: str) -> PublishedFormat:
    """`wire_name` as the facts it states, or a refusal at import.

    Raised rather than skipped, and at import rather than mid-run: a name in
    [`PUBLISHED_FORMATS`] this file cannot read is a typo in this file, and the
    loud way to report that is a module which does not load. A format quietly
    dropped from the table instead would be a promise nothing ever asks about,
    and every claim below would stay green ([LAW:no-silent-failure]).
    """
    named = _WIRE_NAME.match(wire_name)
    if named is None:
        raise ValueError(
            f"{wire_name!r} is not a published format name — a codec, a sample "
            "rate, and a bitrate only for the lossy families"
        )
    return PublishedFormat(
        wire_name=wire_name, codec=named.group(1), sample_rate=int(named.group(2))
    )


FORMATS = tuple(map(_published, PUBLISHED_FORMATS))

#: The three families whose rate a claim can read without a decoder, derived from
#: [`FORMATS`] rather than listed again: raw samples for `pcm_*`, a RIFF header
#: for `wav_*`, and for `mp3_*` only the frame sync `FMT-6` asks after.
PCM_FORMATS = tuple(f for f in FORMATS if f.codec == "pcm")
WAV_FORMATS = tuple(f for f in FORMATS if f.codec == "wav")
MP3_FORMATS = tuple(f for f in FORMATS if f.codec == "mp3")


@dataclass(frozen=True)
class Spoken:
    """One voice asked for every published format, and twice for one of them.

    Drawn once in [`discover`] rather than inside the eight claims that read it,
    because two of them compare one answer against another. A prober
    re-synthesizing per claim would be comparing answers from two different
    draws, and against a backend that samples — which `piper-pipeline-uor`
    established chatterbox is — the difference it measured would be the
    sampler's rather than the deployment's ([LAW:one-source-of-truth]).

    Thirty syntheses a run, and twenty-eight of them are the honest price of "all
    28 are accepted": on the slowest engine this project ships that is the whole
    cost of a run, and a prober sampling the formats would report `held` on a
    claim about every one of them.

    `answers` holds a [`Blocked`] where a draw never finished rather than dropping
    the row, because `FMT-1` is the claim that the format was served at all — its
    failure is the finding, and a missing key would be a claim nobody could ask
    ([LAW:dataflow-not-control-flow]).
    """

    voice_id: str
    answers: Mapping[str, Reply | Blocked]
    #: The answer to a request naming no format at all, which is `FMT-7`'s
    #: subject and not a copy of any row above: what it carries is what this
    #: deployment chose when nobody chose for it.
    default: Reply | Blocked
    #: [`PCM_FORMAT`] asked a second time, which is what makes every length
    #: comparison in this file mean anything — see [`repeats`].
    again: Reply | Blocked

    def served(self, wire_name: str) -> Reply:
        """The audio `wire_name` came back as, or [`Blocked`] if none did."""
        return _served(self.answers[wire_name], f"the {wire_name} draw")

    def repeats(self) -> bool:
        """Whether this voice answered two identical requests identically.

        Asked in audio rather than in lengths, and `speaks.py:401` settles why: a
        backend that samples draws a different *utterance* each time, and two
        draws that happen to come back the same length are still two utterances.
        The bytes are the question — is this deployment's answer a function of the
        request — where a length is only a proxy for it.
        """
        again = _served(self.again, f"the second {PCM_FORMAT} draw")
        return again.body == self.served(PCM_FORMAT).body


def _served(answer: Reply | Blocked, described: str) -> Reply:
    """`answer` as audio a claim can read, or [`Blocked`] saying why it is not.

    [LAW:parse-dont-validate] The one crossing between a draw and the seven
    claims that read bytes, so none of them asks again whether there are any. A
    refusal and an empty 200 leave those claims unasked for one reason: `FMT-1` is
    the claim about whether the format was served, and every claim about the
    *shape* of that audio has no subject until it was ([LAW:single-enforcer] —
    one reading of what "served" means, wherever the question comes from).
    """
    if isinstance(answer, Blocked):
        raise answer
    if answer.status != 200:
        raise Blocked(
            f"{described} answered {answer.status}, so it carries no audio to "
            f"read: {answer.body[:200]!r} — FMT-1 is the claim that reports it"
        )
    if not answer.body:
        raise Blocked(f"{described} answered 200 carrying no audio at all")
    return answer


def _repeating(drawn: Spoken, wanted: str) -> Spoken:
    """`drawn` if this voice repeats itself, or [`Blocked`] if it samples.

    [`Spoken.repeats`]'s negative case, named once for the reason [`_first_voice`]
    is: two claims read one of this voice's lengths against another, and a
    sampling backend is a precondition failing for both. `wanted` says what the
    caller was going to compare, so the blocker sends a reader to the claim
    rather than to this function.
    """
    if not drawn.repeats():
        raise Blocked(
            f"this voice answered two identical {PCM_FORMAT} requests with two "
            f"different utterances, so {wanted}"
        )
    return drawn


def _draw(target: Target, listing: Reply | Blocked, guarded: bool) -> Spoken:
    """Every published format asked of one voice, plus the two extra draws.

    One voice rather than each of them: `HEALTH-2` is the claim that every voice
    speaks, and asking 28 formats apiece would multiply the dominant cost of a
    run by the size of the catalogue to re-prove it. What the format claims are
    about is this deployment's encoder, which is the same one whichever voice
    feeds it.

    Each draw is caught on its own, so one format that never answered leaves the
    other twenty-seven readable and `FMT-1` reports it. A draw that failed is a
    value here, never a row quietly missing ([LAW:no-silent-failure]).
    """
    voice_id = _first_voice(listing, "no format could be asked of anything")
    key = _key_for(target, guarded)

    def drawn(wire_name: str | None) -> Reply | Blocked:
        try:
            return _spoken(target, voice_id, key, wire_name)
        except Blocked as unanswered:
            return unanswered

    return Spoken(
        voice_id=voice_id,
        answers={f.wire_name: drawn(f.wire_name) for f in FORMATS},
        default=drawn(None),
        # Last, so the two draws `repeats` compares are genuinely two asks
        # separated by every other request this function makes.
        again=drawn(PCM_FORMAT),
    )


# ----------------------------------------------------------------------- probes


def _counted(count: int, noun: str) -> str:
    """`count` of `noun`, pluralised, because a person reads this report."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _parsed_health_voices(reply: Reply) -> tuple[str, ...] | str:
    """`reply`'s voice ids, or the complaint saying why it published none.

    Returned rather than raised for the reason [`_parsed_voices`] returns its own:
    to `HEALTH-1`, whose claim *is* the shape of this body, a malformed one is the
    promise broken, while to `HEALTH-2` it is a precondition that failed. An entry
    that is not a usable string is judged here too — `HEALTH-2` looks its ids up
    in a set, so an unstamped one reaches that set and raises `TypeError`
    ([LAW:parse-dont-validate] — the stamp is what keeps the question from
    arriving inland).

    A body that is not JSON is decoded and complained about here for the same
    reason, and reaches both readers as the one word each of them owns.
    """
    try:
        body = json.loads(reply.body)
    except ValueError:
        return f"answered a body that is not JSON: {reply.body[:200]!r}"
    try:
        entries = body["voices"]
    except (KeyError, TypeError):
        return f"answered without a `voices` array: {body!r:.200}"
    if not isinstance(entries, list):
        return f"published a `voices` that is not an array: {entries!r:.200}"
    for entry in entries:
        if not isinstance(entry, str) or not entry:
            return (
                f"published a voice with no usable id: {entry!r:.200} — nothing "
                "can address it"
            )
    return tuple(entries)


def _health_voices(deployment: Deployment) -> tuple[str, ...]:
    """The ids `/health` published, or [`Blocked`] if it published none usably."""
    parsed = _parsed_health_voices(deployment.health)
    if isinstance(parsed, str):
        raise Blocked(f"GET /health {parsed}")
    return parsed


def probe_health_1(deployment: Deployment) -> Verdict:
    """`/health`'s status line and its body agree about whether this can serve.

    Self-consistent, and the document says why that matters: a router misreporting
    the fleet behind it passes this and four claims like it, so the report has to
    show the split rather than fold them into one count.

    The status and the body are judged here rather than taken from
    [`_health_voices`], because they are this claim's subject: a body with no
    usable `voices` array is this promise broken, where to every other claim the
    same answer is only a precondition that failed. The status is read first for
    the reason `DISC-1` reads the listing's first — a 500 answering prose has no
    body to judge, and naming the status is the finding either way.
    """
    status = deployment.health.status
    if status not in (200, 503):
        return Broken(
            f"GET /health answered {status}, which is neither the 200 of a server "
            "fit for traffic nor the 503 of one that is not"
        )
    parsed = _parsed_health_voices(deployment.health)
    if isinstance(parsed, str):
        return Broken(f"GET /health {parsed}")
    voices = parsed
    if status == 200 and not voices:
        return Broken(
            "GET /health answered 200 with an empty `voices` array — a server "
            "that can say nothing is not fit for traffic"
        )
    if status == 503 and voices:
        return Broken(
            f"GET /health answered 503 while offering {_counted(len(voices), 'voice')}"
            " — the status line and the body disagree"
        )
    return Held(
        f"{status} with {_counted(len(voices), 'voice')} — status line and body agree"
    )


def probe_health_2(deployment: Deployment) -> Verdict:
    """Every id `/health` publishes is listed by `/v1/voices` and really speaks.

    Asked of every one of them rather than of a sample: "every id" is the claim,
    and a prober speaking the first voice and reporting `held` has not earned it.
    That costs one utterance per voice, which on the slowest engine here is the
    dominant cost of a run and is the honest price of the claim.

    A deployment offering nothing blocks this rather than passing it vacuously —
    zero ids all present and all speaking is the shape of a check that cannot
    fail, which is the defect this epic exists to remove.
    """
    published = _health_voices(deployment)
    if not published:
        return Unasked(
            "GET /health offered no voices, so there is no id to look up or speak"
        )
    key = deployment.key_or_blocked()
    listed = {voice.id for voice in deployment.voices}
    missing = [name for name in published if name not in listed]
    if missing:
        return Broken(
            f"/health offers {missing} which GET /v1/voices does not list — a "
            "checker reading /health would send traffic for a voice no caller "
            "can address"
        )
    for name in published:
        reply = _spoken(deployment.target, name, key, PCM_FORMAT)
        if reply.status != 200:
            return Broken(
                f"{name!r} is offered by /health and answered {reply.status} when "
                f"asked to speak: {reply.body[:200]!r}"
            )
        if not reply.body:
            return Broken(
                f"{name!r} is offered by /health and answered 200 with no audio"
            )
        if len(reply.body) % BYTES_PER_SAMPLE:
            return Broken(
                f"{name!r} answered {len(reply.body)} bytes of {PCM_FORMAT}, which "
                "is not a whole number of 16-bit samples"
            )
    return Held(
        f"all {_counted(len(published), 'health voice')} are listed by /v1/voices "
        f"and spoke {PROBE_TEXT!r}"
    )


def probe_health_3(deployment: Deployment) -> Verdict:
    """`/health` answers without a key even where everything else needs one.

    Asked only of a guarded deployment, because on an open one there is no guard
    for `/health` to be outside of and the claim has no subject. That is a
    precondition genuinely failing, not a claim excused from applying.

    The evidence is the keyless `/health` [`discover`] already drew — the prober
    sends no key there whatever the operator handed it, so this is a real
    experiment rather than a reading of a request that carried one.
    """
    if not deployment.guarded:
        return Unasked(
            "nothing is guarded here — GET /v1/voices answered a keyless request, "
            "so /health has no guard to be outside of"
        )
    status = deployment.health.status
    if status in AUTH_REFUSALS:
        return Broken(
            f"GET /health refused a keyless request {status} — every health "
            "checker this deployment has must then hold a key, and Nomad's does "
            "not"
        )
    return Held(f"answered {status} keyless while /v1/voices refuses a keyless read")


def probe_auth_1(deployment: Deployment) -> Verdict:
    """A guarded endpoint refuses a missing or wrong key, and admits the real one.

    All three requests are made, and the third is what makes the first two mean
    anything: a server that answered 401 to everything, key or no key, satisfies
    "a wrong key is refused" completely. Holding a key is therefore the
    precondition — without one the prober cannot tell a correctly guarded
    endpoint from a permanently broken one, and a verdict that cannot tell those
    apart is not a verdict.
    """
    if not deployment.guarded:
        return Unasked(
            "nothing is guarded here — GET /v1/voices answered a keyless request, "
            "so there is no refusal to check"
        )
    key = deployment.target.key
    if key is None:
        return Unasked(
            "the prober holds no --key, so a refusal here cannot be told apart "
            "from an endpoint that refuses everyone"
        )
    admitted = _ask(deployment.target, "/v1/voices", key)
    if admitted.status != 200:
        # Not `broken`. The overwhelmingly likely cause is a wrong `--key`, and
        # from out here that is indistinguishable from an endpoint refusing
        # everyone — so the control this claim rests on was never established.
        # Reporting `broken` would blame the deployment for the operator's
        # invocation, which is the prober lying in the one direction that gets it
        # switched off.
        return Unasked(
            f"GET /v1/voices answered {admitted.status} to the key the prober was "
            "given, so there is no admitted request to read the refusals against "
            "— check --key"
        )
    unusable = ((None, "no xi-api-key"), (key + "-wrong", "a wrong xi-api-key"))
    for sent, described in unusable:
        refused = _ask(deployment.target, "/v1/voices", sent)
        # Every 2xx and not 200 alone: a guard admitting an unusable key with a
        # 204 is as open as one answering 200, and reading the narrower surface
        # would drop it into the arm below to be told it refused — this
        # claim's own version of the defect the four rounds before it found.
        if 200 <= refused.status < 300:
            return Broken(
                f"GET /v1/voices answered {refused.status} to {described} "
                "while admitting the real one — the guard is not closed"
            )
        if refused.status != 401:
            # Split from the admission above because one branch could not
            # carry both. A deployment that shut the caller out and one that
            # let an unusable key through are opposite facts, and the single
            # `!= 401` told the first of them "the guard is not closed" —
            # sending an operator to hunt a bypass that is not there, which is
            # the explanation lying in the one direction that gets a prober
            # switched off. [`AUTH_REFUSALS`] is deliberately not reused here:
            # it answers whether a door is closed, which is discovery's
            # question, and `AUTH-1` asks which door — only 401 is that one.
            return Broken(
                f"GET /v1/voices answered {refused.status} to {described}, "
                "not the documented 401 — it did not let the caller in, so "
                "the guard is not standing open; what is wrong is the status "
                "it refuses with"
            )
        # A refusal that is not JSON is judged here rather than raised past this
        # claim: `AUTH-1` promises the detail, so a body that cannot carry one
        # is the promise broken, not a precondition that failed. Decoded to the
        # raw bytes so the complaint below quotes what actually came back.
        try:
            detail: Any = json.loads(refused.body)
        except ValueError:
            detail = refused.body
        if not isinstance(detail, dict) or detail.get("detail") != INVALID_KEY_DETAIL:
            return Broken(
                f"GET /v1/voices refused {described} with {detail!r:.200} rather "
                f"than a {INVALID_KEY_DETAIL!r} detail — a caller cannot tell this "
                "from any other 401 between it and the server"
            )
    return Held(
        f"401 {INVALID_KEY_DETAIL!r} to no key and to a wrong key, 200 to the real one"
    )


def _addressed(templates: tuple[str, ...], voice_id: str) -> tuple[str, ...]:
    """`templates` addressed at `voice_id`.

    Every path is filled the same way, so a per-voice endpoint is a template
    carrying a value rather than a second kind of endpoint with a branch
    selecting it ([LAW:dataflow-not-control-flow]). Reads and syntheses differ in
    the tuple handed here and in nothing else
    ([LAW:one-type-per-behavior] — the tuple is the only thing that varies).
    """
    quoted = urllib.parse.quote(voice_id, safe="")
    return tuple(path.format(voice_id=quoted) for path in templates)


#: One documented endpoint asked without a key: the method, the addressed path,
#: and the body to post, `None` for a read.
_KeylessAsk = tuple[str, str, Mapping[str, Any] | None]


def _keyless_asks(voice_id: str) -> tuple[_KeylessAsk, ...]:
    """Every documented endpoint `AUTH-2` asks, addressed at `voice_id`.

    One sequence rather than a read loop beside a synthesis loop, so what a
    keyless answer may be is decided in one place ([LAW:single-enforcer]) and a
    read and a synthesis differ in the values carried here and in nothing else
    ([LAW:dataflow-not-control-flow]).
    """
    reads: tuple[_KeylessAsk, ...] = tuple(
        ("GET", path, None) for path in _addressed(DOCUMENTED_READS, voice_id)
    )
    syntheses: tuple[_KeylessAsk, ...] = tuple(
        ("POST", f"{path}?output_format={PCM_FORMAT}", {"text": PROBE_TEXT})
        for path in _addressed(DOCUMENTED_SYNTHESES, voice_id)
    )
    return reads + syntheses


def probe_auth_2(deployment: Deployment) -> Verdict:
    """With nothing configured, every documented endpoint answers without a key.

    Asked of [`DOCUMENTED_READS`] and [`DOCUMENTED_SYNTHESES`], because the claim
    is about every endpoint and a guard applied to only the expensive half — or
    to only the streaming part of that half — would pass a narrower check
    completely. One voice, and an utterance per documented synthesis, which is
    what that costs.

    A 404 is the one status not judged: the claim is about keylessness, so a
    documented path this build does not route answers 404 and belongs to some
    other claim. Every other non-2xx is this claim failing, named by its status.
    Judging only a 401 would be assuming a foreign deployment shares this
    checkout's choice of status for a refusal — a proxy in front of it refusing
    403, or the deployment answering 500, is an endpoint a keyless caller did not
    get an answer from either way.
    """
    if deployment.guarded:
        return Unasked(
            "GET /v1/voices refuses a keyless request, so this deployment has a "
            "key configured and the open arm cannot be asked"
        )
    voices = deployment.voices
    if not voices:
        return Unasked(
            "this deployment lists no voice, so the per-voice reads and the "
            "syntheses this claim asks have nothing to address"
        )
    voice_id = voices[0].id
    asks = _keyless_asks(voice_id)
    for method, path, body in asks:
        reply = _ask(deployment.target, path, key=None, method=method, body=body)
        if not (200 <= reply.status < 300 or reply.status == 404):
            return Broken(
                f"{method} {path} answered {reply.status} to a keyless request "
                "while GET /v1/voices allows one — a deployment configuring no "
                "key answers every documented endpoint without one, and this one "
                "does not"
            )
    return Held(
        f"{_counted(len(asks), 'documented endpoint')} in {voice_id!r} all "
        "answered without a key"
    )


#: What `DISC-1` requires of every published voice: the field, the JSON type it
#: must arrive as, and whether an empty one is a real answer. `aliases`,
#: `capabilities` and `models` may legitimately be empty — a router publishes no
#: aliases, and an engine may honour nothing beyond plain speech — while a voice
#: with no name, no category or no language is one a stock client cannot render.
#:
#: [LAW:dataflow-not-control-flow] A table rather than a run of `if`s, so the
#: shape `DISC-1` promises is a value this file reads and a reviewer can check
#: against `api._voice_json` by eye.
_PUBLISHED_FIELDS: tuple[tuple[str, type, bool], ...] = (
    ("voice_id", str, True),
    ("name", str, True),
    ("category", str, True),
    ("aliases", list, False),
    ("capabilities", list, False),
    ("models", list, False),
    ("language", str, True),
)


def probe_disc_1(deployment: Deployment) -> Verdict:
    """Every voice carries ElevenLabs' fields plus this service's four extensions.

    This is the claim the epic's central finding is about. The official SDK holds
    `name` and `category` optional, so a deployment that dropped either would
    hand a real client a `Voice` it parsed happily and cannot render. The check
    is therefore on the named fields and their types, past any parse
    ([LAW:verifiable-goals]).

    The listing's own status and shape are judged here rather than taken from
    [`Deployment.voices`], because they are this claim's subject: a `500` or a
    body with no `voices` array is this promise *broken*, where to every other
    claim the same answer is only a precondition that failed.
    """
    reply = deployment.listed()
    if reply.status != 200:
        return Broken(
            f"GET /v1/voices answered {reply.status} rather than a listing: "
            f"{reply.body[:200]!r}"
        )
    parsed = _parsed_voices(reply)
    if isinstance(parsed, str):
        return Broken(f"GET /v1/voices {parsed}")
    voices = parsed
    if not voices:
        return Unasked(
            "GET /v1/voices published an empty list, so there is no entry whose "
            "fields could be read"
        )
    for voice in voices:
        for field, expected, required in _PUBLISHED_FIELDS:
            if field not in voice.published:
                return Broken(
                    f"voice {voice.id!r} published no {field!r} — a client "
                    "reading the listing cannot see it at all"
                )
            value = voice.published[field]
            if not isinstance(value, expected):
                return Broken(
                    f"voice {voice.id!r} published {field!r} as "
                    f"{type(value).__name__} rather than {expected.__name__}: "
                    f"{value!r:.100}"
                )
            if required and not value:
                return Broken(
                    f"voice {voice.id!r} published an empty {field!r}, which the "
                    "SDK would parse and no client can render"
                )
    named = ", ".join(field for field, _, _ in _PUBLISHED_FIELDS)
    return Held(f"{_counted(len(voices), 'voice')}, each carrying {named}")


# ---------------------------------------------------------------- format probes


def _riff(body: bytes) -> bool:
    """Whether `body` announces a RIFF/WAVE container."""
    return body[:4] == b"RIFF" and body[8:12] == b"WAVE"


def _ogg(body: bytes) -> bool:
    """Whether `body` begins with an Ogg page, which is what Opus arrives in."""
    return body[:4] == b"OggS"


def _id3(body: bytes) -> bool:
    """Whether `body` begins with an ID3v2 tag."""
    return body[:3] == b"ID3"


def _mp3_frame(body: bytes) -> bool:
    """Whether `body` begins at an MP3 frame sync: eleven set bits, no tag.

    Eleven rather than the sixteen a reader might expect, because the twelfth bit
    onward is the MPEG version and layer — `\\xff\\xfb` and `\\xff\\xf3` are both
    real frame starts off this build, at 44.1kHz and 22.05kHz respectively.
    """
    return len(body) > 1 and body[0] == 0xFF and body[1] & 0xE0 == 0xE0


def _bare(body: bytes) -> bool:
    """Whether `body` announces no container, which is what `pcm_*` promises.

    Deliberately blind to the frame sync [`_mp3_frame`] reads, and that blindness
    is the interesting half of the predicate: eleven set bits are all a sync is,
    and real audio in these families begins with them constantly — a `ulaw_8000`
    answer of digital silence is `\\xff\\xff\\xff…` on this very build, observed.
    Reading the sync here would report the correct answer broken.

    So this asks only that no container is announced, and `FMT-4` is what catches
    the substitution it cannot see: a deployment answering one engine's own bytes
    under every `pcm_*` name satisfies this and then states seven different
    durations for one utterance.
    """
    return not (_riff(body) or _ogg(body) or _id3(body))


#: What each codec's bytes must begin with, and the words for saying so. Total
#: over the codecs [`_WIRE_NAME`] admits; a codec added to that pattern and not to
#: this table raises `KeyError` out of a probe, which [`_finding`] deliberately
#: does not catch — a hole here is a bug in the prober, never an `unasked` claim.
_SIGNATURES: dict[str, tuple[Callable[[bytes], bool], str]] = {
    "mp3": (_mp3_frame, "an MP3 frame sync"),
    "opus": (_ogg, "an Ogg page"),
    "wav": (_riff, "a RIFF/WAVE header"),
    "pcm": (_bare, "raw samples in no container"),
    "ulaw": (_bare, "raw samples in no container"),
    "alaw": (_bare, "raw samples in no container"),
}

#: What each codec's answer must be labelled. Written out rather than imported
#: from `elvenspeak.formats` for the reason the whole module imports nothing from
#: this package: the deployment under test is routinely a different build, and a
#: table agreeing with this checkout's renderer by construction could not catch
#: that renderer changing ([LAW:behavior-not-structure]).
_CONTENT_TYPES: dict[str, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "ulaw": "audio/basic",
    "alaw": "audio/x-alaw-basic",
}


def probe_fmt_1(deployment: Deployment) -> Verdict:
    """All 28 published formats are accepted.

    The status and the emptiness are judged here rather than taken from
    [`_served`], because they are this claim's subject: a refusal, or a 200
    carrying nothing, is this promise *broken*, where to the six claims that read
    the audio's shape the same answer is only a precondition that failed. One
    draw, two readings — the pattern [`_parsed_voices`] established.

    A draw that never finished is re-raised instead, because from out here a dead
    socket cannot be told from a refusal, and a verdict that cannot tell those
    apart is not a verdict.
    """
    drawn = deployment.spoken()
    for published in FORMATS:
        answer = drawn.answers[published.wire_name]
        if isinstance(answer, Blocked):
            raise answer
        if answer.status != 200:
            return Broken(
                f"output_format={published.wire_name} answered {answer.status} "
                f"rather than audio: {answer.body[:200]!r} — a caller asking for "
                "a published format got none"
            )
        if not answer.body:
            return Broken(
                f"output_format={published.wire_name} answered 200 carrying no "
                "audio at all, which no caller can play"
            )
    return Held(
        f"all {_counted(len(FORMATS), 'published format')} answered 200 with audio"
    )


def probe_fmt_2(deployment: Deployment) -> Verdict:
    """Each response's `Content-Type` is that codec's.

    The media type is compared and the raw header quoted, so a legal
    `audio/mpeg; charset=binary` is not reported broken over a parameter while the
    complaint still shows what actually arrived.
    """
    drawn = deployment.spoken()
    for published in FORMATS:
        answer = drawn.served(published.wire_name)
        wanted = _CONTENT_TYPES[published.codec]
        if answer.headers.get_content_type() != wanted:
            return Broken(
                f"output_format={published.wire_name} answered "
                f"{answer.headers.get('content-type')!r} rather than {wanted!r} — "
                "a client routing on the type plays it into the wrong decoder"
            )
    return Held(f"{_counted(len(FORMATS), 'format')} each labelled its codec's type")


def probe_fmt_3(deployment: Deployment) -> Verdict:
    """Each response's bytes carry that format's real signature.

    THE CLAIM THIS TICKET IS FOR. A deployment that accepts all 28, labels every
    answer with the content type the caller asked for, and can really produce
    exactly one of them passes every check that reads a status and every check
    that reads a header. Only the bytes give it away.
    """
    drawn = deployment.spoken()
    for published in FORMATS:
        answer = drawn.served(published.wire_name)
        carries, described = _SIGNATURES[published.codec]
        if not carries(answer.body):
            return Broken(
                f"output_format={published.wire_name} answered bytes beginning "
                f"{answer.body[:12]!r}, which is not {described} — the label says "
                f"{published.codec} and the audio is something else"
            )
    return Held(
        f"{_counted(len(FORMATS), 'format')} each carrying its own signature"
    )


def probe_fmt_4(deployment: Deployment) -> Verdict:
    """Every `pcm_*` answer's byte count states the same duration.

    Raw signed 16-bit samples at a rate the name gives, so a byte count is a
    duration with no decoder in the way — and one utterance has one duration, so
    seven rates that disagree about it have at least one whose bytes are not
    samples at the rate it names.

    Read across the rates rather than against a number this file holds, because
    how long an engine takes to say three words is the engine's business and no
    promise at all. The comparison is only meaningful if this voice repeats
    itself, which is what [`_repeating`] establishes first.
    """
    drawn = _repeating(
        deployment.spoken(), "one rate's length cannot be read against another's"
    )
    seconds = {
        published.wire_name: len(drawn.served(published.wire_name).body)
        / BYTES_PER_SAMPLE
        / published.sample_rate
        for published in PCM_FORMATS
    }
    longest = max(seconds, key=seconds.__getitem__)
    shortest = min(seconds, key=seconds.__getitem__)
    spread = seconds[longest] - seconds[shortest]
    if spread > _LENGTH_SPREAD_SECONDS:
        return Broken(
            f"one utterance came back as {seconds[longest]:.3f}s at {longest} and "
            f"{seconds[shortest]:.3f}s at {shortest} — a {spread:.3f}s spread, so "
            "at least one of them is not samples at the rate its name states"
        )
    return Held(
        f"{_counted(len(PCM_FORMATS), 'pcm rate')} all state "
        f"{seconds[longest]:.3f}s within {_LENGTH_SPREAD_SECONDS:g}s"
    )


def _riff_sample_rate(body: bytes) -> int | None:
    """The rate `body`'s `fmt ` chunk states, or `None` if it states none.

    The chunks are walked rather than the rate read at byte 24, which is only
    where it sits in a file whose `fmt ` happens to come first. Real answers are
    not reliably that file — `ffmpeg` writes a `LIST`/`INFO` chunk, and nothing in
    the format forbids a `JUNK` chunk ahead of `fmt ` — so a prober reading the
    canonical offset gets padding and reports a perfectly legal file broken.

    A chunk size the file does not honour cannot mislead this: `ffmpeg` writes
    `data` with a size of `0xffffffff` because it is streaming, which walks `at`
    off the end and ends the search, and `fmt ` is required to precede `data`.

    `None` rather than a raise, because a `wav_*` answer with no readable `fmt `
    chunk is `FMT-5` broken and that claim is the one that should say so.
    """
    at = 12
    while at + 8 <= len(body):
        kind = body[at : at + 4]
        size = int.from_bytes(body[at + 4 : at + 8], "little")
        if kind == b"fmt " and at + 16 <= len(body):
            return int.from_bytes(body[at + 12 : at + 16], "little")
        at += 8 + size + size % 2
    return None


def probe_fmt_5(deployment: Deployment) -> Verdict:
    """A `wav_*` response's RIFF header states the rate the format named."""
    drawn = deployment.spoken()
    for published in WAV_FORMATS:
        answer = drawn.served(published.wire_name)
        stated = _riff_sample_rate(answer.body)
        if stated != published.sample_rate:
            return Broken(
                f"output_format={published.wire_name} answered a RIFF header "
                f"stating {stated} rather than {published.sample_rate} — every "
                "player reads that header, so the audio comes out at the wrong "
                "speed and the wrong pitch"
            )
    return Held(
        f"{_counted(len(WAV_FORMATS), 'wav rate')} each stated in its own header"
    )


def probe_fmt_6(deployment: Deployment) -> Verdict:
    """An `mp3_*` response starts at a frame sync with no ID3 tag.

    README:62's byte-for-byte promise, and it restates `FMT-3`'s `mp3_*` arm on
    purpose: the two read the same bytes against different promises, and a
    deployment one forgotten `ffmpeg` flag away from an ID3 header breaks this one
    by name. What a decoder would have to settle — that the stream really runs at
    its named bitrate — is listed in the document as needing an oracle, and the
    frame sync is the checkable remainder.
    """
    drawn = deployment.spoken()
    for published in MP3_FORMATS:
        answer = drawn.served(published.wire_name)
        if not _mp3_frame(answer.body):
            return Broken(
                f"output_format={published.wire_name} answered bytes beginning "
                f"{answer.body[:12]!r} rather than an MP3 frame sync — an ID3 tag "
                "is the ordinary way here, and it is what README:62 forbids"
            )
    return Held(
        f"{_counted(len(MP3_FORMATS), 'mp3 format')} each starting at a frame sync"
    )


def probe_fmt_7(deployment: Deployment) -> Verdict:
    """Omitting `output_format` gives `mp3_44100_128`.

    Asked as byte identity against this deployment's own answer to the format
    named explicitly, which is the strongest falsifiable form the claim has from
    out here: no decoder is available to read a rate out of an MP3, and a check
    that the default is merely *an* MP3 would pass a deployment defaulting to
    `mp3_22050_32` — an answer at a quarter of the documented rate, under the
    right content type, starting at a frame sync with no tag.

    Identity is only evidence if this voice repeats itself, so [`_repeating`] runs
    first: against a sampling backend two draws differ whatever the default is.
    """
    drawn = _repeating(
        deployment.spoken(),
        f"the default cannot be read against {DEFAULT_FORMAT}'s own answer",
    )
    default = _served(drawn.default, "the draw naming no output_format")
    named = drawn.served(DEFAULT_FORMAT)
    if default.body != named.body:
        return Broken(
            f"a request naming no output_format answered {len(default.body)} bytes "
            f"of {default.headers.get_content_type()!r} where "
            f"output_format={DEFAULT_FORMAT} answered {len(named.body)} bytes of "
            f"{named.headers.get_content_type()!r} — not the same audio, so this "
            f"deployment's default is not {DEFAULT_FORMAT}"
        )
    return Held(
        f"a request naming no output_format answered {DEFAULT_FORMAT} byte for byte"
    )


def _supported_arrays(value: Any) -> list[list[Any]]:
    """Every `supported` array anywhere in `value`, at any depth.

    Searched structurally rather than read at a path, which is what accepting both
    422 shapes without demanding either has to mean in code: this service's own
    refusal carries the array under `detail.message`'s sibling, pydantic's carries
    whatever a deployment put in its error records, and `FMT-8` is entitled to the
    array without being entitled to the shape around it.

    The three arms are JSON's own variants handled exhaustively, which is the one
    thing [LAW:dataflow-not-control-flow] asks a branch to be.
    """
    if isinstance(value, dict):
        here = [value["supported"]] if isinstance(value.get("supported"), list) else []
        return here + [
            found for nested in value.values() for found in _supported_arrays(nested)
        ]
    if isinstance(value, list):
        return [found for nested in value for found in _supported_arrays(nested)]
    return []


def probe_fmt_8(deployment: Deployment) -> Verdict:
    """The 422's `supported` array lists exactly the 28.

    Equality in both directions, because the two failures mislead a caller
    differently: a set that omits a format sends them hunting for another one when
    the one they asked for is really served, and a set that invents one sends them
    to a 422 on the deployment's own advice.

    A refusal carrying no such array at all is this claim broken rather than a
    shape complaint — `REF-1` accepts every shape, and what is missing here is the
    thing a caller was promised beside the refusal.
    """
    answer = _refused(deployment, REFUSALS["REF-1"])
    try:
        detail: Any = json.loads(answer.body)
    except ValueError:
        return Broken(
            f"the 422 refusing {UNKNOWN_FORMAT} answered a body that is not JSON: "
            f"{answer.body[:200]!r}"
        )
    arrays = _supported_arrays(detail)
    if not arrays:
        return Broken(
            f"the 422 refusing {UNKNOWN_FORMAT} carries no `supported` array: "
            f"{answer.body[:200]!r} — a caller is told no is the answer and not "
            "what yes would have been"
        )
    offered = set(arrays[0])
    faults = tuple(
        f"{word} {sorted(names)!r:.200}"
        for word, names in (
            ("omits", set(PUBLISHED_FORMATS) - offered),
            ("invents", offered - set(PUBLISHED_FORMATS)),
        )
        if names
    )
    if faults:
        return Broken(
            f"the 422's `supported` array {' and '.join(faults)} — it is the list "
            "a caller reaches for after a refusal, so a wrong one sends them "
            "somewhere that will refuse them too"
        )
    return Held(f"the 422 lists exactly the {len(PUBLISHED_FORMATS)} published formats")


# --------------------------------------------------------------- refusal probes


@dataclass(frozen=True)
class Refusal:
    """One request this service documents refusing, and how to tell that it did.

    `named` is the rule `piper-conformance-e16.4` settled in place of demanding a
    body shape: **the refusal must name what was wrong** — the offending value
    where it is distinctive enough to be named back, and the offending *field*
    where it is not. The document recommended checking that the value appears
    somewhere in the body, and that recommendation cannot be taken literally here:
    `REF-2` sends `""` and `REF-3` sends `"   "`, so "the value appears in the
    body" is satisfied by every 422 ever written. A check that cannot fail is the
    defect this epic exists to remove, not one it inherits.

    Matched as raw bytes against the whole body rather than at a key, which is how
    both documented shapes answer it and neither is required: the object under
    `detail.message` names the field in prose, and pydantic's array of error
    records names it in `loc`. One rule over a table rather than a shape read five
    ways ([LAW:dataflow-not-control-flow]).
    """

    described: str
    #: The `output_format` to send. Every row but `REF-1` names a published one,
    #: so the refusal a row earns is the one that row is about rather than a
    #: format complaint arriving ahead of it.
    query: str
    body: Mapping[str, Any]
    named: tuple[str, ...]


#: The refusals a documented request earns, keyed by the claim that reports it.
#: `REF-5` reads every row's *headers* and `REF-7` is the claim that a request is
#: **not** refused, so neither is a row here — which is what
#: `tests/test_prober.py` holds by requiring every key to be a claim rather than
#: every claim to be a key.
REFUSALS: dict[str, Refusal] = {
    "REF-1": Refusal(
        described=f"an output_format of {UNKNOWN_FORMAT!r}",
        query=UNKNOWN_FORMAT,
        body={"text": PROBE_TEXT},
        # The value *and* the word: README:32 promises the supported set beside
        # the refusal, and `FMT-8` is the claim about what that set contains.
        named=(UNKNOWN_FORMAT, "supported"),
    ),
    "REF-2": Refusal(
        described="an empty `text`",
        query=PCM_FORMAT,
        body={"text": ""},
        named=("text",),
    ),
    "REF-3": Refusal(
        described="a `text` of nothing but whitespace",
        query=PCM_FORMAT,
        body={"text": "   "},
        named=("text",),
    ),
    "REF-4": Refusal(
        described=f"a `text` of {MAX_TEXT_CHARACTERS + 1} characters",
        query=PCM_FORMAT,
        body={"text": "x" * (MAX_TEXT_CHARACTERS + 1)},
        named=("text",),
    ),
    "REF-6": Refusal(
        described="a `language_code` that is not a string",
        query=PCM_FORMAT,
        body={"text": PROBE_TEXT, "language_code": 7},
        named=("language_code",),
    ),
}

#: The prefix README:115 publishes for this service's own response headers.
_ELVENSPEAK_HEADER = "x-elvenspeak-"

#: A body field no build of this service has ever modelled, which is `REF-7`'s
#: experiment. Dated rather than nonsense, because the claim is about the
#: ElevenLabs field added next year and a caller correct against it.
INVENTED_FIELD = "invented_2027"


def _asked(deployment: Deployment, wire_name: str, body: Mapping[str, Any]) -> Reply:
    """One synthesis asked of a discovered deployment, in its voice and its key.

    Reaches through [`Deployment.spoken`] rather than the catalogue so every claim
    below inherits the precondition the format claims already have: a deployment
    whose voice or key could not be resolved leaves all fifteen `unasked` behind
    one blocker, rather than fifteen readings of one failure
    ([LAW:single-enforcer]).
    """
    return _posted(
        deployment.target,
        deployment.spoken().voice_id,
        deployment.key_or_blocked(),
        wire_name,
        body,
    )


def _refused(deployment: Deployment, refusal: Refusal) -> Reply:
    """`refusal`'s request really refused, or [`Blocked`] if it was answered.

    The reading `FMT-8` needs, which is about what a refusal *carries* and so has
    no subject until there is one. [`probe_refusal`] judges the same request
    instead — one request, two readings, for the reason [`_parsed_voices`] has
    two.
    """
    answer = _asked(deployment, refusal.query, refusal.body)
    if answer.status != 422:
        raise Blocked(
            f"this deployment answered {refusal.described} with {answer.status} "
            "rather than refusing it, so it composed no refusal to read — the "
            "REF- table's own claims report that"
        )
    return answer


def probe_refusal(refusal: Refusal, deployment: Deployment) -> Verdict:
    """`refusal`'s request is refused 422, and the body names what was wrong.

    One function over [`REFUSALS`] rather than five probes: what separates `REF-2`
    from `REF-6` is a body and the word its refusal must carry, and both are
    values ([LAW:one-type-per-behavior] — the five are instances of one claim
    shape, not five kinds of claim). Bound to its row with `functools.partial` in
    [`CLAIMS`], so a row added to the table arrives as a claim rather than as
    another function to write.
    """
    answer = _asked(deployment, refusal.query, refusal.body)
    if answer.status != 422:
        return Broken(
            f"{refusal.described} answered {answer.status} rather than the "
            f"documented 422: {answer.body[:200]!r} — this deployment accepted a "
            "request the contract says it refuses"
        )
    missing = [name for name in refusal.named if name.encode() not in answer.body]
    if missing:
        return Broken(
            f"{refusal.described} was refused 422 by a body naming none of "
            f"{missing}: {answer.body[:200]!r} — a caller cannot read out of it "
            "what to change"
        )
    return Held(f"{refusal.described} was refused 422 naming {list(refusal.named)}")


def _elven_headers(reply: Reply) -> tuple[str, ...]:
    """Every `x-elvenspeak-*` header `reply` carries, lowercased."""
    return tuple(
        name.lower()
        for name in reply.headers.keys()
        if name.lower().startswith(_ELVENSPEAK_HEADER)
    )


def probe_ref_5(deployment: Deployment) -> Verdict:
    """A refusal carries no `x-elvenspeak-*` header.

    The bound on README:115's "every synthesis response carries
    `x-elvenspeak-voice`": a 422 is not a synthesis, nothing spoke, and a header
    naming a voice reports a decision the server never reached.

    Read across the whole [`REFUSALS`] table rather than one row, because a header
    leaking from the refusal path leaks wherever that path is entered and the five
    rows enter it five different ways. A deployment that refused none of them
    leaves this `unasked`: a 200's headers are not this claim's subject, and
    reading them here would file five conformance failures under the one claim
    that has nothing to do with them.
    """
    answers = {
        claim_id: _asked(deployment, refusal.query, refusal.body)
        for claim_id, refusal in REFUSALS.items()
    }
    refusals = {name: a for name, a in answers.items() if a.status == 422}
    if not refusals:
        return Unasked(
            "this deployment refused nothing in the REF- table, so it composed no "
            "refusal whose headers could be read — the REF- claims report that"
        )
    leaked = {
        name: carried
        for name, a in refusals.items()
        if (carried := _elven_headers(a))
    }
    if leaked:
        return Broken(
            f"refusals carry {leaked} — nothing spoke, so there is no voice and no "
            "ignored parameter to report having decided on"
        )
    return Held(
        f"{_counted(len(refusals), 'refusal')} carried no {_ELVENSPEAK_HEADER}* header"
    )


def probe_ref_7(deployment: Deployment) -> Verdict:
    """An unmodelled body field is kept and reported, never refused.

    Two failures, opposite in shape, and a status check waves one of them
    straight through. A deployment choosing `extra="forbid"` refuses a client that
    is correct against a newer ElevenLabs, which is the compatibility this whole
    service is. One that drops the field and answers 200 has told the caller their
    whole request was honoured when part of it was ignored — the quieter half, and
    README:24's rule 2 exactly inverted.
    """
    answer = _asked(
        deployment, PCM_FORMAT, {"text": PROBE_TEXT, INVENTED_FIELD: "probe"}
    )
    if answer.status == 422:
        return Broken(
            f"a body carrying {INVENTED_FIELD!r} was refused 422: "
            f"{answer.body[:200]!r} — a client correct against a newer ElevenLabs "
            "cannot speak to this deployment at all"
        )
    if answer.status != 200:
        return Broken(
            f"a body carrying {INVENTED_FIELD!r} answered {answer.status}: "
            f"{answer.body[:200]!r}"
        )
    reported = answer.headers.get(f"{_ELVENSPEAK_HEADER}ignored", "")
    if INVENTED_FIELD not in reported:
        return Broken(
            f"a body carrying {INVENTED_FIELD!r} was answered 200 with "
            f"{_ELVENSPEAK_HEADER}ignored {reported!r} — the field was dropped "
            "without a word, so the caller believes a parameter was honoured that "
            "was not"
        )
    return Held(
        f"{INVENTED_FIELD!r} was kept and named back in {_ELVENSPEAK_HEADER}ignored"
    )


# ------------------------------------------------------------------ the claims


@dataclass(frozen=True)
class Claim:
    """One documented promise, and how it is decided.

    `evidence` is `falsifiable` when something outside the deployment's own
    account decides it, and `self-consistent` when the claim is checked only
    against what the deployment said about itself elsewhere. The distinction is
    carried per claim and reported as a split because a router misreporting the
    fleet behind it passes every self-consistent claim there is — folding the two
    into one count is what would hide that whole question inside a green run.
    """

    id: str
    evidence: str
    probe: Callable[[Deployment], Verdict]


FALSIFIABLE = "falsifiable"
SELF_CONSISTENT = "self-consistent"

#: The claims this build asks, in the order `docs/conformance-claims.md` lists
#: them. Ids and evidence classes are that document's to define;
#: `tests/test_prober.py` holds this table equal to it in both directions, so an
#: id invented here is a test failure rather than a line in a report that traces
#: back to no promise ([LAW:one-source-of-truth]).
#:
#: The remaining claims belong to `piper-conformance-e16.4` through `e16.7`, and
#: each arrives as a row here rather than as a change to anything below
#: ([LAW:composability]).
CLAIMS: tuple[Claim, ...] = (
    Claim("HEALTH-1", SELF_CONSISTENT, probe_health_1),
    Claim("HEALTH-2", FALSIFIABLE, probe_health_2),
    Claim("HEALTH-3", FALSIFIABLE, probe_health_3),
    Claim("AUTH-1", FALSIFIABLE, probe_auth_1),
    Claim("AUTH-2", FALSIFIABLE, probe_auth_2),
    Claim("DISC-1", FALSIFIABLE, probe_disc_1),
    Claim("FMT-1", FALSIFIABLE, probe_fmt_1),
    Claim("FMT-2", FALSIFIABLE, probe_fmt_2),
    Claim("FMT-3", FALSIFIABLE, probe_fmt_3),
    Claim("FMT-4", FALSIFIABLE, probe_fmt_4),
    Claim("FMT-5", FALSIFIABLE, probe_fmt_5),
    Claim("FMT-6", FALSIFIABLE, probe_fmt_6),
    Claim("FMT-7", FALSIFIABLE, probe_fmt_7),
    Claim("FMT-8", FALSIFIABLE, probe_fmt_8),
    Claim("REF-1", FALSIFIABLE, partial(probe_refusal, REFUSALS["REF-1"])),
    Claim("REF-2", FALSIFIABLE, partial(probe_refusal, REFUSALS["REF-2"])),
    Claim("REF-3", FALSIFIABLE, partial(probe_refusal, REFUSALS["REF-3"])),
    Claim("REF-4", FALSIFIABLE, partial(probe_refusal, REFUSALS["REF-4"])),
    Claim("REF-5", FALSIFIABLE, probe_ref_5),
    Claim("REF-6", FALSIFIABLE, partial(probe_refusal, REFUSALS["REF-6"])),
    Claim("REF-7", FALSIFIABLE, probe_ref_7),
)


# ----------------------------------------------------------------- the report


@dataclass(frozen=True)
class Finding:
    """What one claim turned out to be."""

    claim: Claim
    verdict: Verdict


@dataclass(frozen=True)
class Report:
    """Every claim this run asked, with its verdict.

    Complete by construction: [`probe`] builds one finding per row of [`CLAIMS`],
    so a report that omitted a claim would be a report of a different table.
    """

    target: Target
    findings: tuple[Finding, ...]

    def counts(self, findings: tuple[Finding, ...] | None = None) -> dict[str, int]:
        """How many of `findings` landed on each verdict, every word present.

        Every word appears even at zero, because "0 broken" is the number a
        reader is looking for and an absent line reads like a question nobody
        asked.
        """
        chosen = self.findings if findings is None else findings
        return {
            word: sum(1 for finding in chosen if finding.verdict.word == word)
            for word in (Held.word, Broken.word, Unasked.word)
        }


#: The exit codes, strongest finding first. `2` is absent because it is not a
#: verdict about a deployment — it is the code for a run that never got far
#: enough to have findings, and [`main`] spends it. A `broken` claim outranks an
#: `unasked` one because it is the stronger finding, and `3` is separate from `0`
#: because an operator reading a run needs to know whether the deployment is
#: wrong or whether the prober could not find out.
_EXIT_CODES: tuple[tuple[str, int], ...] = (
    (Broken.word, 1),
    (Unasked.word, 3),
    (Held.word, 0),
)


def exit_code(findings: tuple[Finding, ...]) -> int:
    """The code a run producing `findings` exits with."""
    seen = {finding.verdict.word for finding in findings}
    for word, code in _EXIT_CODES:
        if word in seen:
            return code
    raise AssertionError("a run that asked no claim at all cannot report a verdict")


def _detail(verdict: Verdict) -> str:
    """The one sentence `verdict` carries, whichever verdict it is.

    Each verdict holds exactly one string under a name saying what belongs there
    — `evidence`, `why`, `blocker` — because one name common to all three would
    have to be vague enough to fit all three, and the constructor is where a
    probe author is told what to write. The report needs them uniformly, and this
    is the single place that reconciles the two. The branch is on the sum type's
    own variants, which is the one discriminator
    [LAW:dataflow-not-control-flow] asks a branch to be.
    """
    match verdict:
        case Held(evidence):
            return evidence
        case Broken(why):
            return why
        case Unasked(blocker):
            return blocker


def rendered(report: Report) -> str:
    """`report` as the text a run prints and `e16.8` commits.

    Pure, so the whole report contract is testable with no server anywhere near
    it ([LAW:effects-at-boundaries]). Every claim appears with its verdict, `held`
    ones included: a report printing only failures cannot be read for coverage,
    and coverage is the number that says whether a green run meant anything.
    """
    width = max(len(finding.claim.id) for finding in report.findings)
    verdict_width = max(len(finding.verdict.word) for finding in report.findings)
    evidence_width = max(len(finding.claim.evidence) for finding in report.findings)
    lines = [f"elvenspeak conformance: {report.target.base_url}"]
    for finding in report.findings:
        lines.append(
            f"  {finding.claim.id:<{width}}  "
            f"{finding.claim.evidence:<{evidence_width}}  "
            f"{finding.verdict.word:<{verdict_width}}  {_detail(finding.verdict)}"
        )
    lines.append("")
    lines.append(f"{len(report.findings)} claims: {_tally(report.counts())}")
    # The split, which is the point of carrying an evidence class at all: five of
    # this service's claims can only be checked against its own account of itself,
    # and a router misreporting the fleet behind it passes every one of them.
    for evidence in (FALSIFIABLE, SELF_CONSISTENT):
        chosen = tuple(f for f in report.findings if f.claim.evidence == evidence)
        lines.append(f"{len(chosen)} {evidence}: {_tally(report.counts(chosen))}")
    return "\n".join(lines)


def _tally(counts: dict[str, int]) -> str:
    """`counts` as `3 held, 1 broken, 2 unasked`."""
    return ", ".join(f"{count} {word}" for word, count in counts.items())


# ------------------------------------------------------------------- the run


def _finding(claim: Claim, deployment: Deployment) -> Finding:
    """`claim` asked of `deployment`, whatever happened.

    [LAW:single-enforcer] The one place a probe becomes a finding, so every
    precondition failure in this file turns into the same verdict. Only
    [`Blocked`] is caught: a probe raising anything else is a bug in the prober,
    and reporting a bug as `unasked` would be the prober telling the lie it exists
    to catch ([LAW:no-silent-failure]).
    """
    try:
        return Finding(claim, claim.probe(deployment))
    except Blocked as blocker:
        return Finding(claim, Unasked(str(blocker)))


def probe(target: Target) -> Report:
    """Every claim in [`CLAIMS`], asked of `target`.

    Raises [`Blocked`] only when discovery itself never got an answer, which is
    the state exit code `2` is for. Past that every claim is asked and every
    outcome is a verdict.
    """
    deployment = discover(target)
    return Report(
        target=target,
        findings=tuple(_finding(claim, deployment) for claim in CLAIMS),
    )


def _header_safe_key(given: str) -> str:
    """`given` as a key that can really be sent, or an `argparse` refusal.

    [LAW:parse-dont-validate] The crossing [`_timeout_seconds`] is, for the
    other word an operator hands this file. A key carrying CR or LF reaches
    `http.client.putheader` as a `ValueError` and one outside latin-1 as a
    `UnicodeEncodeError` — neither of them what [`_ask`] catches — so the
    traceback leaves `main` with exit `1`, the code for a deployment whose
    claims are broken, spent on the operator's own invocation. A trailing
    newline off a key file is the ordinary way in.

    Refused rather than stripped: which bytes are the key is the operator's to
    say, and a prober that quietly edits the credential it was given can report
    a refusal the real key would not have earned ([LAW:no-silent-failure]).

    Every control character goes, which is wider than the two that crash: the
    rest are header injection wearing a credential's clothes, and no key this
    service issues contains one.
    """
    unsendable = [c for c in given if ord(c) < 32 or ord(c) == 127]
    if unsendable:
        raise argparse.ArgumentTypeError(
            f"{given!r} carries {unsendable[0]!r}, which cannot be sent in "
            "a header — strip it rather than let the run blame the deployment"
        )
    try:
        given.encode("latin-1")
    except UnicodeEncodeError:
        raise argparse.ArgumentTypeError(
            f"{given!r} is not latin-1, so it cannot be sent as a header value"
        ) from None
    return given


def _timeout_seconds(given: str) -> float:
    """`given` as a timeout a socket can really take, or an `argparse` refusal.

    [LAW:parse-dont-validate] The one crossing between an operator's word and a
    number this file spends on a socket, so nothing inland asks again. A
    negative, NaN or infinite value reaches `socket.settimeout` as a `ValueError`
    or an `OverflowError` — neither of them what [`_ask`] catches — and the
    traceback leaves `main` with exit `1`, the code for a deployment whose
    claims are broken, spent on the operator's own invocation.
    """
    try:
        seconds = float(given)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{given!r} is not a number") from None
    if not (math.isfinite(seconds) and seconds > 0):
        raise argparse.ArgumentTypeError(
            f"{given!r} is not a finite positive number of seconds"
        )
    return seconds


def main(argv: list[str] | None = None) -> int:
    """Probe one deployment, print the report, and return its exit code.

    Argument errors exit `2` because `argparse` already spends that code on
    them, which is the code `docs/conformance-claims.md` assigns to a run that
    could not start — the two agree without this file arranging it.
    """
    parser = argparse.ArgumentParser(
        prog="python -m elvenspeak.prober",
        description="Ask a deployed elvenspeak which of its documented promises hold.",
    )
    parser.add_argument(
        "base_url", help="the deployment to probe, e.g. http://localhost:8000"
    )
    parser.add_argument(
        "--key",
        type=_header_safe_key,
        default=None,
        help="the xi-api-key to send; omit for a deployment that configures none",
    )
    parser.add_argument(
        "--timeout",
        type=_timeout_seconds,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"seconds one request may take (default {DEFAULT_TIMEOUT_SECONDS:g})",
    )
    args = parser.parse_args(argv)
    target = Target(base_url=args.base_url.rstrip("/"), key=args.key, timeout=args.timeout)
    try:
        report = probe(target)
    except Blocked as unreachable:
        # [LAW:no-silent-failure] Loud, on stderr, and distinct from every
        # conformance verdict: a URL that never answered says nothing about the
        # deployment, and reporting it as a table of failures would invent 57
        # findings out of one unanswered socket.
        print(
            f"elvenspeak conformance: {target.base_url} could not be probed: "
            f"{unreachable}",
            file=sys.stderr,
        )
        return 2
    print(rendered(report))
    return exit_code(report.findings)


if __name__ == "__main__":
    sys.exit(main())
