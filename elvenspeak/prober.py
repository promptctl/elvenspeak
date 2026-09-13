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
anything is guarded — and `GET /v1/models`, which every `MOD-` and `CAP-5` claim
reads for the deployment's account of what its voices answer to.

Discovery also *speaks*: once per published format, which [`Spoken`] owns, and
then once per voice per question that voice's own declaration selects, which
[`Asking`] owns. Both belong there rather than in the claims that read them,
because claims compare one answer against another and two draws of one request
are two utterances against a backend that samples. [`Asking`] states the budget
that buys, so this line does not have to.

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
import base64
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


def _in_format(path: str, wire_name: str | None) -> str:
    """`path` asking for `wire_name`, or `path` naming no format at all.

    [LAW:one-source-of-truth] How a format is named on the wire, written once:
    [`_posted`] and [`_keyless_asks`] both address syntheses, and a query key that
    drifted between them would send `AUTH-2` at a URL no other claim uses.
    """
    return path if wire_name is None else f"{path}?output_format={wire_name}"


def _posted(
    target: Target,
    voice_id: str,
    key: str | None,
    wire_name: str | None,
    body: Mapping[str, Any],
    endpoint: str = "",
) -> Reply:
    """`body` posted to `voice_id`'s `endpoint`, in `wire_name`.

    [LAW:dataflow-not-control-flow] The format a request names, the body it
    carries and which of the four synthesis endpoints it goes to are values
    crossing one boundary rather than a function per combination: twenty-eight
    format draws, five refusals, an unmodelled-field request and every per-voice
    draw [`Asking`] makes differ in what is passed here and in nothing else.
    `AUTH-2` addresses its own syntheses and shares only [`_in_format`].

    `endpoint` is what follows the voice id, and its default is the empty string
    because the other three documented syntheses are that same path with a suffix
    — `README.md`'s own table reads that way. The plain endpoint is therefore one
    value of this parameter rather than the absence of one.

    `wire_name` of `None` names no format at all, which is not the absence of an
    argument but one of the requests this file has to be able to make: `FMT-7`'s
    whole subject is what a deployment answers when nobody chose.
    """
    quoted = urllib.parse.quote(voice_id, safe="")
    return _ask(
        target,
        _in_format(f"/v1/text-to-speech/{quoted}{endpoint}", wire_name),
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
    #: `GET /v1/models`, which every `MOD-` claim and `CAP-5` read. Held as the
    #: answer rather than as the ids parsed out of it, because `MOD-1` and `MOD-2`
    #: are claims about the document itself and a parse here would decide them
    #: before they were asked.
    models: Reply | Blocked
    #: Every published format asked of one voice, or the [`Blocked`] that stopped
    #: them being asked. Drawn in [`discover`] for the reason `listing` is, and
    #: the same shape: one field with two shapes rather than an answer beside a
    #: reason that could both be present at once.
    drawn: "Spoken | Blocked"
    #: Every voice asked what its own declaration selects, in the same shape and
    #: for the same reason. [`Asking`] states what it costs.
    asked: "Asking | Blocked"

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

    def asking(self) -> "Asking":
        """The per-voice draws, or [`Blocked`] saying why none were made."""
        if isinstance(self.asked, Blocked):
            raise self.asked
        return self.asked

    def listed_models(self) -> Reply:
        """`GET /v1/models`' answer, or [`Blocked`] if it could not be asked."""
        if isinstance(self.models, Blocked):
            raise self.models
        return self.models


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
    try:
        models: Reply | Blocked = _ask(target, "/v1/models", _key_for(target, guarded))
    except Blocked as unlisted:
        models = unlisted
    try:
        asked: Asking | Blocked = _ask_each_voice(target, listing, guarded)
    except Blocked as unasked:
        asked = unasked
    return Deployment(
        target=target,
        health=health,
        guarded=guarded,
        listing=listing,
        models=models,
        drawn=drawn,
        asked=asked,
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

    There is deliberately no bitrate field, though twelve of the names carry one.
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


# ------------------------------------------- what each voice says about itself


#: The two capability words `README.md:100` publishes on every voice. This
#: service's own extension vocabulary, not ElevenLabs', and written out for the
#: reason [`_CONTENT_TYPES`] is written out: importing `engine.Capability` would
#: hold this file's reading of a declaration equal to the build under test by
#: construction, and the deployment is routinely a different build
#: ([LAW:behavior-not-structure]).
CAPABILITY_SPEED = "speed"
CAPABILITY_TIMESTAMPS = "timestamps"

#: What `voice_settings.speed` is set to where a voice declares it. Two, because
#: the claim is that the audio really gets faster and a rate a hair off 1.0 would
#: make that a measurement of the encoder's padding instead.
FASTER = 2.0

#: How much shorter a doubled speed must really come back. Far below the half a
#: linear engine gives and far above any padding: `_LENGTH_SPREAD_SECONDS` worth
#: of frame alignment is under a percent of this utterance, so a tenth is a
#: threshold rather than a guess, and a deployment honouring the rate by a
#: rounding error fails it.
_FASTER_BY = 0.10

#: Language families to offer a voice that speaks none of them, tried in order.
#: `CAP-8`'s named-back arm needs a tag *no offered voice speaks*, and the reason
#: is observed rather than reasoned: `Catalog.resolve` narrows by language before
#: it matches the id, so asking an `en` voice for `es` on a deployment that also
#: offers an `es` voice **substitutes to that voice** and honours the tag. A
#: prober sending a language some other voice speaks would report `CAP-8` broken
#: against a deployment doing exactly what it documents.
#:
#: Several, so a deployment speaking one of them is not a prober that cannot ask:
#: every entry is a real ISO 639-1 family, so each is a tag this service reduces
#: to itself rather than to `None`.
UNSPOKEN_LANGUAGES = ("zu", "mi", "gd", "yo", "ha", "si", "km", "am")

#: A `model_id` shaped like one ElevenLabs publishes that no build here maps.
#: Dated for the reason [`INVENTED_FIELD`] is: `MOD-5`'s subject is the stock
#: client sending a real ElevenLabs model this deployment has never heard of.
#: Checked against the deployment's own served set before it is sent, so a
#: deployment that does declare it leaves `MOD-5` unasked rather than reporting a
#: correct 200 broken.
UNMAPPED_MODEL = "eleven_unmapped_2027"

#: A `voice_settings` leaf this service models and no engine can honour, which is
#: `CAP-6`'s subject. It is not [`INVENTED_FIELD`]: `REF-7` already asks what
#: happens to a body field nobody modelled, and `CAP-6`'s promise is the wider one
#: — *any* parameter that cannot be honoured is named, the enumerated ones
#: included. A deployment reporting only what it failed to parse keeps `REF-7` and
#: breaks this.
UNHONOURABLE_SETTING = "stability"

#: A voice id nothing can be offering, which is `SUB-1`'s and `SUB-4`'s subject —
#: one request, and the arm it lands on says whether substitution is configured.
FOREIGN_VOICE = "no-voice-named-this-2027"

#: A voice id whose bytes are not latin-1, which is `SUB-6`'s subject. A header
#: value must be latin-1 on the wire, so a server echoing this one into
#: `x-elvenspeak-voice-requested` unescaped answers 500 — observed on this build
#: before `api.py:1118` escaped it.
UNPRINTABLE_VOICE = "vøîce-ñ"


@dataclass(frozen=True)
class Declared:
    """One voice's own account of itself, in the words it published.

    [LAW:parse-dont-validate] Built only by [`_declared`], so every per-voice
    claim reads a declaration whose four fields are present and well-typed and
    none of them asks again. This is the stamp [`ProbedVoice`] deliberately does
    not carry: that type is thin because `DISC-1` is the claim these fields are
    all there, and a parser strict enough to refuse a voice missing its
    `capabilities` would crash before that verdict could be reported.

    So both types exist and neither is redundant. `ProbedVoice` is what a claim
    addressing a voice needs; `Declared` is what a claim reading a voice's
    promises needs, and the crossing between them is where `DISC-1`'s subject
    stops being a precondition and starts being a fact.
    """

    id: str
    #: What this voice says speaking in it really does, out of
    #: [`CAPABILITY_SPEED`] and [`CAPABILITY_TIMESTAMPS`]. Unknown words are kept
    #: rather than dropped: a capability this build has not heard of is a
    #: deployment ahead of this checkout, not a malformed listing.
    capabilities: frozenset[str]
    models: frozenset[str]
    language: str
    aliases: tuple[str, ...]


#: What [`_declared`] requires of a voice: the field, the JSON type, and whether
#: the elements of a list must be non-empty strings. A subset of
#: [`_PUBLISHED_FIELDS`] on purpose — `DISC-1` owns the whole shape, and this
#: table names only what a per-voice claim actually reads
#: ([FRAMING:representation] — a field with no reader is a map nothing keeps true).
_DECLARED_FIELDS: tuple[tuple[str, type], ...] = (
    ("capabilities", list),
    ("models", list),
    ("language", str),
    ("aliases", list),
)


def _declared(voice: ProbedVoice) -> Declared | str:
    """`voice`'s declaration, or the complaint saying why it published none.

    Returned rather than raised for the reason [`_parsed_voices`] returns its own:
    to `DISC-1`, whose claim *is* that these fields are present and well-typed, a
    voice missing one is the promise broken, while to a claim that needed the
    declaration to choose an input it is a precondition that failed. One parser,
    two readings ([LAW:decomposition] — parsing and judging are two jobs).
    """
    for field, expected in _DECLARED_FIELDS:
        value = voice.published.get(field)
        if not isinstance(value, expected):
            return (
                f"published {field!r} as {type(value).__name__} rather than "
                f"{expected.__name__}: {value!r:.100}"
            )
        if expected is list and not all(
            isinstance(item, str) and item for item in value
        ):
            return f"published a {field!r} holding something that is not a name: {value!r:.100}"
    language = voice.published["language"]
    if not language:
        return "published an empty 'language', so no tag can be read against it"
    return Declared(
        id=voice.id,
        capabilities=frozenset(voice.published["capabilities"]),
        models=frozenset(voice.published["models"]),
        language=language,
        aliases=tuple(voice.published["aliases"]),
    )


def _declarations(listing: Reply | Blocked) -> tuple[Declared, ...]:
    """Every offered voice's declaration, or [`Blocked`] if one cannot be read.

    All or nothing, because a per-voice claim's subject is *every* voice: asking
    it of the readable ones and reporting `held` would be the sampling this epic
    exists to remove, wearing a parse failure as the excuse.
    """
    read = []
    for voice in _voices_of(listing):
        parsed = _declared(voice)
        if isinstance(parsed, str):
            raise Blocked(
                f"voice {voice.id!r} {parsed} — so no per-voice claim can read "
                "what it promises, and DISC-1 is the claim that reports it"
            )
        read.append(parsed)
    return tuple(read)


@dataclass(frozen=True)
class Catalogue:
    """Every probe input this deployment's own account of itself decides.

    Derived once, in [`discover`], so the language `CAP-8` sends and the model id
    `MOD-4` sends are chosen from one reading of the listing rather than from as
    many readings as there are claims ([LAW:one-source-of-truth]).

    The two choosers return the [`Blocked`] rather than raising it, because each
    is a *draw* that cannot be composed and [`Asking`] already carries an
    uncomposable draw as a value — raising would make the whole voice's table
    fail for one input it could not pick ([LAW:dataflow-not-control-flow]).
    """

    voices: tuple[Declared, ...]

    @property
    def served(self) -> frozenset[str]:
        """Every `model_id` some offered voice answers to.

        The union this deployment's `GET /v1/models` is claimed to publish —
        derived here from the voices rather than read from that endpoint, because
        `MOD-2` is the claim that the two agree and a probe reading only one of
        them could not tell.
        """
        return frozenset[str]().union(*(voice.models for voice in self.voices))

    @property
    def spoken(self) -> frozenset[str]:
        """Every language family some offered voice speaks."""
        return frozenset(voice.language for voice in self.voices)

    def unspoken(self) -> str | Blocked:
        """A language family no offered voice speaks."""
        for family in UNSPOKEN_LANGUAGES:
            if family not in self.spoken:
                return family
        return Blocked(
            f"this deployment speaks every one of {list(UNSPOKEN_LANGUAGES)}, so "
            "the prober has no language left to ask for that no voice here speaks"
        )

    def elsewhere_than(self, voice: Declared) -> str | Blocked:
        """A `model_id` this deployment serves that `voice` does not answer to."""
        others = sorted(self.served - voice.models)
        if not others:
            return Blocked(
                f"every model_id this deployment publishes is answered by voice "
                f"{voice.id!r}, so none of them names an engine that is not "
                "speaking it — which is the state a single-engine deployment is "
                "always in, and the state a router fronting two engines is not"
            )
        return others[0]

    def unmapped(self) -> str | Blocked:
        """A `model_id` this deployment maps to nothing at all."""
        if UNMAPPED_MODEL in self.served:
            return Blocked(
                f"this deployment serves {UNMAPPED_MODEL!r}, which the prober "
                "sends as a model id naming no engine — it names one here"
            )
        return UNMAPPED_MODEL


# ------------------------------------------------------- the per-voice draws


WITH_TIMESTAMPS = "/with-timestamps"
STREAMED_TIMESTAMPS = "/stream/with-timestamps"

#: A second sentence, so what the streaming endpoint is asked really streams.
#: `TIME-3` promises *one object per line* and `TIME-6` that consecutive objects
#: lay end to end, and neither is falsifiable against a one-sentence run: the
#: endpoint emits one object per sentence, so [`PROBE_TEXT`] alone makes "one per
#: line" indistinguishable from "one in total" and leaves `TIME-6` with no pair
#: to compare. Built from [`PROBE_TEXT`] rather than spelled out, so the two texts
#: cannot drift in the words an engine has to pronounce
#: ([LAW:one-source-of-truth]).
STREAMED_TEXT = f"{PROBE_TEXT} Four five six."

#: The endpoint suffixes the timestamp claims read, in the order `README.md:44`
#: lists them, each with the text it is asked. Both, because `CAP-3` and `CAP-4`
#: are claims about *both* endpoints and a deployment answering one of them
#: correctly is half a verdict.
#:
#: The text is carried here rather than chosen where the draw is composed because
#: it is a fact about the endpoint — one of them streams and the other does not —
#: and a table is what keeps that from becoming a conditional at the one call site
#: that builds these draws ([LAW:dataflow-not-control-flow]).
_TIMESTAMP_TEXTS: Mapping[str, str] = {
    WITH_TIMESTAMPS: PROBE_TEXT,
    STREAMED_TIMESTAMPS: STREAMED_TEXT,
}

#: The endpoints alone, derived so the two orders cannot disagree.
_TIMESTAMP_ENDPOINTS = tuple(_TIMESTAMP_TEXTS)

#: The name of each draw, which is what a probe asks for and what a report quotes.
#: Written as words rather than as identifiers because a blocker naming one goes
#: to an operator ([LAW:one-source-of-truth] — the probe and the table agree by
#: sharing the constant, not by two spellings staying in step).
PLAIN = "asking for nothing but the text"
AGAIN = "asking for nothing but the text, a second time"
FASTER_DRAW = f"voice_settings.speed of {FASTER:g}"
OWN_FAMILY = "a language_code this voice speaks"
OWN_VARIANT = "that same language, region-tagged and upper-cased"
UNSPOKEN_DRAW = "a language_code no voice here speaks"
LISTED_MODEL = "a model_id this voice lists"
ELSEWHERE_MODEL = "a model_id another voice here lists"
UNMAPPED_DRAW = "a model_id naming no engine here"
UNHONOURABLE_DRAW = f"a voice_settings.{UNHONOURABLE_SETTING} no engine honours"
EMPTY_LANGUAGE = "an empty language_code"
BLANK_LANGUAGE = "a language_code of nothing but whitespace"
FOREIGN_DRAW = "a voice_id nothing here offers"
UNPRINTABLE_DRAW = "a voice_id whose bytes are not latin-1"


@dataclass(frozen=True)
class Draw:
    """One synthesis request: who it addresses, where it goes, what it carries.

    [LAW:dataflow-not-control-flow] Every per-voice question is a value of this
    type, so the eleven a voice is asked and the five the deployment is asked
    differ in the rows of a table and in nothing else. `voice_id` is carried
    rather than taken from the voice the table is about, because two of the
    deployment's own draws address an id no voice has — which is their whole
    point, and a field defaulting to "this table's voice" would make them a mode.
    """

    voice_id: str
    endpoint: str
    body: Mapping[str, Any]


def _saying(voice_id: str, **asked: Any) -> Draw:
    """[`PROBE_TEXT`] asked of `voice_id`'s plain endpoint, carrying `asked`."""
    return Draw(voice_id=voice_id, endpoint="", body={"text": PROBE_TEXT, **asked})


def _carrying(voice_id: str, field: str, value: str | Blocked) -> Draw | Blocked:
    """[`_saying`] with `field` set to `value`, or the [`Blocked`] that picked none.

    One adapter for both fallible inputs — the language `CAP-8` needs and the two
    model ids `MOD-4` and `MOD-5` need — so a chooser that could not pick becomes
    an uncomposable draw once rather than at each of the three call sites
    ([LAW:single-enforcer]).
    """
    if isinstance(value, Blocked):
        return value
    return _saying(voice_id, **{field: value})


def _voice_draws(voice: Declared, catalogue: Catalogue) -> dict[str, Draw | Blocked]:
    """Every request `voice` is asked, because its own declaration selects it.

    **The budget, decided here and nowhere else.** Eight of these synthesize, plus
    two where the voice declares `timestamps` and none where it does not, plus one
    refusal that costs nothing — so a voice costs 8 utterances, and a voice
    reporting timings costs 11 across 10 requests: the streaming draw is one
    request that synthesizes once per *sentence*, and [`STREAMED_TEXT`] carries
    two. That extra utterance is what `TIME-3` and `TIME-6` cost, and it is
    deliberately spent here rather than on a draw of their own — a second draw
    would cost the same two syntheses plus a request, and would still have to be
    gated on the same declaration. That multiplies by the catalogue, on top of the
    30 [`Spoken`] spends.

    Eight is the answer to "which claims are really about *this voice*", and the
    rule is: **a claim is asked of every voice when its subject is that voice's
    own declaration** — the `capabilities` list, the `models` list, the `language`
    on it. Every one of the eight reads one of those three. A claim whose subject
    is instead this deployment's *request handling* is asked once, of one voice,
    by [`_handling_draws`], because asking it again of a second voice re-asks one
    code path at the price of an utterance.

    There is deliberately no sampling shortcut, for the reason [`Spoken`] has
    none: "a voice declaring `speed` really changes the rate" is a claim about
    every voice, and behind a router the voices come from different engines in one
    process — which is the whole reason this epic exists. A prober asking one
    voice and reporting `held` would be making a claim about all of them having
    checked one, and that is the defect being removed, not a cost being saved.

    The alias rows are the one place the count is not fixed, and they are per voice
    for the same reason as the rest: an alias is something the voice itself
    published. A router publishes none (`piper-routing-7e2.15`), so `SUB-5` costs
    nothing there and the report shows an empty alias set rather than an absent
    claim.
    """
    return {
        PLAIN: _saying(voice.id),
        FASTER_DRAW: _saying(voice.id, voice_settings={"speed": FASTER}),
        OWN_FAMILY: _saying(voice.id, language_code=voice.language),
        OWN_VARIANT: _saying(voice.id, language_code=_mangled(voice.language)),
        UNSPOKEN_DRAW: _carrying(voice.id, "language_code", catalogue.unspoken()),
        LISTED_MODEL: _carrying(voice.id, "model_id", _one_model(voice)),
        ELSEWHERE_MODEL: _carrying(
            voice.id, "model_id", catalogue.elsewhere_than(voice)
        ),
        UNMAPPED_DRAW: _carrying(voice.id, "model_id", catalogue.unmapped()),
        **{
            _timestamp_draw(endpoint): Draw(
                voice_id=voice.id, endpoint=endpoint, body={"text": asked}
            )
            for endpoint, asked in _TIMESTAMP_TEXTS.items()
        },
        **{_alias_draw(alias): _saying(alias) for alias in voice.aliases},
        # Last, so the two draws [`Asked.repeats`] compares are genuinely two asks
        # separated by every other request this table makes — the ordering
        # [`Spoken.again`] keeps, and for the same reason.
        AGAIN: _saying(voice.id),
    }


def _timestamp_draw(endpoint: str) -> str:
    """The name of the draw asking `endpoint`."""
    return f"POST {endpoint}"


def _alias_draw(alias: str) -> str:
    """The name of the draw asking for `alias`."""
    return f"the alias {alias!r}"


def _mangled(family: str) -> str:
    """`family` spelled the three ways `CAP-10` says are one request.

    One draw rather than three, and that is a strengthening rather than a saving:
    upper-casing, an underscore separator and a region subtag are the three
    reductions `engine.spoken_language` performs, so a tag carrying all three at
    once is honoured only by a deployment doing all three. Three separate draws
    would each leave the other two unasked and cost two more utterances per voice.

    The region is numeric so it cannot be read as a language of its own: `es-419`
    is a real tag of this shape, where an `EN_GB` would leave a reader wondering
    whether `gb` was the family that got compared.
    """
    return f"{family.upper()}_419"


def _one_model(voice: Declared) -> str | Blocked:
    """A `model_id` `voice` itself lists, or [`Blocked`] if it lists none."""
    if not voice.models:
        return Blocked(
            f"voice {voice.id!r} lists no model_id at all, so there is none of "
            "its own to send — it is unreachable by engine, which DISC-1 reports"
        )
    return sorted(voice.models)[0]


def _handling_draws(voice: Declared, catalogue: Catalogue) -> dict[str, Draw | Blocked]:
    """Every request about this deployment's handling rather than about a voice.

    Asked once, of `voice`, because each is a claim about one code path: what a
    blank `language_code` means, what happens to a parameter nothing can honour,
    what an unknown voice id earns. Asking them again of a second voice would
    re-ask the same path at the price of an utterance, which is the half of the
    budget [`_voice_draws`] does *not* spend per voice.

    These rows join the first voice's own table rather than making a second
    [`Asked`] beside it, and that is what keeps every draw name readable from one
    place. Two tables for one voice meant a claim had to know which of them held
    the name it wanted, and nothing in the type said: `CAP-7` and `CAP-9` both
    read [`PLAIN`] off the deployment-wide table, which never drew it, and every
    run raised `KeyError` before a report could be printed
    ([LAW:types-are-the-program] — the lookup that cannot fail is the one where
    there is only one table to look in).

    Five requests, four of which synthesize — [`FOREIGN_DRAW`] synthesizes only
    where substitution is configured, and answers 404 where it is not, which is
    `SUB-4`.
    """
    return {
        UNHONOURABLE_DRAW: _saying(
            voice.id, voice_settings={UNHONOURABLE_SETTING: 0.4}
        ),
        EMPTY_LANGUAGE: _saying(voice.id, language_code=""),
        BLANK_LANGUAGE: _saying(voice.id, language_code="   "),
        FOREIGN_DRAW: _saying(FOREIGN_VOICE),
        UNPRINTABLE_DRAW: _saying(UNPRINTABLE_VOICE),
    }


@dataclass(frozen=True)
class Asked:
    """One voice, and what it answered to every draw made of it.

    `answers` holds a [`Blocked`] where a draw could not be composed or never
    finished, rather than dropping the row, for the reason [`Spoken.answers`] does:
    a claim whose draw is missing must report that it could not be asked, and a
    row quietly absent is a claim nobody notices went unasked
    ([LAW:no-silent-failure]).

    [LAW:parse-dont-validate] Built only by [`_asked_of`], which derives both maps
    from one table in one pass, so they carry the same keys by construction rather
    than by two comprehensions staying in step. `draws` is kept beside the answers
    because `SUB-2` is a claim about the relationship between the two — that
    `x-elvenspeak-voice-requested` appears exactly when the voice that spoke is not
    the id that was addressed — and an answer alone cannot say what was asked for.
    """

    voice: Declared
    draws: Mapping[str, Draw | Blocked]
    answers: Mapping[str, Reply | Blocked]

    def addressed(self, draw: str) -> str:
        """The voice id `draw` was sent to."""
        composed = self.draws[draw]
        if isinstance(composed, Blocked):
            raise composed
        return composed.voice_id

    def answered(self, draw: str) -> Reply:
        """`draw`'s answer, or [`Blocked`] saying why there is none.

        [LAW:parse-dont-validate] The one crossing between a draw and the claims
        that read it, so none of them asks again whether an answer arrived. A
        `KeyError` here is a probe naming a draw this table does not make, which
        is a bug in the prober and deliberately not caught by [`_finding`].
        """
        answer = self.answers[draw]
        if isinstance(answer, Blocked):
            raise answer
        return answer

    def spoke(self, draw: str) -> Reply:
        """`draw`'s answer, proven to carry audio.

        The reading every claim about a *successful* synthesis needs, so a refusal
        leaves it unasked behind one blocker rather than being read as a 200 whose
        headers say nothing ([LAW:single-enforcer] — one account of what counts as
        having spoken, wherever the question comes from).

        What this cannot separate is *why* there is no audio, and for a claim that
        promised the draw would be served the difference decides the verdict: a
        draw that could not be **composed** is a precondition failing and `unasked`
        is right, while one composed, sent and **refused** is the deployment
        answering. [`refusal`] is the reading for the second, and every claim that
        requires a served draw asks it first.
        """
        return _served(self.answers[draw], f"{self.voice.id!r} {draw}")

    def refusal(self, draw: str) -> str | None:
        """How this deployment turned `draw` down, or `None` if it served it.

        The half of [`spoke`]'s blocker that is a verdict rather than a
        precondition. A claim promising that a draw is served — `CAP-1`, `CAP-2`,
        `CAP-3`, `CAP-6`, `CAP-8`, `CAP-9`, `CAP-10`, `MOD-3`, `MOD-6`, `SUB-5`
        and `TIME-2` through `TIME-6` — is *false* when the deployment refuses
        it, and reading that
        through [`spoke`] alone reported every one of them as "could not be
        asked". A draw carrying no such promise keeps [`spoke`]'s reading: a
        refused `PLAIN` is `FMT-1`'s subject, not `CAP-7`'s or `CAP-9`'s. [`answered`] still raises for a
        draw that never got an answer at all, so the two readings partition what
        can become of a draw at exactly the line where the deployment's own answer
        begins ([LAW:single-enforcer] — what a refusal means to a claim that needed
        the audio is decided here, not in each of the faults that read it).
        """
        answer = self.answered(draw)
        if answer.status == 200:
            return None
        return f"answered {draw} with {answer.status}: {answer.body[:200]!r}"

    def repeats(self) -> bool:
        """Whether this voice answered two identical requests identically.

        [`Spoken.repeats`] asks this of the one voice the format draws were made
        of; this asks it of each voice in turn, because behind a router they are
        different engines and one of them sampling says nothing about the next.
        `CAP-1` is the claim that needs it: a length read against another length
        is evidence only where the backend is a function of the request.
        """
        return self.spoke(AGAIN).body == self.spoke(PLAIN).body

    def declares(self, capability: str) -> bool:
        """Whether this voice's own list names `capability`."""
        return capability in self.voice.capabilities


@dataclass(frozen=True)
class Asking:
    """Every voice asked what its own declaration selects, drawn once.

    Drawn in [`discover`] rather than inside the claims that read it, for the
    reason [`Spoken`] is: `CAP-1` reads one of a voice's lengths against another,
    and a prober re-synthesizing per claim would be comparing two different
    utterances against a backend that samples ([LAW:one-source-of-truth]).

    **What a run costs, stated where it is spent.** [`_voice_draws`] spends 8
    utterances per voice and 11 for a voice reporting timings;
    [`_handling_draws`] spends 4 more, once. So a catalogue of `V` voices costs
    `8V + 4` at least, against `Spoken`'s flat 30 — the per-voice half multiplies
    and the deployment-wide half does not, which is the whole of the decision
    [`_voice_draws`] records. [`syntheses`] reports what it really came to, so an
    operator reads the price off the run rather than off this docstring.
    """

    catalogue: Catalogue
    #: One entry per offered voice, in the order the listing published them. The
    #: first also carries the deployment-wide draws [`_handling_draws`] makes,
    #: which is what [`once`] names.
    voices: tuple[Asked, ...]

    @property
    def once(self) -> Asked:
        """The table carrying the draws asked of this deployment rather than of a voice.

        The first offered voice's own table, because that is where
        [`_handling_draws`] puts them — a derived name for one of `voices` rather
        than a second [`Asked`] holding a copy of them ([LAW:one-source-of-truth]).
        A claim reading `once` and a claim reading `voices[0]` are therefore
        reading one set of answers, and every draw name this run makes is readable
        from either.
        """
        return self.voices[0]

    def syntheses(self) -> int:
        """How many of these requests really made audio.

        Derived rather than counted as the draws are made, so there is no second
        number free to disagree with the answers it describes
        ([LAW:one-source-of-truth]). A 200 is the test because every draw here is
        a synthesis endpoint: a refusal and a 501 spent no engine time, which is
        exactly the difference an operator pricing a run cares about.
        """
        return sum(
            1
            for asked in self.voices
            for answer in asked.answers.values()
            if isinstance(answer, Reply) and answer.status == 200
        )

    def declaring(self, capability: str) -> tuple[Asked, ...]:
        """The voices whose own list names `capability`."""
        return tuple(asked for asked in self.voices if asked.declares(capability))

    def omitting(self, capability: str) -> tuple[Asked, ...]:
        """The voices whose own list omits `capability`.

        The other arm of [`declaring`], and a method rather than the caller's
        negation so the two arms are one partition of the catalogue. A voice is in
        exactly one of them, which is what makes a claim per arm add up to a claim
        about every voice ([LAW:dataflow-not-control-flow] — the declaration is
        the discriminator, read off the deployment rather than guessed from which
        engine the prober thinks it is talking to).
        """
        return tuple(asked for asked in self.voices if not asked.declares(capability))


def _ask_each_voice(
    target: Target, listing: Reply | Blocked, guarded: bool
) -> Asking:
    """Every offered voice asked what its declaration selects, plus the five.

    Each draw is caught on its own, so one request that never answered leaves
    every other readable and the claim that owns it reports the failure. A draw
    that failed is a value here, never a row quietly missing
    ([LAW:no-silent-failure]).
    """
    voices = _declarations(listing)
    if not voices:
        raise Blocked(
            "this deployment lists no voice, so no claim about what a voice "
            "promises has a subject"
        )
    key = _key_for(target, guarded)
    catalogue = Catalogue(voices=voices)
    first, rest = voices[0], voices[1:]
    return Asking(
        catalogue=catalogue,
        # The deployment-wide draws are made first so [`_voice_draws`]' own
        # ordering survives the merge — [`AGAIN`] stays the last request of this
        # table, which is what makes it a second ask rather than a repeat of the
        # one before it.
        voices=(
            _asked_of(
                target,
                first,
                key,
                {
                    **_handling_draws(first, catalogue),
                    **_voice_draws(first, catalogue),
                },
            ),
            *(
                _asked_of(target, voice, key, _voice_draws(voice, catalogue))
                for voice in rest
            ),
        ),
    )


def _asked_of(
    target: Target,
    voice: Declared,
    key: str | None,
    draws: Mapping[str, Draw | Blocked],
) -> Asked:
    """`draws` all made, each failure kept as the value that stops its own claim."""

    def answered(draw: Draw | Blocked) -> Reply | Blocked:
        if isinstance(draw, Blocked):
            return draw
        try:
            return _posted(
                target, draw.voice_id, key, PCM_FORMAT, draw.body, draw.endpoint
            )
        except Blocked as unanswered:
            return unanswered

    return Asked(
        voice=voice,
        draws=draws,
        answers={name: answered(draw) for name, draw in draws.items()},
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
        ("POST", _in_format(path, PCM_FORMAT), {"text": PROBE_TEXT})
        for path in _addressed(DOCUMENTED_SYNTHESES, voice_id)
    )
    return reads + syntheses


def _answered_keylessly(reply: Reply) -> bool:
    """Whether `reply` is an answer a keyless caller really got.

    Two non-2xx statuses are answers rather than refusals, and both belong to
    another claim. A 404 is a documented path this build does not route. A 501
    carrying [`CANNOT_REPORT_TIMINGS`] is a voice that cannot report timings —
    `CAP-4` owns that refusal and *requires* exactly it, so judging it here means
    this file demands a 2xx from the same request another of its claims demands a
    501 for, and a deployment whose voices honour nothing beyond plain speech is
    reported broken for being conformant ([LAW:single-enforcer] — whether a status
    is the documented capability refusal is one question, answered by the sentence
    `CAP-4` already reads rather than by a second status table drifting beside it).

    The sentence is what selects the arm, not the bare 501: a deployment guarding
    its endpoints behind an undocumented 501 turned a keyless caller away, and
    that is this claim failing.
    """
    return (
        200 <= reply.status < 300
        or reply.status == 404
        or (reply.status == 501 and _detail_of(reply) == CANNOT_REPORT_TIMINGS)
    )


def probe_auth_2(deployment: Deployment) -> Verdict:
    """With nothing configured, every documented endpoint answers without a key.

    Asked of [`DOCUMENTED_READS`] and [`DOCUMENTED_SYNTHESES`], because the claim
    is about every endpoint and a guard applied to only the expensive half — or
    to only the streaming part of that half — would pass a narrower check
    completely. One voice, and an utterance per documented synthesis, which is
    what that costs.

    What counts as an answer is [`_answered_keylessly`]. Every other non-2xx is
    this claim failing, named by its status: judging only a 401 would be assuming
    a foreign deployment shares this checkout's choice of status for a refusal —
    a proxy in front of it refusing 403, or the deployment answering 500, is an
    endpoint a keyless caller did not get an answer from either way.
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
        if not _answered_keylessly(reply):
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


def _supported_arrays(value: Any) -> list[list[str]]:
    """Every `supported` array of format names anywhere in `value`, at any depth.

    Searched structurally rather than read at a path, which is what accepting both
    422 shapes without demanding either has to mean in code: this service's own
    refusal carries the array under `detail.message`'s sibling, pydantic's carries
    whatever a deployment put in its error records, and `FMT-8` is entitled to the
    array without being entitled to the shape around it.

    [LAW:parse-dont-validate] An array is one of these only if every element is a
    name, so what comes back is `list[str]` and [`probe_fmt_8`] may put it in a
    set without asking again. An array holding objects is not a `supported` array
    with a flaw in it — it is a body that carries none, which is a verdict that
    claim already writes.

    The three arms are JSON's own variants handled exhaustively, which is the one
    thing [LAW:dataflow-not-control-flow] asks a branch to be.
    """
    if isinstance(value, dict):
        offered = value.get("supported")
        here = [offered] if _is_name_array(offered) else []
        return here + [
            found for nested in value.values() for found in _supported_arrays(nested)
        ]
    if isinstance(value, list):
        return [found for nested in value for found in _supported_arrays(nested)]
    return []


def _is_name_array(value: Any) -> bool:
    """`value` is an array of strings, the only thing `supported` may name."""
    return isinstance(value, list) and all(isinstance(name, str) for name in value)


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
        arrays = _supported_arrays(json.loads(answer.body))
    except ValueError:
        return Broken(
            f"the 422 refusing {UNKNOWN_FORMAT} answered a body that is not JSON: "
            f"{answer.body[:200]!r}"
        )
    except RecursionError:
        return Broken(
            f"the 422 refusing {UNKNOWN_FORMAT} answered a body nested too deeply "
            f"to read: {answer.body[:200]!r} — this prober will not walk it, and a "
            "caller's JSON parser has the same limit"
        )
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


def _names(body: bytes, name: str) -> bool:
    """`body` names `name` as a word, rather than spelling it inside a longer one.

    A raw byte substring is not the rule [`Refusal`] states: `text` is contained in
    `context` and `supported` in `unsupported`, so a refusal reading `request
    exceeds the maximum context window` — or `unsupported format`, carrying no list
    — satisfies `REF-2` and `REF-1` without naming the thing it refused. That is
    the check-that-cannot-fail this table was written to replace, so matching on a
    word boundary is what "name" meant all along.

    JSON puts quotes, commas and braces around both field names and values, and
    none of those are word characters, so every legitimate naming still matches.
    """
    return re.search(rb"\b" + re.escape(name.encode()) + rb"\b", body) is not None


def _refused(deployment: Deployment, refusal: Refusal) -> Reply:
    """`refusal`'s request really refused, or [`Blocked`] if it was answered.

    The reading `FMT-8` needs, which is about what a refusal *carries* and so has
    no subject until there is one. [`probe_refusal`] asks its own copy: a refusal
    is one cheap idempotent POST, unlike the draw [`Spoken`] is cached to buy once.
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
    missing = [name for name in refusal.named if not _names(answer.body, name)]
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


# --------------------------------------------------- per-voice claim machinery


#: What a voice that does not report timings answers both timestamp endpoints,
#: verbatim. Quoted rather than matched loosely because `CAP-4`'s promise is that a
#: caller can *tell* this refusal from every other 501 — README:106 publishes the
#: sentence, and a bare 501 leaves a client guessing whether the route exists.
CANNOT_REPORT_TIMINGS = (
    "this service cannot report how long each part of an utterance took"
)


#: What [`_of_every`] says when a claim asked of the *whole* catalogue has no
#: subject — which no run reaches: [`_ask_each_voice`] raises [`Blocked`] on an
#: empty listing before an [`Asking`] exists to pass here, so only the filtered
#: callers ([`Asking.declaring`] and [`Asking.omitting`], which are `CAP-1`/`CAP-2`
#: and `CAP-3`/`CAP-4` selecting their arm) can really be empty and they say which
#: arm the deployment chose instead. Named once rather than spelled six times, so
#: six sentences about one unreachable state cannot drift into six accounts of it
#: ([LAW:one-source-of-truth]).
NO_VOICE_OFFERED = (
    "this deployment lists no voice, so a claim asked of every voice has no subject"
)


#: What `CAP-3` and the five timestamp claims all say when the catalogue selects
#: the other arm. One sentence rather than six, so six accounts of one state
#: cannot drift apart ([LAW:one-source-of-truth]).
NO_TIMESTAMP_VOICE = (
    f"no voice here declares {CAPABILITY_TIMESTAMPS!r}, so there is none owed "
    "an alignment — CAP-4 is the arm this deployment's voices select"
)


def _of_every(
    subjects: tuple[Asked, ...],
    unasked: str,
    fault: Callable[[Asked], str | None],
    held: str,
) -> Verdict:
    """`fault` asked of every subject: the first one found, or `held` over all.

    [LAW:one-type-per-behavior] Every per-voice claim is this same loop — an
    empty set of subjects is `unasked`, the first fault is `broken`, and a clean
    sweep is `held` over a count — and what separates them is only which fault a
    voice can have. That is a value, so they are instances of one claim shape
    rather than copies of a loop free to drift in how a claim over an empty
    catalogue reads ([LAW:single-enforcer] — what "no subject" means is decided
    here).

    Counted in neither direction on purpose: a tally of the callers here would be
    a second record of something the call sites already hold, kept true by memory
    alone. It read "eleven" while there were fifteen.

    A [`Blocked`] out of `fault` is left to propagate, and that is deliberate: the
    claim is about *every* subject, so a draw that never answered for one of them
    is a claim that could not be asked in full. Catching it to carry on would
    report `held` over the voices that happened to answer, which is the sampling
    this epic exists to remove.
    """
    if not subjects:
        return Unasked(unasked)
    for subject in subjects:
        found = fault(subject)
        if found is not None:
            return Broken(f"voice {subject.voice.id!r} {found}")
    return Held(f"{_counted(len(subjects), 'voice')} {held}")


def _ignored_header(reply: Reply) -> str | None:
    """`x-elvenspeak-ignored` as it arrived, or `None` if it did not.

    `None` and `""` are different answers and `CAP-7` is the claim about the
    difference: absent means everything asked for was honoured, while an empty
    header is a server saying "these were dropped" and naming none of them.
    """
    return reply.headers.get(f"{_ELVENSPEAK_HEADER}ignored")


def _ignored(reply: Reply) -> tuple[str, ...]:
    """Every parameter `reply` names as not honoured."""
    named = _ignored_header(reply)
    return () if named is None else tuple(part.strip() for part in named.split(","))


def _spoke_in(reply: Reply) -> str | None:
    """The voice `reply` says spoke, or `None` if it named none."""
    return reply.headers.get(f"{_ELVENSPEAK_HEADER}voice")


def _asked_for(reply: Reply) -> str | None:
    """The voice `reply` says was requested, or `None` if it named none."""
    return reply.headers.get(f"{_ELVENSPEAK_HEADER}voice-requested")


def _echoed(requested: str | None, addressed: str) -> bool:
    """Whether `requested` is this deployment's record of `addressed`.

    Equality wherever a header value can hold the id as it was sent, and presence
    alone where it cannot. An id outside the printable ASCII range reaches the
    caller *escaped* — a caller who sends a bare CRLF in a voice id must not get
    it back unescaped — and which escape a build picks is that build's own
    business. Holding this file to one spelling would report `broken` against a
    deployment doing the only safe thing: a real server answered
    [`UNPRINTABLE_VOICE`] with `'v\\xf8\\xeece-\\xf1'`, and `SUB-2` read that as a
    substitution misrecording what was asked for.

    Backslash and comma count as unholdable for the same reason rather than a
    different one: a backslash is what introduces an escape and a comma is what
    separates the parts of these headers, so an id carrying either cannot be told
    apart from the rendering. That is a fact about the wire, so the two arms are
    selected by the id rather than by which draw is asking
    ([LAW:dataflow-not-control-flow]) — and it is deliberately not this build's
    escape table, which the prober must never assert against a foreign deployment
    ([LAW:behavior-not-structure]). [`probe_sub_6`] reads its own id the same way
    and says why at greater length.
    """
    if all(" " <= char <= "~" and char not in "\\," for char in addressed):
        return requested == addressed
    return requested is not None


def _detail_of(reply: Reply) -> Any:
    """`reply`'s `detail`, or the raw bytes where it carries none.

    The raw bytes rather than `None`, so a complaint quoting this shows what really
    came back: a body that is not JSON and one whose `detail` is genuinely absent
    are both answers a claim reports, and neither should print as nothing.
    """
    try:
        body = json.loads(reply.body)
    except ValueError:
        return reply.body
    return body.get("detail") if isinstance(body, dict) else body


#: The two per-character timelines every alignment publishes, in the order a
#: reader lays them side by side.
_TIME_ARRAYS = ("character_start_times_seconds", "character_end_times_seconds")


@dataclass(frozen=True)
class Timed:
    """One object off a timestamp endpoint, parsed into what a claim reads.

    [LAW:parse-dont-validate] Built only by [`_timed`], so holding one is proof
    that this object carried an alignment whose three arrays are non-empty and
    the same length, whose every time is finite, and audio that really decoded.
    That is why the five claims about these endpoints contain no shape guards at
    all: the question of whether the body had what they read is answered once, at
    the crossing, and cannot be asked again inland.

    [`fidelity`] is deliberately *not* in that list. It is carried exactly as it
    arrived, unnarrowed and unvalidated — `None` when the body published none —
    because `TIME-3`'s whole reading is whether the value is one of the two
    README:187 publishes, and a parse that admitted only those two could never
    report a third ([LAW:no-silent-failure]). Naming it among the proven
    invariants would promise a guard that does not exist.

    Finite is part of the stamp rather than a detail of [`_number`] because it is
    the part these claims lean on hardest and the only part that fails silently:
    a `NaN` is shaped like a time and passes every other reading here, and then
    answers `False` to every `<` and every `>` inland, so each claim that trusts
    this stamp reports a scrambled timeline held.
    """

    #: `alignment` and `normalized_alignment` exactly as they arrived. `TIME-5`'s
    #: whole claim is that the two are one object, and a comparison of two
    #: re-parsed projections could hold while the published objects differed.
    alignment: Mapping[str, Any]
    normalized: Any
    #: `alignment_fidelity` as it arrived, unnarrowed: `TIME-3` judges the value
    #: against the two words README:187 publishes, so a parse that admitted only
    #: those two could never report a third ([LAW:no-silent-failure]).
    fidelity: Any
    #: The two timelines only. `characters` is *checked* at the crossing — the
    #: arrays must be non-empty and lie against it one for one — and then not
    #: carried, because no claim reads it back. Validating a thing and keeping a
    #: thing are separate decisions, and keeping one nothing reads is weight on
    #: every parsed object ([LAW:polishing-by-subtraction]).
    starts: tuple[float, ...]
    ends: tuple[float, ...]
    #: This object's own audio, decoded. What makes `TIME-4` arithmetic within one
    #: answer rather than a comparison across two draws: these are the very
    #: samples this alignment describes, so a deployment that samples — which
    #: `piper-conformance-e16.ysu` records the live piper one doing — cannot make
    #: the claim unaskable the way it does the length-comparison claims.
    audio: bytes

    @property
    def timelines(self) -> tuple[tuple[str, tuple[float, ...]], ...]:
        """Both per-character timelines, each beside the field it arrived in.

        Named pairs rather than two attributes read separately, because every
        claim about how a timeline runs is true of both and a probe reaching for
        one of them by name is a probe that checks half the answer
        ([LAW:dataflow-not-control-flow] — which array is being read is a value
        here, so `TIME-4` orders both by iterating rather than by carrying a
        second copy of the loop). `starts` was the half nothing checked: a
        deployment that scrambles it keeps `ends` ascending and [`span`] honest.
        """
        return tuple(zip(_TIME_ARRAYS, (self.starts, self.ends)))

    @property
    def span(self) -> float:
        """How long this object's own timeline says its own audio runs.

        Read as a span rather than as `ends[-1]` because a streamed object after
        the first starts partway through the run: `api.py:979` lays each sentence
        where the last ended, so the second object's end time is absolute across
        the whole response while its audio is only its own sentence. The
        difference is the one reading true of both endpoints.
        """
        return self.ends[-1] - self.starts[0]


#: The widest magnitude a time may carry, and so the widest [`_one_timed`] hands
#: inland: past it a JSON integer no longer converts to the `float` every claim
#: does its arithmetic in.
_FINITE_SECONDS = sys.float_info.max


def _number(value: Any) -> bool:
    """Whether `value` is a time a claim can compare, not merely one shaped like it.

    `bool` is excluded because Python makes it an `int`, so a timeline carrying
    `true` would otherwise parse as a time of 1 second. The range comparison
    excludes the rest, and it is the half [`Timed`]'s stamp depends on:
    `json.loads` reads the non-standard `NaN`, `Infinity` and `-Infinity` that
    `json.dumps` itself writes for those floats, and every comparison against a
    `NaN` is False — so one admitted here would answer "no" to `TIME-4`'s
    reversal test, to its length test and to `TIME-6`'s gap test alike, and a
    timeline of garbage would be reported held by all three
    ([LAW:single-enforcer] — the probes read comparisons, so the question of
    whether a value can answer one is settled here or nowhere).

    Written as a range rather than `math.isfinite` because `json` parses an
    integer of any width at all, and both that call and the `float` below
    overflow on one too large to be a `float` — killing the probe mid-run
    instead of reporting the deployment broken. Agrees with the serving side,
    which already refuses all of these — see
    `test_a_timestamp_that_is_not_a_usable_number_fails_the_request`.
    """
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and -_FINITE_SECONDS <= value <= _FINITE_SECONDS
    )


def _timed(body: bytes) -> tuple[Timed, ...] | str:
    """Every timestamped object `body` carries, or why it carries none.

    One reader for both timestamp endpoints, because `api.py:1299` gives them one
    body shape and the non-streaming answer is the one-line case of the streaming
    one. Parsed here once rather than by a reader per claim, so six claims cannot
    hold six accounts of what a malformed alignment is
    ([LAW:one-source-of-truth]).

    A returned tuple is never empty, and callers read it that way:
    `probe_time_4` takes `objects[0]` to see where the timeline begins. The
    empty-body arm below is what makes that true. No deployment reaches it —
    [`Asked.spoke`] refuses a 200 carrying no body before any claim gets here,
    which `test_an_empty_timestamp_body_never_reaches_the_timestamp_parser`
    pins — so it is the guarantee rather than a path, and deleting it as
    unreachable would hand `objects[0]` an empty tuple.
    """
    lines = body.splitlines()
    if not lines:
        return "answered an empty body, carrying no alignment at all"
    parsed = []
    for line in lines:
        one = _one_timed(line)
        if isinstance(one, str):
            return one
        parsed.append(one)
    return tuple(parsed)


def _one_timed(line: bytes) -> Timed | str:
    """One object of a timestamp endpoint's body, or why it could not be read."""
    try:
        parsed = json.loads(line)
    except ValueError:
        return f"answered a body that is not JSON: {line[:200]!r}"
    if not isinstance(parsed, dict):
        return f"answered {parsed!r:.200} rather than an object carrying an alignment"
    alignment = parsed.get("alignment")
    if not isinstance(alignment, dict):
        return f"answered no `alignment` object: {parsed!r:.200}"
    characters = alignment.get("characters")
    if not isinstance(characters, list):
        return f"answered an `alignment` with no `characters` array: {alignment!r:.200}"
    if not characters:
        return (
            "answered an empty `characters` array — a caller is handed an "
            "alignment covering none of the utterance"
        )
    times = []
    for field in _TIME_ARRAYS:
        seconds = alignment.get(field)
        if not isinstance(seconds, list) or not all(map(_number, seconds)):
            return (
                f"answered an `alignment` whose `{field}` is not an array of "
                f"numbers: {alignment!r:.200}"
            )
        if len(seconds) != len(characters):
            return (
                f"answered {_counted(len(characters), 'character')} and "
                f"{len(seconds)} `{field}` — a timeline that cannot be laid "
                "against the text it is supposed to time"
            )
        times.append(tuple(float(second) for second in seconds))
    encoded = parsed.get("audio_base64")
    if not isinstance(encoded, str):
        return f"answered no `audio_base64` string: {parsed!r:.200}"
    try:
        audio = base64.b64decode(encoded, validate=True)
    except ValueError:
        return f"answered an `audio_base64` that is not base64: {encoded[:200]!r}"
    starts, ends = times
    return Timed(
        alignment=alignment,
        normalized=parsed.get("normalized_alignment"),
        fidelity=parsed.get("alignment_fidelity"),
        starts=starts,
        ends=ends,
        audio=audio,
    )


# -------------------------------------------------- per-voice capability probes


def probe_cap_1(deployment: Deployment) -> Verdict:
    """A voice declaring `speed` really speaks faster when asked to.

    The falsifiable half of the capability contract, and the reason the whole
    per-voice pass is worth its price: a deployment can publish `speed` on every
    voice, report it honoured in every header, and hand back the same audio every
    time. Only the byte count gives it away, and only in `pcm_*`, where a byte
    count is a sample count.

    The header is judged beside it because the two can disagree in one direction
    that matters behind a router: a backend with the capability withheld reports
    `voice_settings.speed` ignored while the listing that reached the caller still
    declares it, and a caller reading the listing to decide has been told the
    opposite of what happened.

    Read against this voice's own neutral draw rather than against a duration this
    file holds, because how long an engine takes to say three words is the
    engine's business. That comparison is evidence only where the backend is a
    function of the request, which [`Asked.repeats`] establishes per voice.
    """

    def fault(asked: Asked) -> str | None:
        # Before the lengths, because a refused rate has no length to read and is
        # this claim being false rather than a reason it could not be asked.
        refused = asked.refusal(FASTER_DRAW)
        if refused is not None:
            return (
                f"declares {CAPABILITY_SPEED!r} and then {refused} — a rate this "
                "voice promises to speak at is one it must accept when asked"
            )
        if not asked.repeats():
            raise Blocked(
                f"voice {asked.voice.id!r} answered two identical {PCM_FORMAT} "
                "requests with two different utterances, so a length at one speed "
                "cannot be read against a length at another"
            )
        named = _ignored(asked.spoke(FASTER_DRAW))
        if "voice_settings.speed" in named:
            return (
                f"declares {CAPABILITY_SPEED!r} and then reported "
                f"voice_settings.speed ignored — a caller reading the listing to "
                "decide whether to send a rate was told the opposite of what this "
                "deployment did"
            )
        neutral = len(asked.spoke(PLAIN).body)
        faster = len(asked.spoke(FASTER_DRAW).body)
        if faster > neutral * (1 - _FASTER_BY):
            return (
                f"declares {CAPABILITY_SPEED!r} but answered {faster} bytes at "
                f"speed {FASTER:g} against {neutral} at the neutral rate — "
                f"{1 - _FASTER_BY:.0%} of the audio or more, so the rate reached "
                "no engine and the declaration is a promise nothing keeps"
            )
        return None

    return _of_every(
        deployment.asking().declaring(CAPABILITY_SPEED),
        f"no voice here declares {CAPABILITY_SPEED!r}, so there is none whose rate "
        "could be varied — CAP-2 is the arm this deployment's voices select",
        fault,
        f"declaring {CAPABILITY_SPEED!r} each answered materially shorter audio at "
        f"speed {FASTER:g}",
    )


def probe_cap_2(deployment: Deployment) -> Verdict:
    """A voice not declaring `speed` says so rather than dropping the rate.

    `CAP-1`'s other arm, selected by the voice's own list. The failure this
    catches is the quiet one: a server that accepts the rate, ignores it, and
    answers 200 with no header has told the caller their whole request was
    honoured when part of it was discarded — README:24's rule 2 exactly inverted.
    """

    def fault(asked: Asked) -> str | None:
        refused = asked.refusal(FASTER_DRAW)
        if refused is not None:
            return (
                f"omits {CAPABILITY_SPEED!r} and then {refused} — a rate it does "
                "not honour is one to serve and report ignored, never one to "
                "refuse over"
            )
        named = _ignored(asked.spoke(FASTER_DRAW))
        if "voice_settings.speed" not in named:
            return (
                f"omits {CAPABILITY_SPEED!r} and was asked for speed "
                f"{FASTER:g} anyway, answering 200 with "
                f"{_ELVENSPEAK_HEADER}ignored {named or 'absent'} — the rate was "
                "dropped without a word, so the caller believes it was honoured"
            )
        return None

    return _of_every(
        deployment.asking().omitting(CAPABILITY_SPEED),
        f"every voice here declares {CAPABILITY_SPEED!r}, so there is none that "
        "must report the rate ignored — CAP-1 is the arm they select",
        fault,
        "omitting it each named voice_settings.speed as ignored",
    )


def probe_cap_3(deployment: Deployment) -> Verdict:
    """A voice declaring `timestamps` answers both endpoints with real timings.

    Both, because a deployment answering one of them is half a verdict and the
    streaming half is the one a caller reaches for under load. A non-empty
    `characters` array is what separates timings from an alignment-shaped hole:
    the endpoint can answer 200 with every array empty and satisfy every check
    that reads a status.

    Read through [`_timed`], which asks more of the body than this claim once
    did: a malformed timing array or an `audio_base64` that is not base64 now
    breaks `CAP-3`, where a body carrying real `characters` beside them used to
    hold. That widening is this claim's own argument carried the rest of the way
    — garbage times are an alignment-shaped hole by exactly the reasoning that
    rejects empty ones, and a `held` over them is the check that cannot fail this
    epic exists to delete. It is a contract rather than a side effect of sharing
    the parser: `tests/test_prober.py` pins it, so narrowing [`_timed`] back
    fails there rather than quietly restoring the tolerance.
    """

    def fault(asked: Asked) -> str | None:
        for endpoint in _TIMESTAMP_ENDPOINTS:
            refused = _refused_own_endpoint(asked, endpoint)
            if refused is not None:
                return f"{refused}, which is CAP-4's answer and not this arm's"
            answer = asked.spoke(_timestamp_draw(endpoint))
            objects = _timed(answer.body)
            if isinstance(objects, str):
                return f"declares {CAPABILITY_TIMESTAMPS!r} and {endpoint} {objects}"
        return None

    return _of_every(
        deployment.asking().declaring(CAPABILITY_TIMESTAMPS),
        NO_TIMESTAMP_VOICE,
        fault,
        f"declaring {CAPABILITY_TIMESTAMPS!r} each answered both timestamp "
        "endpoints with a non-empty alignment",
    )


def probe_cap_4(deployment: Deployment) -> Verdict:
    """A voice not declaring `timestamps` refuses both endpoints, saying why.

    501 and the published sentence, not merely a non-200: a caller that gets a
    bare 501 cannot tell a route this build does not serve from a voice that
    cannot report timings, and the second is a reason to try another voice.
    """

    def fault(asked: Asked) -> str | None:
        for endpoint in _TIMESTAMP_ENDPOINTS:
            answer = asked.answered(_timestamp_draw(endpoint))
            if answer.status != 501:
                return (
                    f"omits {CAPABILITY_TIMESTAMPS!r} and {endpoint} answered "
                    f"{answer.status}: {answer.body[:200]!r} — a voice that cannot "
                    "report timings must refuse rather than invent them"
                )
            if _detail_of(answer) != CANNOT_REPORT_TIMINGS:
                return (
                    f"omits {CAPABILITY_TIMESTAMPS!r} and {endpoint} refused 501 "
                    f"with {answer.body[:200]!r} rather than the published "
                    f"{CANNOT_REPORT_TIMINGS!r} — a caller cannot tell this from a "
                    "route this build does not serve"
                )
        return None

    return _of_every(
        deployment.asking().omitting(CAPABILITY_TIMESTAMPS),
        f"every voice here declares {CAPABILITY_TIMESTAMPS!r}, so there is none "
        "that must refuse — CAP-3 is the arm they select",
        fault,
        "omitting it each refused both timestamp endpoints 501, saying why",
    )


def probe_cap_6(deployment: Deployment) -> Verdict:
    """A parameter this service cannot honour is named rather than dropped.

    Asked with a `voice_settings` leaf this service *models* and no engine
    honours, which is the half of README:24's rule 2 that `REF-7` does not reach:
    `REF-7` sends a field nobody modelled, so a deployment reporting only what its
    parser could not place keeps that claim and breaks this one. The two together
    say the rule holds for the parameters this build enumerates *and* for the ones
    it has never heard of.
    """
    asked = deployment.asking().once
    # Rule 2 is a promise about a request that is *served*: the parameter is
    # dropped and named back in the header, so a refusal over one is the rule
    # broken rather than a draw this claim could not read.
    refused = asked.refusal(UNHONOURABLE_DRAW)
    if refused is not None:
        return Broken(
            f"{refused} — README:24's rule 2 is that a parameter this service "
            "cannot honour is dropped and named back, so a request carrying one "
            "is served rather than refused over"
        )
    named = _ignored(asked.spoke(UNHONOURABLE_DRAW))
    wanted = f"voice_settings.{UNHONOURABLE_SETTING}"
    if wanted not in named:
        return Broken(
            f"a request carrying {wanted} was answered 200 with "
            f"{_ELVENSPEAK_HEADER}ignored {named or 'absent'} — a setting no "
            "engine here honours was dropped without a word"
        )
    return Held(f"{wanted} was named as ignored rather than silently dropped")


def probe_cap_7(deployment: Deployment) -> Verdict:
    """`x-elvenspeak-ignored` is absent, not empty, when nothing was dropped.

    The distinction is the whole claim: a caller reading the header's *presence*
    as "something was dropped" is reading it exactly as README:22 documents, and an
    empty header sends them hunting for a parameter nobody dropped. Read off the
    draw that asks for nothing but the text, which is the only request in this run
    with nothing to report.
    """
    asked = deployment.asking().once
    header = _ignored_header(asked.spoke(PLAIN))
    if header is not None:
        return Broken(
            f"a request asking for nothing but the text came back with "
            f"{_ELVENSPEAK_HEADER}ignored {header!r} — the header is present where "
            "nothing was dropped, so its presence no longer means anything"
        )
    return Held(
        f"a request asking for nothing but the text carried no "
        f"{_ELVENSPEAK_HEADER}ignored header at all"
    )


def probe_cap_8(deployment: Deployment) -> Verdict:
    """A language a voice does not speak is named back; one it speaks is not.

    Both arms per voice, because a header that names every `language_code` and one
    that names none each satisfy half of this and a prober asking one arm cannot
    tell them apart.

    The unspoken tag is chosen from a family *no offered voice speaks*, and
    [`UNSPOKEN_LANGUAGES`] records why: this service narrows the eligible voices by
    language before it matches the id, so asking an `en` voice for `es` where an
    `es` voice exists substitutes to that voice and honours the tag. The resolved
    voice is therefore checked here too — the claim is about the language the voice
    that *spoke* does not speak, and a substitution means a different voice
    answered and this claim was never asked of the one addressed.
    """

    def fault(asked: Asked) -> str | None:
        refused = asked.refusal(UNSPOKEN_DRAW)
        if refused is not None:
            return (
                f"speaks {asked.voice.language!r} and then {refused} — a language "
                "it does not speak is one to serve and report ignored, never one "
                "to refuse over"
            )
        unspoken = asked.spoke(UNSPOKEN_DRAW)
        spoke = _spoke_in(unspoken)
        if spoke != asked.voice.id:
            raise Blocked(
                f"a language no voice here speaks was answered by {spoke!r} rather "
                f"than by {asked.voice.id!r}, which was addressed — this "
                "deployment substituted, so the claim has a different subject "
                "than the one asked"
            )
        if "language_code" not in _ignored(unspoken):
            return (
                f"speaks {asked.voice.language!r} and was asked for a language no "
                f"voice here speaks, answering 200 with "
                f"{_ELVENSPEAK_HEADER}ignored {_ignored(unspoken) or 'absent'} — "
                "the preference was dropped and the caller was not told"
            )
        refused_own = asked.refusal(OWN_FAMILY)
        if refused_own is not None:
            return (
                f"publishes {asked.voice.language!r}, was asked for exactly that, "
                f"and {refused_own} — a voice that refuses the language it "
                "publishes has contradicted its own listing, which is the same "
                "lie as `MOD-3`'s over a model id"
            )
        own = asked.spoke(OWN_FAMILY)
        if "language_code" in _ignored(own):
            return (
                f"speaks {asked.voice.language!r}, was asked for exactly that, and "
                "reported language_code ignored — a preference this voice met was "
                "reported as dropped"
            )
        return None

    return _of_every(
        deployment.asking().voices,
        NO_VOICE_OFFERED,
        fault,
        "each named an unspeakable language_code as ignored and an honoured one not",
    )


def probe_cap_10(deployment: Deployment) -> Verdict:
    """A region-tagged, upper-cased tag is the same request as the bare family.

    `en-GB`, `es_MX` and `ES` are one question, and this asks all three reductions
    at once: [`_mangled`] upper-cases the family, separates with an underscore and
    adds a numeric region, so only a deployment performing every reduction answers
    it as it answered the bare family. A server comparing raw strings honours the
    bare tag and reports the variant ignored, which is the failure — and it is
    invisible to any check that sends only the canonical spelling.
    """

    def fault(asked: Asked) -> str | None:
        mangled = _mangled(asked.voice.language)
        # Both spellings, because this claim's subject is that they are one
        # request: refusing either is the deployment reading a tag it should have
        # reduced as a value it can reject.
        for draw, spelling in ((OWN_VARIANT, mangled), (OWN_FAMILY, asked.voice.language)):
            refused = asked.refusal(draw)
            if refused is not None:
                return (
                    f"speaks {asked.voice.language!r} and then {refused} — "
                    f"{spelling!r} spells the language this voice publishes, so it "
                    "is one to reduce and serve rather than to refuse over"
                )
        variant = asked.spoke(OWN_VARIANT)
        bare = asked.spoke(OWN_FAMILY)
        if _ignored(variant) != _ignored(bare):
            return (
                f"speaks {asked.voice.language!r} and answered {mangled!r} with "
                f"{_ELVENSPEAK_HEADER}ignored {_ignored(variant) or 'absent'} "
                f"where the bare {asked.voice.language!r} got "
                f"{_ignored(bare) or 'absent'} — the tag was compared before it "
                "was reduced to its family"
            )
        if _spoke_in(variant) != _spoke_in(bare):
            return (
                f"answered {mangled!r} in {_spoke_in(variant)!r} and the bare "
                f"{asked.voice.language!r} in {_spoke_in(bare)!r} — two spellings "
                "of one language reached two different voices"
            )
        return None

    return _of_every(
        deployment.asking().voices,
        NO_VOICE_OFFERED,
        fault,
        "each answered a region-tagged upper-cased tag exactly as its bare family",
    )


# ---------------------------------------------------------------- model probes


def _model_entries(deployment: Deployment) -> list[Mapping[str, Any]]:
    """`GET /v1/models`' entries, or [`Blocked`] saying why it published none.

    `MOD-1` — that this endpoint answers a bare array — is `e16.2`'s claim, so a
    body of the wrong shape is a precondition failing here rather than a verdict
    this file writes twice ([LAW:single-enforcer]).
    """
    reply = deployment.listed_models()
    if reply.status != 200:
        raise Blocked(
            f"GET /v1/models answered {reply.status}: {reply.body[:200]!r}, so "
            "this deployment published no account of what its voices answer to"
        )
    try:
        listing = json.loads(reply.body)
    except ValueError:
        raise Blocked(
            f"GET /v1/models answered a body that is not JSON: {reply.body[:200]!r}"
        ) from None
    if not isinstance(listing, list) or not all(
        isinstance(entry, dict) for entry in listing
    ):
        raise Blocked(
            f"GET /v1/models answered {listing!r:.200} rather than an array of "
            "model objects — MOD-1 is the claim that reports it"
        )
    return listing


def _model_ids(entries: list[Mapping[str, Any]]) -> tuple[str, ...]:
    """Every `model_id` `entries` names, or [`Blocked`] if one is not a name."""
    named = [entry.get("model_id") for entry in entries]
    if not all(isinstance(one, str) and one for one in named):
        raise Blocked(
            f"GET /v1/models published an entry with no usable `model_id`: "
            f"{named!r:.200} — nothing can be sent to it"
        )
    return tuple(str(one) for one in named)


def _model_sent(asked: Asked, draw: str) -> str:
    """The `model_id` `draw` carried, for a complaint to quote back.

    Read off the draw rather than re-derived from the voice, so the id a fault
    names is the id that was really sent ([LAW:one-source-of-truth]). A draw that
    could not be composed raises past here, which is the same [`Blocked`] the
    answer it has no answer for would have raised.
    """
    composed = asked.draws[draw]
    if isinstance(composed, Blocked):
        raise composed
    return str(composed.body["model_id"])


def probe_mod_2(deployment: Deployment) -> Verdict:
    """The models listing is the union of the offered voices', and never `router`.

    Self-consistent, and it is the claim that makes a router's account of the
    fleet behind it observable at all: `piper-routing-7e2.17` moved the served set
    onto the voice so this listing could be derived from the voices instead of from
    the process's own name, and a router that regressed to naming itself advertises
    an id no voice answers to. A deployment wrong in both places passes this, which
    is why the report counts it apart from the falsifiable claims.
    """
    published = set(_model_ids(_model_entries(deployment)))
    served = set(deployment.asking().catalogue.served)
    faults = tuple(
        f"{word} {sorted(names)!r:.200}"
        for word, names in (("omits", served - published), ("invents", published - served))
        if names
    )
    if "router" in published:
        return Broken(
            "GET /v1/models offers 'router', which is the name of a deployment "
            "that synthesizes nothing — a caller sending it reaches no engine"
        )
    if faults:
        return Broken(
            f"GET /v1/models {' and '.join(faults)} against the union of what the "
            "offered voices answer to — the two accounts of one fact disagree"
        )
    return Held(
        f"{_counted(len(published), 'model id')}, exactly the union of what "
        f"{_counted(len(deployment.asking().voices), 'offered voice')} answer to"
    )


def probe_mod_3(deployment: Deployment) -> Verdict:
    """A `model_id` the resolved voice lists is served and reported honoured.

    Per voice, because behind a router the voices answer to different ids in one
    process: an id piper's voices list is honoured for them and must be refused
    for kokoro's, and a deployment-wide check could not tell a server reading the
    voice from one reading its own name.
    """

    def fault(asked: Asked) -> str | None:
        refused = asked.refusal(LISTED_MODEL)
        if refused is not None:
            return (
                f"lists {_model_sent(asked, LISTED_MODEL)!r} among its own models "
                f"and then {refused} — an id a voice publishes as its own is one "
                "that voice must serve"
            )
        served = asked.spoke(LISTED_MODEL)
        if "model_id" in _ignored(served):
            return (
                f"lists {_model_sent(asked, LISTED_MODEL)!r} among its own models "
                "and then reported model_id ignored — the id chose the engine that "
                "spoke, so calling it ignored is the header lying in the one "
                "direction rule 2 cannot afford"
            )
        return None

    return _of_every(
        deployment.asking().voices,
        NO_VOICE_OFFERED,
        fault,
        "each served a model_id it lists without reporting it ignored",
    )


def probe_mod_4(deployment: Deployment) -> Verdict:
    """An id naming an engine that is not speaking this voice is a 422 that says so.

    The refusal `piper-routing-7e2.2` exists to force: a caller asks for kokoro,
    hears fluent piper, and nothing in the response says the engine it named never
    spoke. A voice may substitute because a substitute voice is still an answer;
    an engine may not, because the engine *is* the answer.

    The body must name all three things a caller needs to act — the id they sent,
    the voice that resolved, and the set that would have worked — and it is matched
    on a word boundary by [`_names`], the rule `e16.4` settled for every refusal
    here. Reused rather than re-derived: three copies of that rule existed and two
    of them were satisfied by `context` for `text`.

    Unasked on a single-engine deployment, and that is the honest verdict rather
    than a gap: every id such a deployment publishes is answered by every voice it
    offers, so from outside there is no published id that names an engine which is
    not speaking. [`Catalogue.elsewhere_than`] says so in the blocker. Behind a
    router fronting two engines the claim becomes askable, which is one of the
    real differences between the two deployments this epic compares.
    """

    def fault(asked: Asked) -> str | None:
        answer = asked.answered(ELSEWHERE_MODEL)
        named = _model_sent(asked, ELSEWHERE_MODEL)
        if answer.status != 422:
            return (
                f"answered {answer.status} to model_id {named!r}, which another "
                "voice here lists and this one does not: "
                f"{answer.body[:200]!r} — a caller who named an engine heard "
                "another one and was not told"
            )
        missing = [
            word
            for word in (named, asked.voice.id, "served")
            if not _names(answer.body, word)
        ]
        if missing:
            return (
                f"refused model_id {named!r} 422 with a body naming none of "
                f"{missing}: {answer.body[:200]!r} — a caller cannot read out of "
                "it which engine they reached or what would have worked"
            )
        return None

    return _of_every(
        deployment.asking().voices,
        NO_VOICE_OFFERED,
        fault,
        "each refused another voice's model_id 422, naming the id, itself and the "
        "served set",
    )


def probe_mod_5(deployment: Deployment) -> Verdict:
    """A `model_id` naming no engine here is served, and named back as ignored.

    `MOD-4`'s other reading of one decision, and the doc says why both are needed:
    a prober asking only one of them cannot tell a server that refuses everything
    from one that ignores everything. Every stock ElevenLabs client sends a
    `model_id` this service has never mapped, so this arm is the one that decides
    whether such a client works at all.
    """

    def fault(asked: Asked) -> str | None:
        answer = asked.answered(UNMAPPED_DRAW)
        if answer.status == 422:
            return (
                f"refused model_id {UNMAPPED_MODEL!r} 422: {answer.body[:200]!r} — "
                "it names no engine here, so it contradicted nothing, and every "
                "stock client sending an unmapped model is now refused"
            )
        if answer.status != 200:
            return f"answered {answer.status} to model_id {UNMAPPED_MODEL!r}: {answer.body[:200]!r}"
        if "model_id" not in _ignored(answer):
            return (
                f"served model_id {UNMAPPED_MODEL!r} with "
                f"{_ELVENSPEAK_HEADER}ignored {_ignored(answer) or 'absent'} — the "
                "id steered nothing and the caller was told it was honoured"
            )
        return None

    return _of_every(
        deployment.asking().voices,
        NO_VOICE_OFFERED,
        fault,
        f"each served {UNMAPPED_MODEL!r} and named model_id as ignored",
    )


def probe_mod_6(deployment: Deployment) -> Verdict:
    """`voice_id` decides who speaks and `model_id` is only read against that.

    Read off the two `model_id` draws that are served — the voice's own id and one
    naming no engine — because a server letting `model_id` choose the speaker
    would answer them in two different voices, and neither answer alone shows it.
    The claim is the ranking between the two parameters, so it needs both.
    """

    def fault(asked: Asked) -> str | None:
        for draw in (LISTED_MODEL, UNMAPPED_DRAW):
            refused = asked.refusal(draw)
            if refused is not None:
                return (
                    f"was addressed and then {refused} — the request named this "
                    "voice and its model_id ended it, so the id outranked the "
                    "voice rather than being read against it"
                )
            spoke = _spoke_in(asked.spoke(draw))
            if spoke != asked.voice.id:
                return (
                    f"was addressed and then answered by {spoke!r} when the "
                    f"request carried {draw} — model_id moved the speaker, so a "
                    "caller naming a voice cannot rely on reaching it"
                )
        return None

    return _of_every(
        deployment.asking().voices,
        NO_VOICE_OFFERED,
        fault,
        "each spoke for itself whichever model_id the request carried",
    )


def probe_mod_7(deployment: Deployment) -> Verdict:
    """Each model entry's `languages` are the languages the id really reaches.

    Self-consistent, and a second way a router's account of its fleet is
    observable: an id reaching only piper's English voices must not advertise the
    Spanish a kokoro voice speaks, or a caller picks a model that cannot say what
    they sent. Derived from the voices that list each id rather than from the
    deployment's own summary, which is the only reading that can disagree.
    """
    entries = _model_entries(deployment)
    voices = deployment.asking().catalogue.voices
    for entry, model_id in zip(entries, _model_ids(entries), strict=True):
        reached = frozenset(
            voice.language for voice in voices if model_id in voice.models
        )
        published = entry.get("languages")
        if not isinstance(published, list) or not all(
            isinstance(one, dict) and isinstance(one.get("language_id"), str)
            for one in published
        ):
            return Broken(
                f"model {model_id!r} published `languages` as {published!r:.200} "
                "rather than an array of objects carrying a `language_id`"
            )
        stated = frozenset(str(one["language_id"]) for one in published)
        if stated != reached:
            return Broken(
                f"model {model_id!r} advertises {sorted(stated)} while the voices "
                f"listing it speak {sorted(reached)} — a caller choosing it by "
                "language reaches a voice that cannot say what they sent"
            )
    return Held(
        f"{_counted(len(entries), 'model')} each advertise exactly the languages "
        "of the voices that list them"
    )


# --------------------------------------------------------- substitution probes


def probe_sub_1(deployment: Deployment) -> Verdict:
    """An unknown `voice_id` is answered in audio, saying what spoke and what was asked.

    One of two arms of one request, and the prober reports which it got rather
    than requiring one: a deployment with substitution switched off answers 404
    and that is `SUB-4`, keeping a different documented promise. Both headers are
    required together — the voice that spoke is useless to a caption without the
    id that was asked for, and either alone leaves a caller unable to tell a
    substitution from a direct hit.
    """
    asked = deployment.asking().once
    answer = asked.answered(FOREIGN_DRAW)
    if answer.status == 404:
        return Unasked(
            f"a voice_id nothing here offers answered 404, so this deployment has "
            "substitution switched off and the arm that returns audio has no "
            "subject — SUB-4 is the arm it selects"
        )
    if answer.status != 200:
        return Broken(
            f"a voice_id nothing here offers answered {answer.status}: "
            f"{answer.body[:200]!r} — neither audio from a substitute nor the 404 "
            "of a deployment that refuses to substitute"
        )
    spoke, requested = _spoke_in(answer), _asked_for(answer)
    if spoke is None or requested != FOREIGN_VOICE:
        return Broken(
            f"a voice_id nothing here offers was answered with audio, "
            f"{_ELVENSPEAK_HEADER}voice {spoke!r} and "
            f"{_ELVENSPEAK_HEADER}voice-requested {requested!r} — a caller cannot "
            f"tell that {FOREIGN_VOICE!r} was not what spoke"
        )
    return Held(
        f"{FOREIGN_VOICE!r} was answered in {spoke!r}, with both the voice that "
        "spoke and the voice that was asked for named"
    )


def probe_sub_2(deployment: Deployment) -> Verdict:
    """Every synthesis names what spoke, and names what was asked only when they differ.

    Read across every draw this run made rather than one, because the header is
    assembled once for every synthesis path and a leak or an omission on the
    timestamp endpoints would pass a check of the plain one. The `only when they
    differ` half is what makes the header readable at all: a `voice-requested` on
    every response tells a caller nothing, and one that never appears hides every
    substitution.
    """
    checked = 0
    for asked in deployment.asking().voices:
        for name, answer in asked.answers.items():
            if not (isinstance(answer, Reply) and answer.status == 200):
                continue
            checked += 1
            spoke, requested = _spoke_in(answer), _asked_for(answer)
            addressed = asked.addressed(name)
            if spoke is None:
                return Broken(
                    f"{name} answered 200 with no {_ELVENSPEAK_HEADER}voice — "
                    "README:115 promises it on every synthesis response, and "
                    "without it no caller knows who spoke"
                )
            substituted = spoke != addressed
            if substituted and not _echoed(requested, addressed):
                return Broken(
                    f"{name} was addressed at {addressed!r}, answered by {spoke!r}, "
                    f"and named {requested!r} as requested — a substitution whose "
                    "own record of what was asked for is wrong"
                )
            if not substituted and requested is not None:
                return Broken(
                    f"{name} was answered by the very voice it addressed "
                    f"({spoke!r}) and still carried "
                    f"{_ELVENSPEAK_HEADER}voice-requested {requested!r} — the "
                    "header no longer means a substitution happened"
                )
    if not checked:
        return Unasked(
            "no draw in this run was answered with audio, so no synthesis "
            "response's headers could be read"
        )
    return Held(
        # "synthesis response" rather than "synthesis": [`_counted`] pluralises by
        # appending an `s`, which is every noun in this file but that one.
        f"{_counted(checked, 'synthesis response')} each named the voice that "
        "spoke, and named the requested one exactly where the two differed"
    )


def probe_sub_3(deployment: Deployment) -> Verdict:
    """`x-elvenspeak-voice` always names a voice the listing offers.

    Self-consistent, and the fifth of the five claims a router misreporting its
    fleet passes: a substitute this deployment cannot name in `GET /v1/voices` is
    one a caller can hear and never address again, which makes the substitution
    unreproducible.
    """
    offered = {voice.id for voice in deployment.asking().catalogue.voices}
    spoken = set()
    for asked in deployment.asking().voices:
        for name, answer in asked.answers.items():
            if isinstance(answer, Reply) and answer.status == 200:
                spoke = _spoke_in(answer)
                if spoke is None:
                    # Whether a 200 must name the voice that spoke at all is
                    # `SUB-2`'s claim, not this one. Counted here it would sit in
                    # `spoken` as a voice no listing could offer and never be read
                    # against `offered`, leaving `held` reporting a count of voices
                    # one of which was never named ([LAW:no-silent-failure]).
                    continue
                if spoke not in offered:
                    return Broken(
                        f"{name} was answered by {spoke!r}, which GET /v1/voices "
                        "does not offer — a caller who heard it cannot ask for it "
                        "again"
                    )
                spoken.add(spoke)
    if not spoken:
        return Unasked(
            "no draw in this run was answered with audio, so no voice was named "
            "as having spoken"
        )
    return Held(
        f"{_counted(len(spoken), 'voice')} spoke across this run, every one of "
        "them offered by GET /v1/voices"
    )


def probe_sub_4(deployment: Deployment) -> Verdict:
    """With substitution off, an unknown `voice_id` is a 404 rather than audio.

    `SUB-1`'s other arm, and the one a deployment that must never answer in the
    wrong voice depends on. Unasked where substitution is configured, because
    there the request is answered in a substitute and this promise has no subject
    — which is what `SUB-1` reports.
    """
    asked = deployment.asking().once
    answer = asked.answered(FOREIGN_DRAW)
    if 200 <= answer.status < 300:
        return Unasked(
            f"a voice_id nothing here offers was answered {answer.status} with "
            "audio, so this deployment has substitution configured and the arm "
            "that refuses has no subject — SUB-1 is the arm it selects"
        )
    if answer.status != 404:
        return Broken(
            f"a voice_id nothing here offers answered {answer.status}: "
            f"{answer.body[:200]!r} — a deployment that will not substitute owes a "
            "404 for a voice it does not have"
        )
    if not _names(answer.body, "voice"):
        return Broken(
            f"a voice_id nothing here offers was refused 404 with "
            f"{answer.body[:200]!r}, which names no voice — a caller cannot tell "
            "this from a path this build does not serve"
        )
    return Held(f"{FOREIGN_VOICE!r} was refused 404 rather than answered by a substitute")


def probe_sub_5(deployment: Deployment) -> Verdict:
    """An alias a voice lists really reaches that voice.

    Asked of every alias of every voice, because an alias table is exactly the
    kind of map that drifts: the ids are foreign by definition, so nothing but
    this check reads them.

    An empty alias set leaves this `unasked` rather than `held`, and
    `probe_health_2` settled that question first in this file: zero aliases all
    resolving correctly is the shape of a check that cannot fail. Behind a router
    that is the expected state — the router's table is empty and aliases do not
    resolve through it (`piper-routing-7e2.15`) — so the blocker names the empty
    set, which is the real difference between a routed and a direct deployment
    rather than a defect in either.
    """
    aliased = tuple(
        (asked, alias)
        for asked in deployment.asking().voices
        for alias in asked.voice.aliases
    )
    if not aliased:
        return Unasked(
            "no voice here publishes an alias, so there is no foreign id to try — "
            "which is what a routed deployment looks like, its alias table being "
            "empty by design"
        )
    for asked, alias in aliased:
        draw = _alias_draw(alias)
        # Before the routing, because an alias the deployment refused reached no
        # voice at all, which is this claim being false rather than a reason it
        # could not be asked.
        refused = asked.refusal(draw)
        if refused is not None:
            return Broken(
                f"the alias {alias!r} is published on voice {asked.voice.id!r} "
                f"and then {refused} — an alias a voice publishes as its own is "
                "one that voice must serve"
            )
        spoke = _spoke_in(asked.spoke(draw))
        if spoke != asked.voice.id:
            return Broken(
                f"the alias {alias!r} is published on voice {asked.voice.id!r} and "
                f"reached {spoke!r} instead — the listing promises a mapping this "
                "deployment does not make"
            )
    return Held(
        f"{_counted(len(aliased), 'published alias')} each reached the voice that "
        "published it"
    )


def probe_sub_6(deployment: Deployment) -> Verdict:
    """A voice id whose bytes are not latin-1 substitutes rather than answering 500.

    A header value must be latin-1 on the wire, so a server echoing this id into
    `x-elvenspeak-voice-requested` unencoded raises inside its own response
    assembly — after the status is chosen — and answers 500 to a request it had
    already decided to serve. The escaping is not checked against a spelling:
    which escape a build picks is its own business, and holding this file to one
    would go red on a version skew that is not a defect. What is checked is that
    the caller got audio and a named substitute.
    """
    asked = deployment.asking().once
    answer = asked.answered(UNPRINTABLE_DRAW)
    if answer.status == 404:
        return Unasked(
            "a voice_id whose bytes are not latin-1 answered 404, so this "
            "deployment refuses to substitute and there is no header for it to "
            "have escaped — SUB-4 is where that refusal is reported"
        )
    if answer.status != 200:
        return Broken(
            f"a voice_id whose bytes are not latin-1 answered {answer.status}: "
            f"{answer.body[:200]!r} — the id reached the response headers "
            "unescaped, so a request this deployment meant to serve failed on the "
            "way out"
        )
    spoke, requested = _spoke_in(answer), _asked_for(answer)
    if spoke is None or requested is None:
        return Broken(
            f"a voice_id whose bytes are not latin-1 was answered 200 with "
            f"{_ELVENSPEAK_HEADER}voice {spoke!r} and voice-requested "
            f"{requested!r} — the substitution happened and was not reported"
        )
    return Held(
        f"{UNPRINTABLE_VOICE!r} was answered in {spoke!r}, its requested id "
        f"escaped into the header as {requested!r}"
    )


# --------------------------------------------------- per-voice discovery probes


def probe_disc_2(deployment: Deployment) -> Verdict:
    """`GET /v1/voices/{id}` returns that voice, and 404s an id it does not have.

    Discovery never substitutes, and that is the point of asking it beside
    `SUB-1`: the same unknown id that earns audio from a synthesis endpoint must
    earn a 404 here, because a client reading the catalogue to populate a picker
    would otherwise list a voice that does not exist. Reads only, so this costs
    one request per voice and no synthesis at all.
    """
    key = deployment.key_or_blocked()
    # [`ProbedVoice`] rather than the declarations, because the id is the whole of
    # what this claim reads and [`_declarations`] is all-or-nothing across four
    # fields: gating a read-back on every voice's `capabilities` parsing would
    # leave this unasked over a deployment it could have caught red-handed.
    voices = deployment.voices
    for voice in voices:
        path = f"/v1/voices/{urllib.parse.quote(voice.id, safe='')}"
        reply = _ask(deployment.target, path, key)
        if reply.status != 200:
            return Broken(
                f"GET {path} answered {reply.status} for a voice GET /v1/voices "
                f"offers: {reply.body[:200]!r}"
            )
        try:
            body = json.loads(reply.body)
        except ValueError:
            return Broken(
                f"GET {path} answered a body that is not JSON: {reply.body[:200]!r}"
            )
        if not isinstance(body, dict) or body.get("voice_id") != voice.id:
            return Broken(
                f"GET {path} answered {body!r:.200} — a read addressed at one "
                "voice came back describing another, so a client cannot trust "
                "which voice it is holding"
            )
    unknown = f"/v1/voices/{FOREIGN_VOICE}"
    refused = _ask(deployment.target, unknown, key)
    if refused.status != 404:
        return Broken(
            f"GET {unknown} answered {refused.status} rather than 404: "
            f"{refused.body[:200]!r} — discovery substituted, so a client reading "
            "the catalogue would list a voice nothing offers"
        )
    return Held(
        f"{_counted(len(voices), 'offered voice')} each read back as itself, and "
        "an id nothing offers was refused 404"
    )


def probe_disc_5(deployment: Deployment) -> Verdict:
    """Each voice's `language` is the bare family a `language_code` is compared in.

    Checked as reduction-idempotence rather than against a list of the 184
    registered codes, and the difference is deliberate. What makes this claim
    matter is that the published string is *comparable* to what a caller sends:
    `piper` once published `family: "ES"`, which no caller's `es` could ever
    equal, leaving the voice unreachable by language while the listing said it
    spoke Spanish. A tag already in its reduced form is exactly the falsifiable
    half of that, and it catches `ES`, `en-GB`, `en_US` and a padded ` en`. Whether
    a well-formed code is *registered* would need a table this file does not carry,
    for the reason [`DOCUMENTED_READS`] carries no route table: it would be this
    checkout's vocabulary asserted against a different build's.
    """
    # The language off [`ProbedVoice.published`] rather than off a [`Declared`],
    # for the reason `DISC-2` reads the thin stamp: this claim reads one field, and
    # [`_declarations`] would gate it on every voice's `capabilities`, `models` and
    # `aliases` parsing too. So a malformed `aliases` elsewhere no longer decides
    # whether a language can be read, and a language that is itself unusable still
    # does — that one is the precondition this claim genuinely has.
    voices = deployment.voices
    for voice in voices:
        language = voice.published.get("language")
        if not isinstance(language, str) or not language:
            raise Blocked(
                f"voice {voice.id!r} publishes {language!r:.100} as its language, "
                "so there is no tag to reduce — DISC-1 is the claim that every "
                "published field is present and well-typed"
            )
        reduced = language.strip().lower().replace("_", "-").split("-")[0]
        if language != reduced:
            return Broken(
                f"voice {voice.id!r} publishes language {language!r}, which "
                f"reduces to {reduced!r} — a caller sending {reduced!r} cannot "
                "match it, so the voice is unreachable by the language it claims"
            )
    return Held(
        f"{_counted(len(voices), 'voice')} each publish a language already in the "
        "family form a language_code is compared in"
    )


# --------------------------------------------------- deployment-wide capability


def probe_cap_5(deployment: Deployment) -> Verdict:
    """`GET /v1/models`' `capabilities` is the union over the offered voices.

    Self-consistent. A caller reads this list to decide whether to send a rate at
    all, and a union that overstates it sends them to a voice that reports the
    parameter ignored — which is rule 2 kept per voice and broken per deployment.

    Deployment-wide by contract, and deliberately not scoped per `model_id` the
    way `MOD-7` scopes `languages` beside it: the README makes this field "what
    this deployment can do at all", to be read when choosing a deployment rather
    than when deciding a request, and [`_model_json`] derives it once for every
    entry — where the reason the two fields differ is written, because the two
    overclaims do not cost the same. Read as `MOD-7`'s twin this looks like a
    missing filter, and scoping it here would report an honestly-differentiated
    router broken for publishing each engine's real capabilities.
    """
    entries = _model_entries(deployment)
    declared = frozenset[str]().union(
        *(voice.capabilities for voice in deployment.asking().catalogue.voices)
    )
    for entry, model_id in zip(entries, _model_ids(entries), strict=True):
        published = entry.get("capabilities")
        if not isinstance(published, list) or not all(
            isinstance(one, str) for one in published
        ):
            return Broken(
                f"model {model_id!r} published `capabilities` as "
                f"{published!r:.200} rather than an array of names"
            )
        if frozenset(published) != declared:
            return Broken(
                f"model {model_id!r} advertises capabilities {sorted(published)} "
                f"while the offered voices between them declare {sorted(declared)} "
                "— a caller reading the models listing to decide what to send is "
                "reading a different answer than the voices give"
            )
    return Held(
        f"{_counted(len(entries), 'model')} each advertise {sorted(declared)}, the "
        "union over the offered voices"
    )


def probe_cap_9(deployment: Deployment) -> Verdict:
    """A blank, whitespace-only or absent `language_code` expresses no preference.

    All three, because they are one fact arriving in three shapes and a server
    handling two of them is the bug this claim was written from: `""` is what a
    form or a JS client sends for "unset", and taken literally it is a language no
    voice speaks, so every such request came back reporting `language_code`
    dropped. A caller who expressed no preference was told their preference was
    ignored.
    """
    asked = deployment.asking().once
    # The two shapes that carry a `language_code` at all. `PLAIN` carries none,
    # so a refusal of it says nothing about how this service reads an unset tag
    # and is `FMT-1`'s subject — the claim that judges a refusal of a plain
    # synthesis request — rather than this one's.
    expressed = (EMPTY_LANGUAGE, BLANK_LANGUAGE)
    for draw in expressed:
        refused = asked.refusal(draw)
        if refused is not None:
            return Broken(
                f"{refused} — a caller who expressed no preference was refused "
                "over the absence, which is the literal reading this claim was "
                "written from made louder: the empty tag was taken for a value "
                "rather than for no preference at all"
            )
    for draw in (*expressed, PLAIN):
        named = _ignored(asked.spoke(draw))
        if "language_code" in named:
            return Broken(
                f"{draw} came back naming language_code in "
                f"{_ELVENSPEAK_HEADER}ignored {named} — nothing was asked for, so "
                "there was nothing to report as dropped"
            )
    return Held(
        "an empty, a whitespace-only and an absent language_code were each read as "
        "no preference and none was reported ignored"
    )


# ------------------------------------------------- the timestamp endpoints' shape


#: The two values README:187 publishes for `x-elvenspeak-alignment`, which are
#: also the two an object's `alignment_fidelity` may take. Spelled here rather
#: than imported from `elvenspeak.alignment` for the reason this module imports
#: nothing else from the package: the deployment under test is routinely a
#: different build, so a vocabulary read out of this checkout would be asserting
#: the implementation against itself ([LAW:behavior-not-structure]).
PUBLISHED_FIDELITIES = ("word-exact", "interpolated")

ALIGNMENT_HEADER = f"{_ELVENSPEAK_HEADER}alignment"

def _refused_own_endpoint(asked: Asked, endpoint: str) -> str | None:
    """How this voice turned down an endpoint its own declaration promised.

    Judged before any claim reads a body, because [`Asked.spoke`] raises
    [`Blocked`] on a refusal and [`_finding`] turns that into `unasked` — so a
    deployment refusing the very request a claim tests would be reported as one
    the prober could not look at. The declaration is what makes the refusal a
    verdict here rather than a precondition: the voice published `timestamps`,
    so this draw is a promise it made ([LAW:single-enforcer] — what a refusal
    means to these six claims is decided once, not remembered six times).
    """
    refused = asked.refusal(_timestamp_draw(endpoint))
    if refused is None:
        return None
    return (
        f"declares {CAPABILITY_TIMESTAMPS!r} and then {refused} — a voice that "
        "refuses the endpoint it promises timings from has contradicted its own "
        "declaration"
    )


def _timings(asked: Asked, endpoint: str) -> tuple[Timed, ...] | str:
    """`endpoint`'s objects for this voice, or the fault a claim about them reports.

    The preamble the four `TIME` claims that read objects share, so none of them
    can reach the objects without the refusal already judged.

    Either fault comes back whole, which is what lets those four `return` it
    unexamined. A refusal already names its draw; a parse fault out of [`_timed`]
    knows only the body it read, so the endpoint is put back here — once, rather
    than by the two claims that ask both endpoints and would otherwise report "a
    body that is not JSON" without saying which of the two sent it
    ([LAW:one-source-of-truth]).

    `CAP-3` reads these bodies too and deliberately stands outside this: a
    refusal is `CAP-4`'s answer rather than its own and its message says so,
    which needs the refusal told apart from a malformed body — a distinction the
    `str` returned here collapses on purpose, because these four callers report
    both the same way. That costs `CAP-3` nothing that matters: what a refusal
    *means* is decided in [`_refused_own_endpoint`], which it calls directly
    ([LAW:single-enforcer] — the rule has one home, and this is a composition of
    it rather than a second one).
    """
    refused = _refused_own_endpoint(asked, endpoint)
    if refused is not None:
        return refused
    objects = _timed(asked.spoke(_timestamp_draw(endpoint)).body)
    if isinstance(objects, str):
        return f"{endpoint} {objects}"
    return objects


def probe_time_2(deployment: Deployment) -> Verdict:
    """`/with-timestamps` names which of the two kinds of timing it measured.

    The header is the only place the non-streaming endpoint can say it.
    `alignment.py:20` takes word boundaries from the model where it reports them
    and spreads the utterance evenly where it does not, and those two timelines
    are worth very different amounts to a caller aligning subtitles — but they
    are the same floats. A deployment answering timings and naming no fidelity
    has published a timeline whose worth cannot be known.
    """

    def fault(asked: Asked) -> str | None:
        refused = _refused_own_endpoint(asked, WITH_TIMESTAMPS)
        if refused is not None:
            return refused
        reported = asked.spoke(_timestamp_draw(WITH_TIMESTAMPS)).headers.get(
            ALIGNMENT_HEADER
        )
        if reported is None:
            return (
                f"answered {WITH_TIMESTAMPS} carrying no {ALIGNMENT_HEADER} "
                "header — a caller cannot tell measured word boundaries from a "
                "span spread evenly over characters, and only one of those is "
                "worth aligning against"
            )
        if reported not in PUBLISHED_FIDELITIES:
            return (
                f"answered {ALIGNMENT_HEADER}: {reported!r}, which is neither of "
                f"the two values published for it ({' and '.join(PUBLISHED_FIDELITIES)})"
            )
        return None

    return _of_every(
        deployment.asking().declaring(CAPABILITY_TIMESTAMPS),
        NO_TIMESTAMP_VOICE,
        fault,
        f"declaring {CAPABILITY_TIMESTAMPS!r} each answered {WITH_TIMESTAMPS} "
        f"with an {ALIGNMENT_HEADER} of "
        f"{' or '.join(PUBLISHED_FIDELITIES)}",
    )


def probe_time_3(deployment: Deployment) -> Verdict:
    """The streaming endpoint reports fidelity per object and never in a header.

    Both halves, because they are one decision: `api.py:998` withholds the header
    precisely because fidelity is settled per sentence and can differ between the
    objects of one response, so a single header could only report one of several
    answers and would have to be sent before any of them were known. A deployment
    that sends it anyway has published a value it could not have known, and one
    whose objects carry no fidelity of their own has left the caller with
    nowhere to read it.

    [`STREAMED_TEXT`] is what makes this falsifiable rather than decorative: over
    a one-sentence run "one object per line" and "one object in total" are the
    same observation.
    """

    def fault(asked: Asked) -> str | None:
        objects = _timings(asked, STREAMED_TIMESTAMPS)
        if isinstance(objects, str):
            return objects
        carried = asked.spoke(_timestamp_draw(STREAMED_TIMESTAMPS)).headers.get(
            ALIGNMENT_HEADER
        )
        if carried is not None:
            return (
                f"answered {STREAMED_TIMESTAMPS} carrying {ALIGNMENT_HEADER}: "
                f"{carried!r} — fidelity is settled per object here, so one "
                f"header over {_counted(len(objects), 'object')} reports at most "
                "one of them and was sent before any was known"
            )
        unpublished = tuple(
            timed.fidelity
            for timed in objects
            if timed.fidelity not in PUBLISHED_FIDELITIES
        )
        if unpublished:
            return (
                f"answered {STREAMED_TIMESTAMPS} with "
                f"{_counted(len(unpublished), 'object')} whose "
                f"`alignment_fidelity` is not published: {unpublished!r:.200} — "
                f"the two published are {' and '.join(PUBLISHED_FIDELITIES)}"
            )
        return None

    return _of_every(
        deployment.asking().declaring(CAPABILITY_TIMESTAMPS),
        NO_TIMESTAMP_VOICE,
        fault,
        f"declaring {CAPABILITY_TIMESTAMPS!r} each answered "
        f"{STREAMED_TIMESTAMPS} as one object per line, every one carrying its "
        f"own `alignment_fidelity` and none carrying {ALIGNMENT_HEADER}",
    )


def probe_time_4(deployment: Deployment) -> Verdict:
    """Every object's timeline ascends and accounts for its own audio.

    The claim that makes an alignment worth anything: a timeline that runs
    backwards, or that stops halfway through the samples, is a caller's subtitles
    drifting away from the speech. Read against the object's **own**
    `audio_base64` rather than against a second draw's byte count, which is what
    lets it be asked of a deployment that samples — the sampling that leaves
    `CAP-1`, `FMT-4` and `FMT-7` unaskable against the live piper deployment
    (`piper-conformance-e16.ysu`) cannot touch this, because there is only one
    synthesis involved and the alignment describes exactly those bytes.

    Asked of both endpoints and of every object, because a streamed response is
    where a cumulative offset can slip: `TIME-6` checks that consecutive objects
    meet, and this checks that each one really covers the audio it carries.

    Both timelines are ordered, not just the one the document names. The row says
    "character end times ascend", and that is its shorthand rather than its
    boundary: a deployment that swaps two adjacent *start* times leaves `ends`
    non-decreasing and [`Timed.span`] honest, so reading `ends` alone reports it
    `held` while a caller drawing each character when its start time arrives
    watches the clock run backwards. Both arrays are one timeline and a correct
    deployment has both ordered, so asking of both is strictly stronger and
    cannot refuse an honest answer.

    Non-decreasing rather than strictly ascending, deliberately. A *descending*
    time is unambiguously a broken timeline, but a zero-width character — a word
    whose measured span rounds to nothing — is a fact about an engine rather than
    a promise this service breaks, and demanding strict ascent would report a
    legitimate deployment broken ([LAW:behavior-not-structure]).

    Where the first object begins is read separately, because every other reading
    here is a *difference* and so cannot see a constant: [`Timed.span`] subtracts
    two times, and `TIME-6` subtracts across a pair. A deployment that leaks the
    `elapsed` accumulator `api.py:979` carries between sentences — failing to
    reset it between top-level requests — offsets every time in the response by
    the same amount, leaving every span exact, every gap exact, and the caller's
    subtitles uniformly late against the audio those very times arrived with.
    Only the first object is asked: a later streamed object is *supposed* to
    begin where its predecessor ended, which is what `TIME-6` reads.
    """
    rate = _published(PCM_FORMAT).sample_rate

    def fault(asked: Asked) -> str | None:
        for endpoint in _TIMESTAMP_ENDPOINTS:
            objects = _timings(asked, endpoint)
            if isinstance(objects, str):
                return objects
            begins = objects[0].starts[0]
            if abs(begins) > _LENGTH_SPREAD_SECONDS:
                return (
                    f"answered {endpoint} with a timeline whose first character "
                    f"begins at {begins:.3f}s rather than at the start of the "
                    "audio it arrived with — every span and every gap below can "
                    "be right while the whole timeline sits off its own samples, "
                    "so a caller drawing to these times is uniformly late"
                )
            for timed in objects:
                for field, times in timed.timelines:
                    backwards = tuple(
                        (before, after)
                        for before, after in zip(times, times[1:])
                        if after < before
                    )
                    if backwards:
                        return (
                            f"answered {endpoint} with "
                            f"{_counted(len(backwards), 'time')} in `{field}` "
                            f"that go backwards, the first {backwards[0][0]:.6f}s "
                            f"followed by {backwards[0][1]:.6f}s — a caller "
                            "reading this forwards is handed a timeline that "
                            "reverses"
                        )
                seconds = len(timed.audio) / BYTES_PER_SAMPLE / rate
                if abs(timed.span - seconds) > _LENGTH_SPREAD_SECONDS:
                    return (
                        f"answered {endpoint} with a timeline running "
                        f"{timed.span:.3f}s over audio of {seconds:.3f}s — a "
                        f"{abs(timed.span - seconds):.3f}s disagreement, so the "
                        "timings do not account for the samples they arrived with"
                    )
        return None

    return _of_every(
        deployment.asking().declaring(CAPABILITY_TIMESTAMPS),
        NO_TIMESTAMP_VOICE,
        fault,
        f"declaring {CAPABILITY_TIMESTAMPS!r} each answered both timestamp "
        f"endpoints with start and end times that never reverse and a timeline "
        f"beginning at its own audio and covering it within "
        f"{_LENGTH_SPREAD_SECONDS:g}s",
    )


def probe_time_5(deployment: Deployment) -> Verdict:
    """`alignment` and `normalized_alignment` are one object, not two.

    ElevenLabs distinguishes the alignment of the text as written from the text
    as normalized for speech, and `api.py:1303` answers both with the same object
    because the engine is fed the text as written. A caller that reads
    `normalized_alignment` — the SDK's own examples do — must get a timeline, and
    a deployment answering an empty or differently-timed one there has broken
    exactly those callers while satisfying every check that reads `alignment`.

    Asked of every object of both endpoints, because the two are built by one
    function and a deployment that got it right on the endpoint a prober is most
    likely to ask is the failure worth catching.
    """

    def fault(asked: Asked) -> str | None:
        for endpoint in _TIMESTAMP_ENDPOINTS:
            objects = _timings(asked, endpoint)
            if isinstance(objects, str):
                return objects
            for timed in objects:
                if timed.normalized != timed.alignment:
                    return (
                        f"answered {endpoint} with a `normalized_alignment` that "
                        f"is not its `alignment`: {timed.normalized!r:.200} "
                        f"against {timed.alignment!r:.200}"
                    )
        return None

    return _of_every(
        deployment.asking().declaring(CAPABILITY_TIMESTAMPS),
        NO_TIMESTAMP_VOICE,
        fault,
        f"declaring {CAPABILITY_TIMESTAMPS!r} each answered every object of both "
        "timestamp endpoints with `normalized_alignment` equal to `alignment`",
    )


def probe_time_6(deployment: Deployment) -> Verdict:
    """A streamed run's objects lay end to end, leaving no gap and no overlap.

    The claim a caller concatenating a stream depends on and the one a server is
    most likely to get quietly wrong: `api.py:979` carries the elapsed time
    forward from each sentence's own alignment, and a deployment that restarted
    every object at zero, or derived the offset from the audio instead, would
    answer objects that each look correct alone and slide against each other in
    sequence. Only laying consecutive objects side by side finds it.

    A response carrying one object is reported `unasked` rather than `held`.
    There is no pair to compare, so a verdict of `held` would be a check that
    cannot fail — which is the shape `docs/conformance-claims.md` refuses — and
    the reason is a precondition of this deployment's, not a promise it broke:
    nothing documented here obliges the endpoint to split a text into sentences.
    """

    def fault(asked: Asked) -> str | None:
        objects = _timings(asked, STREAMED_TIMESTAMPS)
        if isinstance(objects, str):
            return objects
        if len(objects) < 2:
            raise Blocked(
                f"voice {asked.voice.id!r} answered {STREAMED_TIMESTAMPS} with "
                f"one object for a text of two sentences ({STREAMED_TEXT!r}), so "
                "there is no consecutive pair to lay end to end"
            )
        for before, after in zip(objects, objects[1:]):
            if abs(after.starts[0] - before.ends[-1]) > _LENGTH_SPREAD_SECONDS:
                return (
                    f"answered {STREAMED_TIMESTAMPS} with an object ending at "
                    f"{before.ends[-1]:.3f}s and the next starting at "
                    f"{after.starts[0]:.3f}s — a "
                    f"{abs(after.starts[0] - before.ends[-1]):.3f}s "
                    "step, so a caller laying these out in sequence has them "
                    "sliding against their own audio"
                )
        return None

    return _of_every(
        deployment.asking().declaring(CAPABILITY_TIMESTAMPS),
        NO_TIMESTAMP_VOICE,
        fault,
        f"declaring {CAPABILITY_TIMESTAMPS!r} each answered "
        f"{STREAMED_TIMESTAMPS} with consecutive objects meeting within "
        f"{_LENGTH_SPREAD_SECONDS:g}s",
    )


# ------------------------------------------------- routes this service refuses


#: A path nothing routes, invented rather than borrowed from ElevenLabs' own
#: surface. `/v1/user/subscription` was the obvious candidate and is the wrong
#: one: a claim that a real endpoint goes unrouted breaks the day a deployment
#: implements it, which is a deployment getting better and the prober calling it
#: worse.
UNROUTED_PATH = "/v1/no-such-endpoint"

#: A documented synthesis path asked under a method it does not accept.
#:
#: `GET` because it is the safe verb. This prober runs against deployments its
#: operator does not own, and reading `DELETE` off the endpoint table would mean
#: sending one at a stranger's server to find out whether it refuses them.
#:
#: The voice id is a placeholder rather than a catalogued one: a method is
#: decided by the router, before any handler resolves a voice, so a real id would
#: buy nothing and would leave this claim waiting on a listing it has no stake in
#: — `ROUTE-1` and `ROUTE-2` are the only claims here with no precondition at all.
WRONG_METHOD_PATH = "/v1/text-to-speech/any-voice"

#: The documented paths a refusal has to name back as endpoints it does serve.
#: Only the untemplated ones, because those have a single spelling: how a foreign
#: build writes the `{voice_id}` segment is its own business, and demanding this
#: one's spelling would report a conforming deployment as broken over a brace
#: ([LAW:behavior-not-structure] — the promise is that the list is real, not that
#: it is formatted like ours).
_NAMED_BACK = tuple(
    path
    for path in (*DOCUMENTED_READS, *DOCUMENTED_SYNTHESES)
    if "{" not in path
)


@dataclass(frozen=True)
class Unrouted:
    """One request this service refuses before any handler sees it, and its status.

    Two rows, because there are two fall-throughs and closing one closes neither:
    a path no route matches, and a path matched under a method it does not
    accept. They differ in the request and in the status it earns, which are
    values — so they are two instances of one claim shape rather than two kinds
    of claim ([LAW:one-type-per-behavior]).
    """

    described: str
    method: str
    path: str
    status: int


#: The two refusals the router owes, keyed by the claim that reports each.
UNROUTED: dict[str, Unrouted] = {
    "ROUTE-1": Unrouted(
        described=f"GET {UNROUTED_PATH}, a path nothing routes",
        method="GET",
        path=UNROUTED_PATH,
        status=404,
    ),
    "ROUTE-2": Unrouted(
        described=f"GET {WRONG_METHOD_PATH}, a path README publishes over POST",
        method="GET",
        path=WRONG_METHOD_PATH,
        status=405,
    ),
}


def _names_path(body: bytes, path: str) -> bool:
    """`body` names `path` as a whole path, rather than as the head of a longer one.

    [`_names`]' reason, one character further out. A word boundary is no boundary
    between paths: `/v1/voices` ends at a `/` in `/v1/voices/settings/default`,
    so `\\b` matches there and the shorter path is satisfied by the longer one
    being present. Both are in [`_NAMED_BACK`], which made that entry's check one
    that could not fail — a deployment naming only the settings path back was
    read as having named the catalogue too.

    So the boundary is "not more path": a segment character after the match means
    this is a different, longer path and not the one asked about.
    """
    return re.search(re.escape(path.encode()) + rb"(?![\w/-])", body) is not None


def probe_unrouted(unrouted: Unrouted, deployment: Deployment) -> Verdict:
    """A request this service does not route is refused naming it, and what is served.

    Read out of the raw body rather than at a key, for the reason [`Refusal`]
    gives: the promise is that the refusal *names* what was sent and what is on
    offer instead, and a check keyed to `detail.message` would report a
    deployment broken for having answered the same facts in a different shape.

    The key is passed when the operator gave one and withheld when they did not,
    and neither row has a precondition of its own — no catalogue, no voice, no
    synthesis — because refusing a request needs none of them.

    What it does need is to reach the router at all, which is not a given: a
    guard in front of one answers a refused key before any route is consulted,
    and this deployment's own `Depends` guard is only the arrangement where the
    router wins. A deployment behind an authenticating proxy refuses first, and
    reading that refusal as the router's verdict would report a sound router
    broken for a key the operator mistyped. So an [`AUTH_REFUSALS`] status is
    [`Blocked`] here exactly as it is when the catalogue cannot be read: the
    claim goes unasked pointing at `--key`, because what this deployment does
    with an unrouted request was never seen ([LAW:no-silent-failure] — the
    blocker has to send an operator to their own invocation, not to the
    server).

    The named-back check is what separates this from a refusal that merely got
    wordier. A body saying only "not found, sorry" carries the method and no
    route a caller could reach for, and [`_NAMED_BACK`] is the smallest set whose
    absence proves the list is not there.
    """
    answer = _ask(
        deployment.target, unrouted.path, deployment.target.key, method=unrouted.method
    )
    if answer.status in AUTH_REFUSALS:
        raise Blocked(
            f"{unrouted.described} answered {answer.status}, a guard refusing "
            "the xi-api-key the prober was given rather than the router's own "
            "verdict, so what this deployment does with a request it cannot "
            "route could not be seen — check --key before reading this as a "
            "fault of the deployment"
        )
    if answer.status != unrouted.status:
        return Broken(
            f"{unrouted.described} answered {answer.status} rather than the "
            f"documented {unrouted.status}: {answer.body[:200]!r}"
        )
    if not _names(answer.body, unrouted.method):
        return Broken(
            f"{unrouted.described} was refused {answer.status} by a body that "
            f"does not name the method it refused: {answer.body[:200]!r} — a "
            "caller that assembled a URL from a base and a suffix cannot see "
            "which of the two it got wrong"
        )
    if not _names_path(answer.body, unrouted.path):
        return Broken(
            f"{unrouted.described} was refused {answer.status} by a body that "
            f"does not quote the path it refused: {answer.body[:200]!r}"
        )
    unnamed = [path for path in _NAMED_BACK if not _names_path(answer.body, path)]
    if unnamed:
        return Broken(
            f"{unrouted.described} was refused {answer.status} by a body naming "
            f"none of {unnamed} as endpoints it does serve: "
            f"{answer.body[:200]!r} — the refusal leaves a caller nowhere to go"
        )
    return Held(
        f"{unrouted.described} was refused {answer.status}, quoting the request "
        f"and naming {list(_NAMED_BACK)} as served"
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
#: The remaining claims belong to `piper-conformance-e16.5` through `e16.7`, and
#: each arrives as a row here rather than as a change to anything below
#: ([LAW:composability]).
CLAIMS: tuple[Claim, ...] = (
    Claim("HEALTH-1", SELF_CONSISTENT, probe_health_1),
    Claim("HEALTH-2", FALSIFIABLE, probe_health_2),
    Claim("HEALTH-3", FALSIFIABLE, probe_health_3),
    Claim("AUTH-1", FALSIFIABLE, probe_auth_1),
    Claim("AUTH-2", FALSIFIABLE, probe_auth_2),
    Claim("DISC-1", FALSIFIABLE, probe_disc_1),
    Claim("DISC-2", FALSIFIABLE, probe_disc_2),
    Claim("DISC-5", FALSIFIABLE, probe_disc_5),
    Claim("MOD-2", SELF_CONSISTENT, probe_mod_2),
    Claim("MOD-3", FALSIFIABLE, probe_mod_3),
    Claim("MOD-4", FALSIFIABLE, probe_mod_4),
    Claim("MOD-5", FALSIFIABLE, probe_mod_5),
    Claim("MOD-6", FALSIFIABLE, probe_mod_6),
    Claim("MOD-7", SELF_CONSISTENT, probe_mod_7),
    Claim("CAP-1", FALSIFIABLE, probe_cap_1),
    Claim("CAP-2", FALSIFIABLE, probe_cap_2),
    Claim("CAP-3", FALSIFIABLE, probe_cap_3),
    Claim("CAP-4", FALSIFIABLE, probe_cap_4),
    Claim("CAP-5", SELF_CONSISTENT, probe_cap_5),
    Claim("CAP-6", FALSIFIABLE, probe_cap_6),
    Claim("CAP-7", FALSIFIABLE, probe_cap_7),
    Claim("CAP-8", FALSIFIABLE, probe_cap_8),
    Claim("CAP-9", FALSIFIABLE, probe_cap_9),
    Claim("CAP-10", FALSIFIABLE, probe_cap_10),
    Claim("SUB-1", FALSIFIABLE, probe_sub_1),
    Claim("SUB-2", FALSIFIABLE, probe_sub_2),
    Claim("SUB-3", SELF_CONSISTENT, probe_sub_3),
    Claim("SUB-4", FALSIFIABLE, probe_sub_4),
    Claim("SUB-5", FALSIFIABLE, probe_sub_5),
    Claim("SUB-6", FALSIFIABLE, probe_sub_6),
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
    Claim("TIME-2", FALSIFIABLE, probe_time_2),
    Claim("TIME-3", FALSIFIABLE, probe_time_3),
    Claim("TIME-4", FALSIFIABLE, probe_time_4),
    Claim("TIME-5", FALSIFIABLE, probe_time_5),
    Claim("TIME-6", FALSIFIABLE, probe_time_6),
    Claim("ROUTE-1", FALSIFIABLE, partial(probe_unrouted, UNROUTED["ROUTE-1"])),
    Claim("ROUTE-2", FALSIFIABLE, partial(probe_unrouted, UNROUTED["ROUTE-2"])),
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
