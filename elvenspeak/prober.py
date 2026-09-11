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
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from email.message import Message
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

#: The read endpoints `AUTH-2` asks without a key. A list rather than a probe of
#: whatever the deployment happens to route, because the claim is about the
#: *documented* surface: these are the four gettable paths README publishes, and
#: a deployment serving fewer of them is answering a different contract. It
#: cannot be derived from `elvenspeak.api` — that is this checkout's route table,
#: not the remote build's ([LAW:one-source-of-truth] reaches only as far as one
#: program).
DOCUMENTED_READS = (
    "/v1/voices",
    "/v1/models",
    "/v1/voices/settings/default",
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

    def json(self, asking: str) -> Any:
        """This body decoded, or [`Blocked`] naming the request that answered it.

        A body that is not JSON blocks the claim rather than breaking it: the
        prober cannot tell a server that answered wrongly from a proxy that
        answered instead of it, and guessing between those is how a prober starts
        reporting on something other than the deployment.
        """
        try:
            return json.loads(self.body)
        except ValueError as unreadable:
            raise Blocked(
                f"{asking} answered a body that is not JSON: {self.body[:200]!r}"
            ) from unreadable


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


def _spoken(target: Target, voice_id: str, key: str | None) -> Reply:
    """[`PROBE_TEXT`] synthesized in `voice_id`, as raw PCM."""
    quoted = urllib.parse.quote(voice_id, safe="")
    return _ask(
        target,
        f"/v1/text-to-speech/{quoted}?output_format={PCM_FORMAT}",
        key,
        method="POST",
        body={"text": PROBE_TEXT},
    )


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


def _parsed_voices(listing: Any) -> tuple[ProbedVoice, ...] | str:
    """`listing`'s voices, or the complaint saying why it published none.

    Returns the complaint rather than raising it, because the same fact means two
    different things to two readers and only the reader knows which. To `DISC-1`,
    whose whole claim is that this endpoint returns `{"voices": [...]}`, a
    malformed listing is the promise *broken*. To a claim that merely needed the
    listing to get at a voice, it is a precondition that failed and the claim goes
    `unasked`. One parser, two readings — which is why the verdict is not decided
    here ([LAW:decomposition]: parsing and judging are two jobs).
    """
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

    def listed(self) -> Reply:
        """`GET /v1/voices`' answer, or [`Blocked`] if it could not be asked.

        A 401 never arrives here: [`discover`] turns the one status meaning "the
        prober could not get in" into the blocker, so every reply this hands back
        is one the deployment really composed and every claim below is free to
        judge it ([LAW:single-enforcer] — which status is a closed door is decided
        in one place).
        """
        if isinstance(self.listing, Blocked):
            raise self.listing
        return self.listing

    @property
    def voices(self) -> tuple[ProbedVoice, ...]:
        """The offered voices, or [`Blocked`] if the listing could not be read.

        Derived on every read rather than cached beside `listing`, which costs a
        parse of a small document and buys the guarantee that the voices a claim
        sees are the ones this deployment actually published.
        """
        reply = self.listed()
        if reply.status != 200:
            raise Blocked(
                f"GET /v1/voices answered {reply.status}: {reply.body[:200]!r}"
            )
        parsed = _parsed_voices(reply.json("GET /v1/voices"))
        if isinstance(parsed, str):
            raise Blocked(f"GET /v1/voices {parsed}")
        return parsed

    def key_or_blocked(self) -> str | None:
        """The key to send a guarded endpoint, or [`Blocked`] if there is none.

        `None` is a complete answer for an open deployment — it is the key that
        works there — so this returns rather than refuses whenever nothing is
        guarded.
        """
        if self.guarded and self.target.key is None:
            raise Blocked(
                "this deployment guards /v1/voices and the prober was given no "
                "--key, so it cannot reach anything behind the guard"
            )
        return self.target.key


def discover(target: Target) -> Deployment:
    """Ask `target` what it is, or raise [`Blocked`] if it never answered.

    The two requests here are the ones a run cannot proceed without, and a
    transport failure in either stops it: a base URL that will not carry
    `/health` and `/v1/voices` says nothing about conformance, and
    `docs/conformance-claims.md` spends exit code `2` on exactly that state. A
    *refusal* is not that — every status either of them can answer is carried
    forward to be judged, and only a dead socket reaches [`main`] as a `2`.
    """
    health = _ask(target, "/health", key=None)
    keyless = _ask(target, "/v1/voices", key=None)
    guarded = keyless.status == 401
    # The keyless answer is reused when nothing is guarded, so an open deployment
    # is asked for its listing once rather than twice.
    try:
        listing: Reply | Blocked = keyless if not guarded else _keyed_listing(target)
    except Blocked as shut_out:
        listing = shut_out
    return Deployment(target=target, health=health, guarded=guarded, listing=listing)


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
    if answer.status == 401:
        raise Blocked(
            "GET /v1/voices refused the xi-api-key the prober was given, so this "
            "deployment's catalogue could not be read — check --key before "
            "reading anything below as a fault of the deployment"
        )
    return answer


# ----------------------------------------------------------------------- probes


def _counted(count: int, noun: str) -> str:
    """`count` of `noun`, pluralised, because a person reads this report."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _health_voices(deployment: Deployment) -> list[Any]:
    """The ids `/health` published, or [`Blocked`] if it published none usably."""
    body = deployment.health.json("GET /health")
    try:
        voices = body["voices"]
    except (KeyError, TypeError) as malformed:
        raise Blocked(
            f"GET /health answered without a `voices` array: {body!r:.200}"
        ) from malformed
    if not isinstance(voices, list):
        raise Blocked(
            f"GET /health published a `voices` that is not an array: {voices!r:.200}"
        )
    return voices


def probe_health_1(deployment: Deployment) -> Verdict:
    """`/health`'s status line and its body agree about whether this can serve.

    Self-consistent, and the document says why that matters: a router misreporting
    the fleet behind it passes this and four claims like it, so the report has to
    show the split rather than fold them into one count.
    """
    voices = _health_voices(deployment)
    status = deployment.health.status
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
    if status not in (200, 503):
        return Broken(
            f"GET /health answered {status}, which is neither the 200 of a server "
            "fit for traffic nor the 503 of one that is not"
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
        reply = _spoken(deployment.target, str(name), key)
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
    if status == 401:
        return Broken(
            "GET /health refused a keyless request 401 — every health checker "
            "this deployment has must then hold a key, and Nomad's does not"
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
            "the prober holds no --key, so a 401 here cannot be told apart from "
            "an endpoint that refuses everyone"
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
        if refused.status != 401:
            return Broken(
                f"GET /v1/voices answered {refused.status} to {described} while "
                "admitting the real one — the guard is not closed"
            )
        detail = refused.json("GET /v1/voices with an unusable key")
        if not isinstance(detail, dict) or detail.get("detail") != INVALID_KEY_DETAIL:
            return Broken(
                f"GET /v1/voices refused {described} with {detail!r:.200} rather "
                f"than a {INVALID_KEY_DETAIL!r} detail — a caller cannot tell this "
                "from any other 401 between it and the server"
            )
    return Held(
        f"401 {INVALID_KEY_DETAIL!r} to no key and to a wrong key, 200 to the real one"
    )


def probe_auth_2(deployment: Deployment) -> Verdict:
    """With nothing configured, every documented endpoint answers without a key.

    Asked of [`DOCUMENTED_READS`] and of one synthesis, because the claim is
    about every endpoint and a guard applied to only the expensive half would
    pass a read-only check completely. One utterance, on one voice, which is what
    that costs.
    """
    if deployment.guarded:
        return Unasked(
            "GET /v1/voices refuses a keyless request, so this deployment has a "
            "key configured and the open arm cannot be asked"
        )
    for path in DOCUMENTED_READS:
        reply = _ask(deployment.target, path, key=None)
        if reply.status == 401:
            return Broken(
                f"GET {path} refused a keyless request 401 while GET /v1/voices "
                "allows one — this deployment is guarded in part, which is "
                "neither of the two states it documents"
            )
    voices = deployment.voices
    if not voices:
        return Unasked(
            f"the {len(DOCUMENTED_READS)} documented reads answered without a "
            "key, but this deployment lists no voice to ask a synthesis of"
        )
    spoken = _spoken(deployment.target, voices[0].id, key=None)
    if spoken.status == 401:
        return Broken(
            "synthesis refused a keyless request 401 while every read allows one "
            f"— {voices[0].id!r} cannot be spoken by a caller this deployment "
            "told it needs no key"
        )
    return Held(
        f"{len(DOCUMENTED_READS)} documented reads and a synthesis in "
        f"{voices[0].id!r} all answered without a key"
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
    parsed = _parsed_voices(reply.json("GET /v1/voices"))
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
        default=None,
        help="the xi-api-key to send; omit for a deployment that configures none",
    )
    parser.add_argument(
        "--timeout",
        type=float,
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
