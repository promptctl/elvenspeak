"""The prober's contract: which claims it asks, and what its verdicts mean.

Two halves, and the split is deliberate. The report format, the verdict
vocabulary and the exit codes are decided by pure functions, so they are asked
here with no server anywhere near them — a contract that needed a deployment to
be readable would be one nobody could check. Everything else drives a *real*
server over a real socket, because the prober speaks HTTP and nothing else and a
transport that shortcut it would be testing a different program
([LAW:behavior-not-structure]).

THE TEST THAT IS THE POINT is [`test_a_voice_missing_its_name_is_broken`]. The
official SDK holds `name` optional, and `piper-conformance-e16.2` proved by
regression that deleting it leaves `voices.get_all()` parsing with no
`ValidationError` at all. A prober built on "did it parse" reports that
deployment green. This suite serves exactly that deployment and requires
`DISC-1` to come back `broken`, which is the one property that separates this
prober from the one the epic exists to avoid building.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import pytest
from conftest import DECLARED_VOICES, published_endpoints
from fastapi import FastAPI, Request, Response
from fleet import cluster, engine_app, router_app, serving
from starlette.exceptions import HTTPException as RouterRefusal

from elvenspeak import prober
from elvenspeak.api import MAX_TEXT_LENGTH
from elvenspeak.engine import Capability
from elvenspeak.formats import DEFAULT_OUTPUT_FORMAT, SUPPORTED_OUTPUT_FORMATS
from elvenspeak.prober import Broken, Claim, Finding, Held, Target, Unasked

#: The document that owns the claim ids and their evidence classes. Read from
#: the repository rather than restated here: this file's whole job in the first
#: three tests is to be the thing that notices the two disagreeing, and a copy of
#: the answer could not ([LAW:one-source-of-truth]).
CLAIMS_DOC = Path(__file__).parent.parent / "docs" / "conformance-claims.md"

#: The issues whose claims `elvenspeak.prober` is expected to ask today. Named
#: because the document's own "Probed by" column is what decides the set, so the
#: day `piper-conformance-e16.5` lands its claims, this tuple grows by editing the
#: document and the prober — never this test.
LANDED_ISSUES = ("e16.3", "e16.4", "e16.5", "e16.enf", "e16.6")

#: A row of any claim table in the document: the id in backticks, then the cells.
_CLAIM_ROW = re.compile(r"^\|\s*`([A-Z]+-\d+)`\s*\|(.*)\|\s*$", re.M)


def documented() -> dict[str, tuple[str, str]]:
    """Every published claim, as `id -> (evidence class, who probes it)`.

    The evidence cell is taken down to its leading word because the document
    qualifies several of them in prose — "falsifiable (only when the prober holds
    a key)" is the falsifiable class with a precondition noted beside it, and the
    class is what the report splits on.
    """
    text = CLAIMS_DOC.read_text(encoding="utf-8")
    claims = {}
    for row in _CLAIM_ROW.finditer(text):
        cells = [cell.strip() for cell in row.group(2).split("|")]
        evidence, probed_by = cells[-2], cells[-1]
        claims[row.group(1)] = (evidence.split("(")[0].split()[0], probed_by)
    return claims


# ----------------------------------------- the table, held equal to the document


def test_the_document_still_publishes_a_table_this_test_can_read() -> None:
    """The three tests below are worthless if the document stopped parsing.

    A regex over prose is a map that can go stale silently: a reformatted table
    yields zero rows, and "every registered claim is documented" then passes
    vacuously against nothing at all. This is the check that cannot be satisfied
    by the document disappearing ([LAW:no-silent-failure]).
    """
    claims = documented()

    assert len(claims) > 50, (
        f"{CLAIMS_DOC.name} yielded {len(claims)} claims, and it publishes "
        "fifty-seven — the table's shape has changed and the checks below are "
        "now reading nothing"
    )


def test_every_claim_probed_here_is_one_the_document_publishes() -> None:
    """No id reaches a report without a promise behind it."""
    published = documented()

    for claim in prober.CLAIMS:
        assert claim.id in published, (
            f"{claim.id} is probed but appears in no table in {CLAIMS_DOC.name} — "
            "a report naming it could not be traced back to a documented promise"
        )


def test_every_claim_the_document_assigns_a_landed_issue_is_probed() -> None:
    """The prober asks everything it was made responsible for, and says so."""
    owed = {
        claim_id
        for claim_id, (_, probed_by) in documented().items()
        if any(issue in probed_by for issue in LANDED_ISSUES)
    }
    asked = {claim.id for claim in prober.CLAIMS}

    assert owed - asked == set(), (
        f"{CLAIMS_DOC.name} says {sorted(owed - asked)} are probed by "
        f"{list(LANDED_ISSUES)} and nothing here asks them"
    )
    assert asked - owed == set(), (
        f"{sorted(asked - owed)} are probed here, and {CLAIMS_DOC.name} assigns "
        f"them to some other issue"
    )


def test_each_claim_carries_the_evidence_class_the_document_assigns() -> None:
    """The split a report prints is the document's split, not this file's.

    The distinction is load-bearing rather than decorative: the document records
    that a router misreporting the fleet behind it passes every self-consistent
    claim there is, so a claim mislabelled `falsifiable` would hide exactly that
    failure inside a green run.
    """
    published = documented()

    for claim in prober.CLAIMS:
        evidence, _ = published[claim.id]
        assert claim.evidence == evidence, (
            f"{claim.id} is probed as {claim.evidence!r} and documented as "
            f"{evidence!r}"
        )


# --------------------------------------------- the vocabulary and the exit codes


def test_there_are_exactly_three_verdicts_and_none_of_them_is_skipped() -> None:
    """The vocabulary the document settled, asserted rather than assumed.

    `skipped` is refused by name here because its absence is the whole design: a
    word that lets "I did not find out" render like "I found out and it was fine"
    is what this prober exists not to have.
    """
    words = {Held.word, Broken.word, Unasked.word}

    assert words == {"held", "broken", "unasked"}
    assert "skipped" not in words


@pytest.mark.parametrize(
    ("verdicts", "expected"),
    [
        ((Held("fine"),), 0),
        ((Held("fine"), Held("also fine")), 0),
        ((Broken("no"),), 1),
        ((Unasked("could not"),), 3),
        # Precedence: broken outranks unasked because it is the stronger finding,
        # and a run carrying both must report the deployment is wrong rather than
        # that the prober could not find out.
        ((Unasked("could not"), Broken("no")), 1),
        ((Held("fine"), Unasked("could not")), 3),
        ((Held("fine"), Broken("no"), Unasked("could not")), 1),
    ],
)
def test_the_exit_code_reports_the_strongest_finding(
    verdicts: tuple[Any, ...], expected: int
) -> None:
    """`2` is absent on purpose: it is not a verdict about any deployment."""
    findings = tuple(Finding(prober.CLAIMS[0], verdict) for verdict in verdicts)

    assert prober.exit_code(findings) == expected


def test_a_run_that_asked_nothing_cannot_report_a_verdict() -> None:
    """[LAW:no-silent-failure] `0` would read as "everything held"."""
    with pytest.raises(AssertionError):
        prober.exit_code(())


def _report(*verdicts: Any) -> prober.Report:
    """A report over the real claim table, one verdict per claim."""
    return prober.Report(
        target=Target("http://example", None, 1.0),
        findings=tuple(
            Finding(claim, verdict) for claim, verdict in zip(prober.CLAIMS, verdicts)
        ),
    )


def test_the_report_names_every_claim_even_when_every_one_of_them_held() -> None:
    """Coverage is the number that says whether a green run meant anything.

    A report printing only failures reads identically whether fifty claims were
    asked or two, which is the shape of a prober nobody can audit.
    """
    rendered = prober.rendered(_report(*(Held("held") for _ in prober.CLAIMS)))

    for claim in prober.CLAIMS:
        assert claim.id in rendered, f"{claim.id} held and was left out of the report"


def test_the_report_separates_falsifiable_claims_from_self_consistent_ones() -> None:
    """The split the document requires, because one count would hide the router.

    Five of this service's claims can only be checked against its own account of
    itself, and a router misreporting the fleet behind it passes every one. A
    report folding them into a single total is where that whole question goes
    missing.
    """
    rendered = prober.rendered(_report(*(Held("held") for _ in prober.CLAIMS)))

    assert prober.FALSIFIABLE in rendered
    assert prober.SELF_CONSISTENT in rendered
    falsifiable = sum(1 for c in prober.CLAIMS if c.evidence == prober.FALSIFIABLE)
    assert f"{falsifiable} {prober.FALSIFIABLE}:" in rendered


def test_a_broken_verdict_carries_its_reason_into_the_report() -> None:
    """A verdict a reader cannot act on is a verdict that did not report."""
    rendered = prober.rendered(
        _report(
            Broken("the listing came back empty"),
            *(Held("h") for _ in prober.CLAIMS[1:]),
        )
    )

    assert "the listing came back empty" in rendered


def test_a_probe_raising_anything_but_blocked_is_not_swallowed() -> None:
    """[LAW:no-silent-failure] A bug in the prober must not render as `unasked`.

    `unasked` means the deployment could not be asked. A prober reporting its own
    defects in that word would be telling precisely the lie it exists to catch.
    """

    def broken_probe(_: prober.Deployment) -> prober.Verdict:
        raise ZeroDivisionError("a bug in a probe")

    claim = Claim("HEALTH-1", prober.FALSIFIABLE, broken_probe)

    with pytest.raises(ZeroDivisionError):
        prober._finding(claim, None)  # type: ignore[arg-type]


# ----------------------------------------------- a real server on a real socket


def _verdicts(base_url: str, key: str | None = None) -> dict[str, prober.Verdict]:
    """Every claim asked of a running deployment, keyed by claim id."""
    report = prober.probe(Target(base_url=base_url, key=key, timeout=30.0))
    return {finding.claim.id: finding.verdict for finding in report.findings}


def _words(base_url: str, key: str | None = None) -> dict[str, str]:
    """The same, reduced to the verdict word each claim came back with."""
    return {name: verdict.word for name, verdict in _verdicts(base_url, key).items()}


def test_an_open_deployment_breaks_nothing() -> None:
    """The control. Every claim that can be asked of an open server holds."""
    with serving(engine_app("piper", DECLARED_VOICES)) as base_url:
        words = _words(base_url)

    assert words["HEALTH-1"] == "held"
    assert words["HEALTH-2"] == "held"
    assert words["DISC-1"] == "held"
    assert words["AUTH-2"] == "held"
    assert "broken" not in words.values()


def test_an_open_deployment_leaves_the_guarded_arms_unasked() -> None:
    """Two-armed claims report the arm they got rather than passing vacuously.

    `AUTH-1` and `HEALTH-3` are both about a deployment that configures a key.
    Against one that does not, they are unasked — never `held`, which is what a
    claim with no subject would be worth.
    """
    with serving(engine_app("piper", DECLARED_VOICES)) as base_url:
        words = _words(base_url)

    assert words["AUTH-1"] == "unasked"
    assert words["HEALTH-3"] == "unasked"


def test_a_guarded_deployment_is_fully_probed_when_the_key_is_held() -> None:
    """The other arm: with a key, the authentication claims are really asked."""
    with serving(engine_app("piper", DECLARED_VOICES, api_key="probe-key")) as base_url:
        words = _words(base_url, key="probe-key")

    assert words["AUTH-1"] == "held"
    assert words["HEALTH-3"] == "held"
    assert words["DISC-1"] == "held"
    assert words["HEALTH-2"] == "held"
    # The open arm is the one with no subject now, and says so rather than
    # holding — which is the same rule read from the other side.
    assert words["AUTH-2"] == "unasked"


def test_a_guarded_deployment_probed_without_a_key_breaks_nothing() -> None:
    """Not holding the key is the prober's problem, never the deployment's.

    Every claim behind the guard comes back `unasked` carrying why, and nothing
    is reported `broken`: a server correctly refusing a caller with no key is
    keeping its promise, and a prober calling that a defect would be blaming a
    deployment for its own invocation.
    """
    with serving(engine_app("piper", DECLARED_VOICES, api_key="probe-key")) as base_url:
        verdicts = _verdicts(base_url, key=None)

    words = {name: verdict.word for name, verdict in verdicts.items()}
    assert "broken" not in words.values(), words
    assert words["DISC-1"] == "unasked"
    assert words["HEALTH-2"] == "unasked"
    # /health is outside the guard, so the one claim that needs no key is asked.
    assert words["HEALTH-3"] == "held"
    assert "--key" in verdicts["DISC-1"].blocker


def test_a_wrong_key_is_reported_as_the_probers_own_problem() -> None:
    """A refused key leaves the control unestablished, so nothing is broken.

    From outside, "you gave me the wrong key" and "this endpoint refuses
    everyone" are the same observation. Reporting `broken` would blame the
    deployment for the likelier of the two, which is how a prober earns a
    reputation for crying wolf and stops being run.
    """
    with serving(engine_app("piper", DECLARED_VOICES, api_key="probe-key")) as base_url:
        verdicts = _verdicts(base_url, key="not-the-key")

    words = {name: verdict.word for name, verdict in verdicts.items()}
    assert "broken" not in words.values(), words
    assert words["AUTH-1"] == "unasked"
    assert "--key" in verdicts["AUTH-1"].blocker


def test_an_unreachable_url_exits_two_rather_than_reporting_on_a_deployment() -> None:
    """A base URL that never answered says nothing about conformance.

    Port 1 is reserved and nothing binds it, so this is a refusal rather than a
    wait. `2` is the code this project already spends on a fault that stops a
    process before it does its work.
    """
    assert prober.main(["http://127.0.0.1:1", "--timeout", "2"]) == 2


@pytest.mark.parametrize("given", ["-1", "0", "nan", "inf", "soon"])
def test_a_timeout_no_socket_can_take_is_a_usage_error_and_not_a_verdict(
    given: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """An operator's typo is the operator's, and `1` would say it was the server's.

    `--timeout -1` reached `socket.settimeout` inside `urlopen` and raised
    `ValueError`, which is neither of the two exceptions `_ask` turns into a
    blocker — so it propagated past `main` and the shell saw `1`, this project's
    code for *at least one claim is broken*. A deployment nobody probed was being
    reported as non-conformant. `2` is the code for a run that could not start,
    which is what a refused invocation is.
    """
    with pytest.raises(SystemExit) as refused:
        prober.main(["http://127.0.0.1:1", "--timeout", given])

    assert refused.value.code == 2, f"--timeout {given} exited {refused.value.code}"
    assert "--timeout" in capsys.readouterr().err


@pytest.mark.parametrize(
    "given", ["key\n", "key\r", "key\twith-tab", "k\U0001f511"]
)
def test_a_key_no_header_can_carry_is_a_usage_error_and_not_a_verdict(
    given: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same hole as `--timeout`, through the other word an operator gives.

    A key carrying CR or LF reaches `http.client.putheader` as a `ValueError`
    and one outside latin-1 as a `UnicodeEncodeError`; `_ask` catches neither,
    so both propagated past `main` and the shell saw `1` — this project's code
    for *at least one claim is broken*, spent on a deployment nobody probed. A
    trailing newline off a key file is the ordinary way in, which is why this is
    a usage error and not a rare one.
    """
    with pytest.raises(SystemExit) as refused:
        prober.main(["http://127.0.0.1:1", "--key", given, "--timeout", "2"])

    code = refused.value.code
    assert code == 2, f"--key {given!r} exited {code}"
    assert "--key" in capsys.readouterr().err


# ------------------------------------- the deployments that lie past the parse


#: What `/health` says when nothing is being spoiled: the one id [`WELL_FORMED`]
#: publishes, so the default fixture is a deployment whose two endpoints agree.
HEALTHY = {"voices": ["v1"]}


#: How long the stand-in's utterance is. Arbitrary and constant: what the format
#: claims read is that every rate states the *same* duration, so the number
#: matters only in being one number.
UTTERANCE_SECONDS = 0.1


def _family(wire_name: str) -> tuple[str, int]:
    """`wire_name` split into its codec and rate.

    Written here rather than taken from `prober._published`, for the reason
    [`WELL_FORMED`] is written out: a stand-in that read the format name the same
    way the prober does could not catch the prober reading it wrongly.
    """
    codec, rate = wire_name.split("_")[:2]
    return codec, int(rate)


def _wav(rate: int, samples: int) -> bytes:
    """A RIFF/WAVE file stating `rate`, carrying `samples` of silence.

    Built by hand so the rate in the header is a value a test sets, which is the
    whole of what `FMT-5` reads. A `JUNK` chunk sits before `fmt ` on purpose: a
    prober reading the rate at the canonical byte 24 gets the padding instead,
    and a deployment writing a perfectly legal file is reported broken.
    """
    audio = b"\0\0" * samples
    fmt = (
        b"\x01\x00\x01\x00"
        + rate.to_bytes(4, "little")
        + (rate * 2).to_bytes(4, "little")
        + b"\x02\x00\x10\x00"
    )
    chunks = (
        b"JUNK" + (4).to_bytes(4, "little") + bytes(4)
        + b"fmt " + len(fmt).to_bytes(4, "little") + fmt
        + b"data" + len(audio).to_bytes(4, "little") + audio
    )
    return b"RIFF" + (len(chunks) + 4).to_bytes(4, "little") + b"WAVE" + chunks


def _audio(wire_name: str) -> tuple[bytes, str]:
    """A conformant answer for `wire_name`: its real signature, at its real rate."""
    codec, rate = _family(wire_name)
    samples = round(rate * UTTERANCE_SECONDS)
    bodies: dict[str, tuple[bytes, str]] = {
        # Every `mp3_*` and `opus_*` answer is the same bytes, which is what a
        # deployment whose default really is `mp3_44100_128` looks like to
        # `FMT-7` — and the rate inside them is not readable without a decoder,
        # which is why no claim asks for it.
        "mp3": (b"\xff\xfb" + bytes(64), "audio/mpeg"),
        "opus": (b"OggS" + bytes(64), "audio/ogg"),
        "wav": (_wav(rate, samples), "audio/wav"),
        "pcm": (b"\0\0" * samples, "audio/pcm"),
        "ulaw": (bytes(samples), "audio/basic"),
        "alaw": (b"\x55" * samples, "audio/x-alaw-basic"),
    }
    return bodies[codec]


def _refusal(detail: Any) -> Response:
    """A 422 carrying `detail`, and deliberately no `x-elvenspeak-*` header."""
    return Response(
        content=json.dumps({"detail": detail}).encode(),
        status_code=422,
        media_type="application/json",
    )


def _refused(output_format: str | None, body: dict[str, Any]) -> Response | None:
    """The 422 this request has earned, or `None` if it has earned none.

    **Both refusal shapes, on purpose.** The unknown format is refused as an
    object under `detail.message`, and everything pydantic owns as an array of
    error records under `detail[].input` — which is what this service really
    sends, and the pair `piper-conformance-e16.4` decided the prober must accept
    without requiring either. A stand-in refusing in one shape could not tell a
    prober that accepts both from one that happens to demand the shape it sends.
    """
    text = body.get("text")
    if output_format is not None and output_format not in SUPPORTED_OUTPUT_FORMATS:
        return _refusal(
            {
                "message": f"unsupported output_format: {output_format!r}",
                "supported": list(SUPPORTED_OUTPUT_FORMATS),
            }
        )
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_LENGTH:
        return _refusal(
            [{"type": "value_error", "loc": ["body", "text"], "input": text}]
        )
    if not isinstance(body.get("language_code", ""), str):
        return _refusal(
            [
                {
                    "type": "string_type",
                    "loc": ["body", "language_code"],
                    "input": body["language_code"],
                }
            ]
        )
    return None


#: The body fields this stand-in models. Everything else is kept and named back,
#: which is `REF-7`.
_MODELLED = frozenset({"text", "language_code"})


def conformant(output_format: str | None, body: dict[str, Any]) -> Response:
    """Every format and refusal claim answered honestly.

    The control the cases below spoil one at a time, and the reason they are
    readable: without a stand-in that holds all fifteen, a fixture broken in some
    unrelated way would turn them red for the wrong reason and read as proof the
    prober works.
    """
    refused = _refused(output_format, body)
    if refused is not None:
        return refused
    audio, content_type = _audio(output_format or DEFAULT_OUTPUT_FORMAT)
    headers = {"x-elvenspeak-voice": str(WELL_FORMED["voice_id"])}
    unmodelled = sorted(set(body) - _MODELLED)
    if unmodelled:
        headers["x-elvenspeak-ignored"] = ", ".join(unmodelled)
    return Response(content=audio, media_type=content_type, headers=headers)


#: One deployment's answer to a synthesis: the `output_format` asked for — `None`
#: where the request named none — and the body it carried.
Speaks = Callable[[str | None, dict[str, Any]], Response]


def spoiling(
    spoil: Callable[[str | None, bytes, str], tuple[bytes, str]],
) -> Speaks:
    """[`conformant`], with `spoil` free to rewrite the audio it would have sent.

    [LAW:dataflow-not-control-flow] Every deployment that lies about a format is
    this one function carrying a different value, rather than a fixture apiece:
    what separates "answers WAV to every request" from "states the wrong rate in
    the header" is two lines of arithmetic, and a second `FastAPI` app around
    each would bury that.

    Refusals pass through untouched, so a case spoiling the audio cannot
    accidentally be testing the refusal claims as well.
    """

    def speaks(output_format: str | None, body: dict[str, Any]) -> Response:
        answered = conformant(output_format, body)
        if answered.status_code != 200:
            return answered
        audio, content_type = spoil(
            output_format, answered.body, answered.headers["content-type"]
        )
        return Response(
            content=audio,
            media_type=content_type,
            headers={"x-elvenspeak-voice": str(WELL_FORMED["voice_id"])},
        )

    return speaks


def unrouted(method: str, path: str, status: int) -> Response:
    """A request nothing routes, answered the way README:36 says it should be.

    Written out rather than reached for in `elvenspeak.api`, for the reason
    [`WELL_FORMED`] is: a fixture borrowing this checkout's refusal would agree
    with the prober whatever either of them came to mean, and `ROUTE-1` and
    `ROUTE-2` would then be two claims that cannot fail.

    The wording is deliberately not this service's. What the claims require is
    that the refusal *names* the method, the path and what is served — never that
    it says so in these words or under these keys.
    """
    return Response(
        content=json.dumps(
            {
                "detail": {
                    "message": f"{method} {path} is not answered by this server",
                    "served": [
                        "GET /health",
                        "GET /v1/voices",
                        "GET /v1/models",
                        "GET /v1/voices/settings/default",
                        "POST /v1/text-to-speech/{voice_id}",
                    ],
                }
            }
        ).encode(),
        status_code=status,
        media_type="application/json",
    )


Unrouted = Callable[[str, str, int], Response]


def lying_deployment(
    voices: list[dict[str, Any]],
    health_body: Any = HEALTHY,
    health_status: int = 200,
    speaks: Speaks = conformant,
    refuses: Unrouted = unrouted,
) -> FastAPI:
    """A server answering whatever it is told to, however untrue.

    Hand-built rather than `create_app` with a doctored engine, because what has
    to be expressible here is a response this service *cannot* produce. A stand-in
    that could only emit well-formed answers would make every test below pass by
    construction ([LAW:verifiable-goals] — the failure needs a shape it can take).

    `/health` is handed its whole body rather than a list of ids, because
    `HEALTH-1`'s claim is that body's shape: a payload with no `voices` key, or
    one whose entries are objects, is a state the claim is *about* and one an id
    list cannot express.

    `speaks` is the same seam for the synthesis endpoint, and its default is
    [`conformant`] rather than a constant blob: the fifteen format and refusal
    claims each need a deployment that keeps every *other* promise, so the
    truthful answer is the default and each case below buys exactly one lie.

    `refuses` is that seam again for the two fall-throughs. It has to exist
    because this stand-in deliberately routes four paths out of nine, so nearly
    every request the prober makes of it arrives here — and a stand-in whose
    fall-through were Starlette's own would report a deployment broken on
    `ROUTE-1` in every test in this file that asks for a *different* failure.
    """
    app = FastAPI()

    @app.exception_handler(RouterRefusal)
    async def fell_through(request: Request, refusal: RouterRefusal) -> Response:
        return refuses(request.method, request.url.path, refusal.status_code)

    @app.get("/health")
    def health(response: Response) -> Any:
        response.status_code = health_status
        return health_body

    @app.get("/v1/voices")
    def listing() -> dict:
        return {"voices": voices}

    @app.get("/v1/models")
    def models() -> list:
        return [{"model_id": "piper"}]

    @app.get("/v1/voices/settings/default")
    def defaults() -> dict:
        return {"stability": 0.5}

    @app.post("/v1/text-to-speech/{voice_id}")
    def speak(
        voice_id: str, body: dict[str, Any], output_format: str | None = None
    ) -> Response:
        return speaks(output_format, body)

    return app


#: A voice carrying everything `DISC-1` requires, which each case below spoils in
#: exactly one way. Written out rather than derived from `api._voice_json`: the
#: prober judges deployments it did not build, and a fixture agreeing with this
#: checkout's renderer by construction could not catch a renderer that changed.
WELL_FORMED = {
    "voice_id": "v1",
    "name": "Vee",
    "category": "premade",
    "aliases": [],
    "capabilities": ["speed"],
    "models": ["piper"],
    "language": "en",
}


def test_the_lying_deployment_fixture_passes_when_it_tells_the_truth() -> None:
    """The control for every case below, so a spoiled field is what differs.

    Without this, a fixture broken in some unrelated way would make all of them
    red for the wrong reason and read as proof the prober works.
    """
    with serving(lying_deployment([WELL_FORMED])) as base_url:
        words = _words(base_url)

    assert words["DISC-1"] == "held"
    assert words["HEALTH-1"] == "held"
    assert words["HEALTH-2"] == "held"
    # Neither per-voice read nor any streaming synthesis is routed here, so they
    # answer 404 — and `AUTH-2` holds anyway, because a path a build does not
    # serve is a different promise than the keylessness being asked.
    assert words["AUTH-2"] == "held"


def test_a_voice_missing_its_name_is_broken() -> None:
    """The test this whole file exists for.

    `piper-conformance-e16.2` verified by regression that deleting `name` from
    the voice payload leaves the official SDK parsing it cleanly, with no
    `ValidationError` raised at all — `Voice` requires only `voice_id` and holds
    twenty-five fields optional. So a prober whose verdict rested on a successful
    parse calls this deployment conformant, and a real client renders a voice
    picker full of blanks.
    """
    spoiled = {key: value for key, value in WELL_FORMED.items() if key != "name"}

    with serving(lying_deployment([spoiled])) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["DISC-1"].word == "broken"
    assert "name" in verdicts["DISC-1"].why


def test_a_voice_carrying_only_its_id_is_broken() -> None:
    """`{"voice_id": "x"}` satisfies every response model the SDK has.

    The document names this exact body as the one a lenient prober waves through,
    and it is the floor the ticket forbids a `held` verdict from resting on.
    """
    with serving(lying_deployment([{"voice_id": "v1"}])) as base_url:
        assert _words(base_url)["DISC-1"] == "broken"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        # Empty rather than absent: the SDK parses both, and a client can render
        # neither. A check testing only for the key's presence would miss this.
        ("name", ""),
        ("category", ""),
        ("language", ""),
        # Right key, wrong JSON type. `models` as a bare string is iterable and
        # truthy, so a check that asked only "is it there and non-empty" passes.
        ("models", "piper"),
        ("capabilities", "speed"),
        ("aliases", {}),
    ],
)
def test_a_field_of_the_wrong_shape_is_broken(field: str, value: Any) -> None:
    """Every field `DISC-1` names is judged for type and emptiness, not presence."""
    with serving(lying_deployment([{**WELL_FORMED, field: value}])) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["DISC-1"].word == "broken", f"{field}={value!r} was let through"
    assert field in verdicts["DISC-1"].why


@pytest.mark.parametrize(
    "missing",
    ["voice_id", "name", "category", "aliases", "capabilities", "models", "language"],
)
def test_every_promised_field_is_required(missing: str) -> None:
    """Dropping any one of the seven is caught, including the four extensions.

    `aliases`, `capabilities`, `models` and `language` are not ElevenLabs fields
    at all, so no SDK model would ever miss them — `elvenspeak.remote` reads
    three of them to route, which makes a router the first thing a silent drop
    breaks.
    """
    spoiled = {key: value for key, value in WELL_FORMED.items() if key != missing}

    with serving(lying_deployment([spoiled])) as base_url:
        assert _words(base_url)["DISC-1"] == "broken", f"a voice with no {missing}"


def test_health_offering_a_voice_the_listing_does_not_have_is_broken() -> None:
    """The fleet-shaped failure: a checker sends traffic for an unaddressable id."""
    body = {"voices": ["v1", "ghost"]}
    with serving(lying_deployment([WELL_FORMED], health_body=body)) as url:
        verdicts = _verdicts(url)

    assert verdicts["HEALTH-2"].word == "broken"
    assert "ghost" in verdicts["HEALTH-2"].why


def test_health_answering_200_with_no_voices_is_broken() -> None:
    """The 2026-09-02 incident, asked as a claim.

    A router that discovered no engines registered in Consul as passing and
    served silence. The status line and the body have to agree, and this is the
    direction that puts a dead deployment into rotation.
    """
    empty = {"voices": []}
    with serving(lying_deployment([WELL_FORMED], health_body=empty)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["HEALTH-1"].word == "broken"
    # And the claim that would have had to speak those voices says it could not,
    # rather than passing because there was nothing to check.
    assert verdicts["HEALTH-2"].word == "unasked"


def test_health_answering_503_while_offering_voices_is_broken() -> None:
    """The other direction: a server withdrawn from rotation that can still speak."""
    with serving(lying_deployment([WELL_FORMED], health_status=503)) as base_url:
        assert _words(base_url)["HEALTH-1"] == "broken"


@pytest.mark.parametrize("status", [404, 500])
def test_health_answering_neither_200_nor_503_is_broken(status: int) -> None:
    """`HEALTH-1` names two statuses, so a third is the claim failing.

    A build that never routed `/health`, and one whose health handler raised, are
    both deployments a checker cannot read — and either would have read `held`
    from a prober that only compared the body against the status it happened to
    get.
    """
    with serving(lying_deployment([WELL_FORMED], health_status=status)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["HEALTH-1"].word == "broken"
    assert str(status) in verdicts["HEALTH-1"].why


def test_a_health_body_with_no_voices_array_is_broken_and_never_unasked() -> None:
    """The shape of `/health`'s body is `HEALTH-1`'s subject, not its precondition.

    `{"status": "ok"}` is this exact promise broken: a checker reading it learns
    nothing about which voices loaded. Reporting `unasked` would file the claim's
    own failure under "I could not find out" — the one confusion the third
    verdict exists to prevent, arriving through the claim it most applies to.
    `HEALTH-2` really is only blocked by it, and says so.
    """
    ok = {"status": "ok"}
    with serving(lying_deployment([WELL_FORMED], health_body=ok)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["HEALTH-1"].word == "broken"
    assert "voices" in verdicts["HEALTH-1"].why
    assert verdicts["HEALTH-2"].word == "unasked"


def test_a_health_voice_that_is_not_a_string_does_not_end_the_run() -> None:
    """An unaddressable id is a malformed body, and it used to end the run.

    `{"voices": [{"id": "v1"}]}` reached `HEALTH-2`'s set lookup and raised
    `TypeError: unhashable type: 'dict'` out of `probe`, so every claim died —
    including the ones with nothing to do with `/health`. A prober that cannot
    report on a deployment that lies in an unanticipated way is a prober whose
    green runs mean less than they look like.
    """
    objects = {"voices": [{"id": "v1"}]}
    with serving(lying_deployment([WELL_FORMED], health_body=objects)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["HEALTH-1"].word == "broken"
    assert verdicts["HEALTH-2"].word == "unasked"
    # The claims that never read `/health` are still asked and still answer.
    assert verdicts["DISC-1"].word == "held"
    assert verdicts["AUTH-2"].word == "held"


def test_a_listing_that_answers_an_error_breaks_the_claim_about_the_listing() -> None:
    """A `500` is `DISC-1` broken, and only a precondition to everything else.

    The same answer means two different things to two readers, and reporting it
    as `unasked` everywhere would lose the finding: `DISC-1` is the promise that
    this endpoint returns a listing, so it is the claim that failed.
    """
    app = FastAPI()

    @app.get("/health")
    def health() -> dict:
        return {"voices": ["v1"]}

    @app.get("/v1/voices")
    def listing() -> Response:
        return Response(content=b"boom", status_code=500)

    with serving(app) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["DISC-1"].word == "broken"
    assert "500" in verdicts["DISC-1"].why
    assert verdicts["HEALTH-2"].word == "unasked"


def guarded_deployment(
    refuses: Callable[[str, str | None], bool],
    status: int = 401,
    detail: str = "invalid xi-api-key",
) -> FastAPI:
    """[`lying_deployment`] behind whatever guard `refuses` describes.

    The states this service cannot be made to produce, and which every
    authentication claim exists to catch: a key covering part of the surface, a
    guard that admits the wrong key, a refusal a caller cannot tell from a
    proxy's. The guard is a predicate over the path and the `xi-api-key` sent
    rather than a flag per state, so a new state is a value a test passes and not
    another fixture ([LAW:dataflow-not-control-flow]).

    A middleware rather than a route, because the paths worth guarding include
    the two per-voice reads this fixture deliberately does not serve — refusing
    before the router is what lets a path be guarded without the fixture
    pretending to implement it.

    `detail` is written out rather than read from `prober.INVALID_KEY_DETAIL`,
    for the reason [`WELL_FORMED`] is: a fixture agreeing with the prober by
    construction cannot catch the prober changing what it demands.
    """
    app = lying_deployment([WELL_FORMED])

    @app.middleware("http")
    async def guard(request: Request, call_next: Any) -> Response:
        if refuses(request.url.path, request.headers.get("xi-api-key")):
            return Response(
                content=json.dumps({"detail": detail}).encode(),
                status_code=status,
                media_type="application/json",
            )
        return await call_next(request)

    return app


def shutting(path: str, status: int = 401) -> FastAPI:
    """[`guarded_deployment`] refusing `path` alone, whatever key is sent."""
    return guarded_deployment(lambda asked, _: asked == path, status=status)


@pytest.mark.parametrize(
    "shut",
    [
        read
        for read in prober._addressed(prober.DOCUMENTED_READS, "v1")
        if read != "/v1/voices"
    ],
)
def test_a_read_guarded_while_the_listing_is_open_breaks_auth_2(shut: str) -> None:
    """Every documented read is really asked, one guard at a time.

    `AUTH-2` promises that a deployment configuring no key answers *every*
    documented endpoint without one, so a path the prober never asks turns a
    partial guard into a `held` verdict nothing earned — and the per-voice reads
    were exactly that gap. Shutting each read in turn is what holds the probed
    set equal to the documented one, including the reads added after this test.
    """
    with serving(shutting(shut)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["AUTH-2"].word == "broken", f"{shut} is never asked keyless"
    assert f"GET {shut}" in verdicts["AUTH-2"].why


@pytest.mark.parametrize("shut", prober._addressed(prober.DOCUMENTED_SYNTHESES, "v1"))
def test_a_synthesis_guarded_while_every_read_is_open_breaks_auth_2(shut: str) -> None:
    """The expensive half guarded alone, which a read-only check passes whole.

    A deployment can plausibly arrive in this state by guarding what costs it CPU
    and leaving the listings open, and every read `AUTH-2` makes would answer.
    Each postable path is shut in turn because a deployment gating only the
    streaming ones is the same failure one endpoint over: the plain synthesis
    answers, and a prober asking nothing else calls that deployment open.
    """
    with serving(shutting(shut)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["AUTH-2"].word == "broken", f"{shut} is never asked keyless"
    assert f"POST {shut}" in verdicts["AUTH-2"].why


@pytest.mark.parametrize("status", [403, 500])
def test_a_read_answering_any_other_status_keyless_breaks_auth_2(status: int) -> None:
    """A keyless caller got no answer, whatever status said so.

    `AUTH-2` promises every documented endpoint answers without a key, and only a
    404 is exempt — a path this build does not route is a different promise. A
    prober judging 401 alone reads a 403 from a proxy in front of the deployment,
    or a 500 from the deployment itself, as "all answered without a key", which is
    false on its face. The status is named so a guard and a fault still read
    differently in the report.
    """
    with serving(shutting("/v1/models", status=status)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["AUTH-2"].word == "broken", f"{status} was read as an answer"
    assert "GET /v1/models" in verdicts["AUTH-2"].why
    assert str(status) in verdicts["AUTH-2"].why


@pytest.mark.parametrize("refusal", [401, 403])
def test_health_refusing_a_keyless_caller_behind_a_guard_is_broken(
    refusal: int,
) -> None:
    """`HEALTH-3` is the claim that `/health` is outside the guard.

    Nomad's health checker holds no key, so a deployment that put `/health`
    behind its own guard is one every checker reads as down — which is the
    failure this claim is for, and a deployment this service cannot produce.

    Both refusals, written out rather than read from `prober.AUTH_REFUSALS`: a
    case list derived from the tuple it polices shrinks silently when the tuple
    does, which is how the per-read gap survived three rounds green.
    """
    guarded = guarded_deployment(
        lambda path, _: path in ("/v1/voices", "/health"), status=refusal
    )
    with serving(guarded) as base_url:
        verdicts = _verdicts(base_url, key="probe-key")

    assert verdicts["HEALTH-3"].word == "broken"
    assert str(refusal) in verdicts["HEALTH-3"].why


@pytest.mark.parametrize("refusal", [401, 403])
def test_a_listing_refusing_a_keyless_caller_is_a_guard_whichever_status(
    refusal: int,
) -> None:
    """Guarded is decided by what a refusal *is*, not by this service's own 401.

    A deployment behind a gateway that answers 403 read as unguarded, and three
    claims then reported something other than authentication: `AUTH-1` and
    `HEALTH-3` said "nothing is guarded here" of a deployment that is, and
    `DISC-1` called the refusal a listing defect. The guard is the precondition
    all three turn on, so one wrong reading of it is three wrong answers, none of
    them naming the key.
    """
    guarded = guarded_deployment(
        lambda path, sent: path == "/v1/voices" and sent != "probe-key",
        status=refusal,
    )
    with serving(guarded) as base_url:
        words = _words(base_url, key="probe-key")

    assert words["HEALTH-3"] == "held"
    assert words["DISC-1"] == "held"
    assert words["HEALTH-2"] == "held"
    # The open arm has no subject against a guarded deployment, and says so.
    assert words["AUTH-2"] == "unasked"


def test_a_guard_refusing_403_breaks_auth_1_rather_than_being_excused() -> None:
    """The refusal `AUTH-1` promises is a 401, so a 403 is that promise broken.

    Naming it is the point: the deployment really is guarded, and it guards with
    a status the documented contract does not offer, which is a conformance
    failure a caller meets as a refusal it cannot tell from a proxy's. A
    carve-out letting 403 pass here would trade a wrong `unasked` for a wrong
    `held`.
    """
    guarded = guarded_deployment(lambda _, sent: sent != "probe-key", status=403)
    with serving(guarded) as base_url:
        verdicts = _verdicts(base_url, key="probe-key")

    assert verdicts["AUTH-1"].word == "broken"
    assert "403" in verdicts["AUTH-1"].why


@pytest.mark.parametrize("refusal", [401, 403])
def test_a_key_the_guard_refuses_blocks_the_catalogue_whichever_status(
    refusal: int,
) -> None:
    """A refused key is the prober's problem behind either refusal.

    The same reading decides both ends of the guard: if only a 401 counted as
    "the key you gave me was refused", a 403 came back as a `Reply` and `DISC-1`
    called the operator's typo a listing this deployment answered wrongly —
    crying wolf at the server for what the command line did.
    """
    guarded = guarded_deployment(
        lambda path, sent: path != "/health" and sent != "probe-key", status=refusal
    )
    with serving(guarded) as base_url:
        verdicts = _verdicts(base_url, key="not-the-key")

    words = {name: verdict.word for name, verdict in verdicts.items()}
    assert "broken" not in words.values(), words
    assert words["DISC-1"] == "unasked"
    assert "--key" in verdicts["DISC-1"].blocker


@pytest.mark.parametrize("status", [404, 500])
def test_a_listing_answering_an_error_is_a_broken_listing_and_not_a_guard(
    status: int,
) -> None:
    """Only the two auth refusals are a guard; every other non-2xx is a defect.

    A deployment whose `/v1/voices` is broken or unrouted is not a deployment
    with a key configured, and reading it as one would ask `AUTH-1` and
    `HEALTH-3` of a guard that does not exist while `DISC-1` — which reports
    this accurately — lost the only claim that owns the listing.
    """
    with serving(shutting("/v1/voices", status=status)) as base_url:
        verdicts = _verdicts(base_url, key="probe-key")

    assert verdicts["DISC-1"].word == "broken"
    assert str(status) in verdicts["DISC-1"].why
    assert verdicts["AUTH-1"].word == "unasked"
    assert verdicts["HEALTH-3"].word == "unasked"


def test_a_closed_guard_holds_auth_1() -> None:
    """The control for the two cases below, so one spoiled guard is what differs.

    Without it a fixture broken in some unrelated way would turn both red for the
    wrong reason and read as proof the prober works.
    """
    closed = guarded_deployment(lambda _, sent: sent != "probe-key")
    with serving(closed) as base_url:
        verdicts = _verdicts(base_url, key="probe-key")

    assert verdicts["AUTH-1"].word == "held"


def test_a_guard_that_admits_a_wrong_key_is_broken() -> None:
    """A guard checking that a key was sent, not which one, is not closed.

    It refuses the keyless caller, so it reads as guarded and every other
    authentication claim behaves — and it lets any string past. `AUTH-1` asks a
    wrong key precisely because that deployment is indistinguishable from a
    correct one until something sends one.
    """
    sloppy = guarded_deployment(lambda _, sent: sent is None)
    with serving(sloppy) as base_url:
        verdicts = _verdicts(base_url, key="probe-key")

    assert verdicts["AUTH-1"].word == "broken"
    assert "the guard is not closed" in verdicts["AUTH-1"].why


def test_a_403_guard_is_not_reported_as_a_guard_standing_open() -> None:
    """Both are `broken`; the reason is the part an operator acts on.

    A single `!= 401` used to answer the two cases either side of this one, so a
    deployment that shut the caller out and one that waved an unusable key
    through came back in the same words — and the words were the open guard's.
    A closed 403 guard reported as a bypass sends an operator hunting something
    that is not there, which is the explanation failing in the one direction
    that gets a prober switched off. Holding the two reasons apart is the
    assertion; that they are both `broken` is why the verdict alone cannot make
    it.

    The statuses are written out rather than read from [`prober.AUTH_REFUSALS`]:
    a case list derived from the constant under test shrinks with it and stays
    green.
    """
    closed = guarded_deployment(lambda _, sent: sent != "probe-key", status=403)
    with serving(closed) as base_url:
        refusing = _verdicts(base_url, key="probe-key")["AUTH-1"]

    admits_anything = guarded_deployment(lambda _, sent: sent is None)
    with serving(admits_anything) as base_url:
        admitting = _verdicts(base_url, key="probe-key")["AUTH-1"]

    assert refusing.word == "broken" and admitting.word == "broken"
    assert "the guard is not closed" in admitting.why
    assert "the guard is not closed" not in refusing.why
    assert "403" in refusing.why and "401" in refusing.why


def test_a_guard_admitting_a_wrong_key_with_204_is_still_not_closed() -> None:
    """Any 2xx is the guard admitting, not 200 alone.

    A deployment waving every non-empty key through with a 204 is exactly as
    open as one answering 200, and reading only 200 as the admission drops this
    into the wrong-status arm to be told it did not let the caller in — the
    reassuring direction, about a deployment anyone can walk into. The fixture
    is written here rather than given to [`guarded_deployment`] as another
    parameter, because one test needing a shape is not the shape becoming a
    mode every other caller has to read past.
    """
    app = lying_deployment([WELL_FORMED])

    @app.middleware("http")
    async def guard(request: Request, call_next: Any) -> Response:
        sent = request.headers.get("xi-api-key")
        if sent is None:
            return Response(
                content=json.dumps({"detail": "invalid xi-api-key"}).encode(),
                status_code=401,
                media_type="application/json",
            )
        if sent != "probe-key":
            return Response(status_code=204)
        return await call_next(request)

    with serving(app) as base_url:
        verdict = _verdicts(base_url, key="probe-key")["AUTH-1"]

    assert verdict.word == "broken"
    assert "the guard is not closed" in verdict.why
    assert "204" in verdict.why


@pytest.mark.parametrize(
    "path, claim", [("/health", "HEALTH-1"), ("/v1/voices", "DISC-1")]
)
def test_a_body_that_is_not_json_breaks_the_claim_whose_subject_it_is(
    path: str, claim: str
) -> None:
    """`unasked` here said "I could not find out" about a deployment caught out.

    Both claims' docstrings call a body with no usable `voices` array this
    promise *broken*, and `_parsed_voices`/`_parsed_health_voices` return a
    complaint rather than raising precisely so the claim that owns the body can
    say so. A body that was not JSON at all never reached them: `Reply.json`
    raised past both readers, and an ingress serving an HTML error page under a
    200 came back `unasked` — the word this project keeps for a precondition
    that failed, reporting a conformance failure as a question nobody asked.

    A proxy answering instead of the deployment was the reason given for the old
    behaviour, and it does not survive the comparison: a proxy answering
    `{"error": "bad gateway"}` is JSON, has always been reported `broken`, and
    is no more the deployment's doing than an HTML page is.
    """
    app = lying_deployment([WELL_FORMED])

    @app.middleware("http")
    async def interpose(request: Request, call_next: Any) -> Response:
        if request.url.path == path:
            return Response(
                content=b"<html><body>502 Bad Gateway</body></html>",
                status_code=200,
                media_type="text/html",
            )
        return await call_next(request)

    with serving(app) as base_url:
        verdict = _verdicts(base_url)[claim]

    assert verdict.word == "broken"
    assert "not JSON" in verdict.why


def test_a_refusal_body_that_is_not_json_breaks_auth_1() -> None:
    """`AUTH-1` promises the detail, so a body that cannot carry one broke it.

    A gateway shutting the caller out with a plain-text 401 is the shape this
    claim exists to catch: the guard is closed and a caller still cannot tell
    this refusal from any other, which is the whole of what `AUTH-1` asks.
    """
    app = lying_deployment([WELL_FORMED])

    @app.middleware("http")
    async def guard(request: Request, call_next: Any) -> Response:
        if request.headers.get("xi-api-key") != "probe-key":
            return Response(
                content=b"Forbidden by the gateway",
                status_code=401,
                media_type="text/plain",
            )
        return await call_next(request)

    with serving(app) as base_url:
        verdict = _verdicts(base_url, key="probe-key")["AUTH-1"]

    assert verdict.word == "broken"
    assert "invalid xi-api-key" in verdict.why


def test_a_refusal_carrying_another_detail_is_broken() -> None:
    """The 401 is not the whole claim: a caller has to be able to read it.

    `AUTH-1` promises a specific body, because a 401 from a proxy between the
    caller and the deployment looks identical otherwise — and a client that
    cannot tell them apart retries the wrong one.
    """
    mumbling = guarded_deployment(lambda _, sent: sent != "probe-key", detail="nope")
    with serving(mumbling) as base_url:
        verdicts = _verdicts(base_url, key="probe-key")

    assert verdicts["AUTH-1"].word == "broken"
    assert "nope" in verdicts["AUTH-1"].why
    assert prober.INVALID_KEY_DETAIL in verdicts["AUTH-1"].why


def test_the_readme_still_publishes_a_table_this_test_can_read() -> None:
    """The check below is worthless if the table stopped parsing.

    A regex over prose is a map that can go stale silently: a reformatted table
    yields no rows, and "every documented endpoint is asked" then passes against
    nothing at all ([LAW:no-silent-failure]).
    """
    published = published_endpoints()

    assert "/health" in published["GET"], published
    assert "/v1/text-to-speech/{voice_id}" in published["POST"], published


def test_auth_2_asks_every_endpoint_the_readme_documents() -> None:
    """The probed surface, held equal to the published one.

    `AUTH-2`'s subject is the *documented* surface, so the tuples cannot be
    derived from `elvenspeak.api` — that is this checkout's route table and the
    prober judges foreign builds. README is the document they mirror, and holding
    them equal to it is what makes an endpoint added there arrive here as a test
    failure rather than as a path nobody keyless ever asks
    ([LAW:one-source-of-truth]).
    """
    published = published_endpoints()

    assert set(prober.DOCUMENTED_SYNTHESES) == published["POST"]
    # `/health` is the one documented read outside any guard — README says it
    # never requires a key, and `HEALTH-3` is the claim that asks it.
    assert set(prober.DOCUMENTED_READS) == published["GET"] - {"/health"}


# ------------------------------------- the twenty-eight formats, and the 422s


def test_the_prober_and_the_server_transcribe_one_published_format_set() -> None:
    """The one table here held against this build's code, and why that is right.

    `DOCUMENTED_READS` is deliberately *not* held against `elvenspeak.api`: which
    paths a build routes is that build's own decision, so a remote one may differ
    without being wrong. The format set is the opposite — it belongs to
    ElevenLabs, both tuples are transcriptions of the same published list, and a
    deployment answering a different 28 is non-conformant by definition. Holding
    two transcriptions equal is what catches a typo in either
    ([LAW:one-source-of-truth]).

    The count is asserted beside them because equality alone cannot catch the two
    losing a row together, and README publishes the number as "All 28".
    """
    assert set(prober.PUBLISHED_FORMATS) == set(SUPPORTED_OUTPUT_FORMATS)
    assert len(prober.PUBLISHED_FORMATS) == 28, prober.PUBLISHED_FORMATS
    assert prober.DEFAULT_FORMAT == DEFAULT_OUTPUT_FORMAT
    assert prober.UNKNOWN_FORMAT not in SUPPORTED_OUTPUT_FORMATS
    assert prober.MAX_TEXT_CHARACTERS == MAX_TEXT_LENGTH


def test_every_refusal_the_prober_carries_is_a_claim_it_asks() -> None:
    """A row of `REFUSALS` nothing names is a refusal nobody is told about.

    `REF-5` iterates the table, so an orphan row would still have its headers
    read and would still never have its own 422 judged — a promise carried in the
    prober and reported by nothing ([LAW:no-silent-failure]).
    """
    asked = {claim.id for claim in prober.CLAIMS}

    assert set(prober.REFUSALS) <= asked, sorted(set(prober.REFUSALS) - asked)


#: Every claim `piper-conformance-e16.4` owns, so a case below asserting one of
#: them is broken cannot be passing because the claim silently stopped being
#: asked. Read off the document rather than written out.
E16_4_CLAIMS = sorted(
    claim_id for claim_id, (_, by) in documented().items() if "e16.4" in by
)


def test_a_conformant_stand_in_holds_every_format_and_refusal_claim() -> None:
    """The control for every case below, so one spoiled answer is what differs.

    Fifteen claims, none of them `unasked`: a stand-in that merely failed to be
    caught would read the same as one that told the truth, and the list is taken
    from the document so a claim dropped from `CLAIMS` fails here rather than
    quietly stopping being the control.
    """
    with serving(lying_deployment([WELL_FORMED])) as base_url:
        words = _words(base_url)

    assert len(E16_4_CLAIMS) == 15, E16_4_CLAIMS
    assert {claim_id: words[claim_id] for claim_id in E16_4_CLAIMS} == {
        claim_id: "held" for claim_id in E16_4_CLAIMS
    }


def test_a_format_the_deployment_will_not_serve_breaks_fmt_1() -> None:
    """"All 28 are accepted" is the claim, so one refused is the claim failing."""

    def refusing(output_format: str | None, body: dict[str, Any]) -> Response:
        if output_format == "opus_48000_96":
            return Response(content=b"no opus here", status_code=415)
        return conformant(output_format, body)

    with serving(lying_deployment([WELL_FORMED], speaks=refusing)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["FMT-1"].word == "broken"
    assert "opus_48000_96" in verdicts["FMT-1"].why
    # And the claims about the *shape* of that audio say they could not look,
    # rather than reporting on the bytes of a refusal.
    assert verdicts["FMT-2"].word == "unasked"
    assert verdicts["FMT-3"].word == "unasked"


def test_the_wrong_content_type_breaks_fmt_2() -> None:
    """A client routing on the type plays the answer into the wrong decoder."""
    mislabelling = spoiling(lambda _, audio, __: (audio, "application/octet-stream"))

    with serving(lying_deployment([WELL_FORMED], speaks=mislabelling)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["FMT-2"].word == "broken"
    assert "application/octet-stream" in verdicts["FMT-2"].why
    # The bytes were never touched, so the claim about them still holds — which
    # is what makes this case about the label and not about the audio.
    assert verdicts["FMT-3"].word == "held"


def test_a_deployment_substituting_bytes_under_the_right_label_breaks_fmt_3() -> None:
    """THE CASE THIS TICKET IS FOR.

    A deployment that accepts all 28, labels every answer with the content type
    the caller asked for, and can really produce exactly one of them. Every check
    that reads a status code passes it; every check that reads a header passes
    it. Only the bytes give it away, which is why `FMT-3` reads the bytes.

    `FMT-1` and `FMT-2` are asserted `held` here on purpose: they are the checks
    the ticket says a deployment like this must not be allowed to pass *on*, and
    a case where they went red too would not have shown that.
    """
    one_format = spoiling(lambda _, __, content_type: (_audio("wav_22050")[0], content_type))

    with serving(lying_deployment([WELL_FORMED], speaks=one_format)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["FMT-1"].word == "held"
    assert verdicts["FMT-2"].word == "held"
    assert verdicts["FMT-3"].word == "broken"
    assert "RIFF" in verdicts["FMT-3"].why


def test_one_length_answered_at_every_pcm_rate_breaks_fmt_4() -> None:
    """The substitution `_bare` cannot see, caught where the arithmetic can.

    A deployment resampling nothing and returning its engine's own bytes under
    every `pcm_*` name answers raw samples every time, so the signature check is
    satisfied — and the seven rates then state seven different durations for one
    utterance, which is the contradiction this claim exists to find.
    """
    one_rate = spoiling(
        lambda name, audio, content_type: (
            (_audio("pcm_22050")[0], content_type)
            if (name or "").startswith("pcm")
            else (audio, content_type)
        )
    )

    with serving(lying_deployment([WELL_FORMED], speaks=one_rate)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["FMT-3"].word == "held", "the bytes still look like raw samples"
    assert verdicts["FMT-4"].word == "broken"
    assert "pcm_8000" in verdicts["FMT-4"].why


def test_a_wav_header_stating_another_rate_breaks_fmt_5() -> None:
    """A RIFF header telling the truth about audio that is not what was asked for."""
    one_rate = spoiling(
        lambda name, audio, content_type: (
            (_wav(11025, 1102), content_type)
            if (name or "").startswith("wav")
            else (audio, content_type)
        )
    )

    with serving(lying_deployment([WELL_FORMED], speaks=one_rate)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["FMT-3"].word == "held", "it is still a RIFF/WAVE file"
    assert verdicts["FMT-5"].word == "broken"
    assert "11025" in verdicts["FMT-5"].why


def test_an_id3_tag_breaks_fmt_6() -> None:
    """README:62's byte-for-byte promise, and the way it is ordinarily broken.

    `ffmpeg` writes an ID3v2 header by default — this is one forgotten flag away
    from being what a real build does, not an exotic lie.
    """
    tagged = spoiling(
        lambda name, audio, content_type: (
            (b"ID3\x04\x00\x00\x00\x00\x00\x0a" + audio, content_type)
            if (name or DEFAULT_OUTPUT_FORMAT).startswith("mp3")
            else (audio, content_type)
        )
    )

    with serving(lying_deployment([WELL_FORMED], speaks=tagged)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["FMT-6"].word == "broken"
    assert "ID3" in verdicts["FMT-6"].why


def test_a_default_that_is_not_the_documented_one_breaks_fmt_7() -> None:
    """An MP3 is not the claim — `mp3_44100_128` is.

    A deployment defaulting to `mp3_22050_32` answers an MP3, under
    `audio/mpeg`, starting at a frame sync with no tag. Every other format claim
    holds against it, and a client that saved no `output_format` gets audio at a
    quarter of the documented rate.
    """
    other_default = spoiling(
        lambda name, audio, content_type: (
            (b"\xff\xfb" + bytes(16), content_type) if name is None else (audio, content_type)
        )
    )

    with serving(lying_deployment([WELL_FORMED], speaks=other_default)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["FMT-7"].word == "broken"
    assert DEFAULT_OUTPUT_FORMAT in verdicts["FMT-7"].why


def sampling() -> Speaks:
    """A deployment whose backend draws a new utterance for every request.

    What chatterbox really is, and what `piper-pipeline-uor` established has to
    be asked before one length is read against another: each answer is a little
    longer than the last, so two identical requests never come back identical.
    """
    drawn = iter(range(1000))

    def speaks(output_format: str | None, body: dict[str, Any]) -> Response:
        answered = conformant(output_format, body)
        if answered.status_code != 200:
            return answered
        return Response(
            content=answered.body + b"\0\0" * next(drawn),
            media_type=answered.headers["content-type"],
            headers={"x-elvenspeak-voice": str(WELL_FORMED["voice_id"])},
        )

    return speaks


def test_a_voice_that_does_not_repeat_leaves_both_comparisons_unasked() -> None:
    """[LAW:no-silent-failure] Never `held`, and never `broken` either.

    `FMT-4` and `FMT-7` both read one of this voice's lengths against another,
    and against a sampling backend that difference is the sampler's. Reporting
    `broken` would blame a deployment for a property `README.md:615` says piper
    already has; reporting `held` would rest a verdict on a comparison that
    happened not to notice. The third word is for exactly this.
    """
    with serving(lying_deployment([WELL_FORMED], speaks=sampling())) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["FMT-4"].word == "unasked"
    assert verdicts["FMT-7"].word == "unasked"
    assert "two different utterances" in verdicts["FMT-4"].blocker
    # Everything that reads one answer on its own is still asked and still holds.
    assert verdicts["FMT-1"].word == "held"
    assert verdicts["FMT-3"].word == "held"
    assert verdicts["FMT-5"].word == "held"


def offering(supported: list[str]) -> Speaks:
    """[`conformant`], refusing an unknown format with `supported` beside it."""

    def speaks(output_format: str | None, body: dict[str, Any]) -> Response:
        if output_format is not None and output_format not in SUPPORTED_OUTPUT_FORMATS:
            return _refusal(
                {
                    "message": f"unsupported output_format: {output_format!r}",
                    "supported": supported,
                }
            )
        return conformant(output_format, body)

    return speaks


@pytest.mark.parametrize(
    ("supported", "named"),
    [
        # Short: a caller is sent looking for another format when the one they
        # asked for is really served.
        (list(SUPPORTED_OUTPUT_FORMATS)[:10], "omits"),
        # Long: a caller is sent to a 422 by the deployment's own advice.
        (list(SUPPORTED_OUTPUT_FORMATS) + ["flac_44100"], "invents"),
    ],
)
def test_a_supported_set_that_is_not_the_28_breaks_fmt_8(
    supported: list[str], named: str
) -> None:
    """Equality in both directions, because the two failures mislead differently."""
    with serving(lying_deployment([WELL_FORMED], speaks=offering(supported))) as url:
        verdicts = _verdicts(url)

    assert verdicts["FMT-8"].word == "broken"
    assert named in verdicts["FMT-8"].why
    # The refusal is still a refusal naming the value, so the claim about *that*
    # holds — which is what keeps the two claims telling a reader different things.
    assert verdicts["REF-1"].word == "held"


def test_a_refusal_carrying_no_supported_set_at_all_breaks_fmt_8() -> None:
    """The pydantic shape arriving where this service's own object belongs.

    Not a shape complaint: `REF-1` accepts it and says so. `FMT-8` is `broken`
    because there is no set to read, which is the thing a caller was promised.
    """
    def pydantic_shaped(output_format: str | None, body: dict[str, Any]) -> Response:
        if output_format is not None and output_format not in SUPPORTED_OUTPUT_FORMATS:
            return _refusal(
                [{"type": "enum", "loc": ["query", "output_format"], "input": output_format,
                  "msg": "supported values are published in the API reference"}]
            )
        return conformant(output_format, body)

    with serving(lying_deployment([WELL_FORMED], speaks=pydantic_shaped)) as url:
        verdicts = _verdicts(url)

    assert verdicts["FMT-8"].word == "broken"
    assert "no `supported` array" in verdicts["FMT-8"].why
    assert verdicts["REF-1"].word == "held", "the value and the word are both there"


def test_a_supported_array_of_objects_breaks_fmt_8_rather_than_ending_the_run() -> None:
    """A refusal this prober cannot read costs one verdict, never the report.

    `set()` over an array of objects raises `TypeError`, and `_finding` catches
    only `Blocked` — so before this was parsed at `_supported_arrays`, one
    malformed deployment took all 21 verdicts down with it.
    """
    def objects_not_names(output_format: str | None, body: dict[str, Any]) -> Response:
        if output_format is not None and output_format not in SUPPORTED_OUTPUT_FORMATS:
            return _refusal(
                {
                    "message": f"unsupported output_format: {output_format!r}",
                    "supported": [{"name": name} for name in SUPPORTED_OUTPUT_FORMATS],
                }
            )
        return conformant(output_format, body)

    with serving(lying_deployment([WELL_FORMED], speaks=objects_not_names)) as url:
        verdicts = _verdicts(url)

    assert verdicts["FMT-8"].word == "broken"
    assert "no `supported` array" in verdicts["FMT-8"].why
    assert len(verdicts) == len(prober.CLAIMS), "every other claim still got asked"


def test_a_refusal_nested_past_reading_breaks_fmt_8_rather_than_ending_the_run() -> None:
    """Valid JSON the walk cannot descend.

    `json.loads` parses a hundred thousand levels without complaint, so the depth
    that matters is the prober's own recursion limit, not the parser's.

    Written as raw bytes rather than a nested object, because `json.dumps` has the
    same limit: serializing one of these server-side raises inside the fixture, and
    the prober would then be reporting an honest `unasked` about a 500 it really
    did receive — a green test proving nothing about the walk.
    """
    depth = sys.getrecursionlimit() * 3
    nested = b'{"detail": ' + b"[" * depth + b'"bottom"' + b"]" * depth + b"}"

    def nested_past_reading(output_format: str | None, body: dict[str, Any]) -> Response:
        if output_format is not None and output_format not in SUPPORTED_OUTPUT_FORMATS:
            return Response(
                content=nested, status_code=422, media_type="application/json"
            )
        return conformant(output_format, body)

    with serving(lying_deployment([WELL_FORMED], speaks=nested_past_reading)) as url:
        verdicts = _verdicts(url)

    assert verdicts["FMT-8"].word == "broken"
    assert "nested too deeply" in verdicts["FMT-8"].why
    assert len(verdicts) == len(prober.CLAIMS), "every other claim still got asked"


def test_a_refusal_spelling_the_field_inside_a_longer_word_does_not_hold() -> None:
    """`context` is not the deployment naming `text`.

    The rule `REF-2` states is that the body names what was wrong. A raw byte
    substring made that rule satisfiable by any refusal mentioning a context
    window — a check that cannot fail, which is the defect this table replaced.
    """
    def refusing_without_naming(
        output_format: str | None, body: dict[str, Any]
    ) -> Response:
        if isinstance(body.get("text"), str) and not body["text"].strip():
            return _refusal("request exceeds the maximum context window")
        return conformant(output_format, body)

    with serving(lying_deployment([WELL_FORMED], speaks=refusing_without_naming)) as url:
        verdicts = _verdicts(url)

    assert verdicts["REF-2"].word == "broken"
    assert "naming none of" in verdicts["REF-2"].why
    assert verdicts["REF-4"].word == "held", "the rows that do name `text` are untouched"
    # This fixture spoils one refusal and nothing else, so the format rows are the
    # evidence it is otherwise conformant. Without them the test passes against a
    # stand-in serving no audio at all, which is what it did until review caught it.
    assert verdicts["FMT-2"].word == "held"
    assert verdicts["FMT-3"].word == "held"


def never_refusing(output_format: str | None, body: dict[str, Any]) -> Response:
    """A deployment that synthesizes whatever it is sent.

    The server this project used to be: it accepted the parameter, ignored what
    it could not honour, and answered 200 with something plausible.
    """
    wanted = output_format if output_format in SUPPORTED_OUTPUT_FORMATS else DEFAULT_OUTPUT_FORMAT
    audio, content_type = _audio(wanted)
    return Response(
        content=audio,
        media_type=content_type,
        headers={"x-elvenspeak-voice": str(WELL_FORMED["voice_id"])},
    )


@pytest.mark.parametrize("claim_id", sorted(prober.REFUSALS))
def test_a_request_that_is_served_instead_of_refused_breaks_its_claim(
    claim_id: str,
) -> None:
    """Each of the five asked of a deployment that refuses nothing.

    One case per row rather than a single deployment checked once, so a row whose
    request this prober never actually sends cannot ride the others to green —
    the gap that survived three rounds on `AUTH-2`'s per-voice reads.
    """
    with serving(lying_deployment([WELL_FORMED], speaks=never_refusing)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts[claim_id].word == "broken", prober.REFUSALS[claim_id].described
    assert "200" in verdicts[claim_id].why


def test_a_deployment_that_refuses_nothing_leaves_ref_5_unasked() -> None:
    """A 200 is not a refusal, so its headers are not `REF-5`'s subject.

    A synthesis is *supposed* to carry `x-elvenspeak-voice`; reading one here and
    reporting `broken` would file five conformance failures under the one claim
    that has nothing to do with them, and name the wrong fix.
    """
    with serving(lying_deployment([WELL_FORMED], speaks=never_refusing)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["REF-5"].word == "unasked"
    assert "REF-" in verdicts["REF-5"].blocker


def mumbling(output_format: str | None, body: dict[str, Any]) -> Response:
    """[`conformant`], with every refusal reduced to a word nobody can act on."""
    answered = conformant(output_format, body)
    if answered.status_code != 422:
        return answered
    return _refusal("nope")


@pytest.mark.parametrize("claim_id", sorted(prober.REFUSALS))
def test_a_refusal_naming_nothing_it_refused_breaks_its_claim(claim_id: str) -> None:
    """The 422 is not the whole claim: a caller has to be able to read it.

    This is the case the ticket's recommendation could not have caught on its own
    terms. "The value you sent appears somewhere in the body" is satisfied by
    every body ever written when the value is `""` or `"   "`, so `REF-2` and
    `REF-3` would have reported `held` against a deployment that refused with a
    shrug. What the prober requires instead is that the body *name* what was
    wrong — the value where it is distinctive, the field where it is not.
    """
    with serving(lying_deployment([WELL_FORMED], speaks=mumbling)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts[claim_id].word == "broken", prober.REFUSALS[claim_id].described
    assert "naming none of" in verdicts[claim_id].why


def _named(output_format: str | None, body: dict[str, Any]) -> tuple[str, Any]:
    """Which field this request is wrong in, and the value it carried."""
    if output_format is not None and output_format not in SUPPORTED_OUTPUT_FORMATS:
        return "output_format", output_format
    if not isinstance(body.get("language_code", ""), str):
        return "language_code", body["language_code"]
    return "text", body.get("text")


def one_shape(record: Callable[[str, Any], Any]) -> Speaks:
    """A deployment refusing everything in whichever shape `record` builds.

    [LAW:one-type-per-behavior] Two deployments, one function: what separates
    this service's object refusal from pydantic's array is the value a `detail`
    carries, and the decision `piper-conformance-e16.4` recorded is that the
    prober reads neither. A fixture apiece would have stated that as two
    coincidences instead of one property.
    """

    def speaks(output_format: str | None, body: dict[str, Any]) -> Response:
        if conformant(output_format, body).status_code != 422:
            return conformant(output_format, body)
        return _refusal(record(*_named(output_format, body)))

    return speaks


def test_the_prober_reads_neither_422_shape_and_accepts_both() -> None:
    """The decision this ticket was left to make, asserted rather than described.

    This service refuses in two shapes — its own object under `detail.message`
    and pydantic's array of error records under `detail[].input` — and
    `README.md:32` promises only "a 422 quoting the value you sent", which both
    keep. A prober demanding either would report `broken` against a deployment
    keeping its documented promise, and the only way to satisfy it would be to
    rewrite pydantic's refusal into a body no other FastAPI-shaped server sends.

    So the same five claims are asked of two deployments that agree on every fact
    and share no byte of structure, and all ten verdicts are `held`.
    """
    shapes = {
        "object": lambda field, value: {
            "message": f"{field} cannot be {value!r:.40}",
            "supported": list(SUPPORTED_OUTPUT_FORMATS),
        },
        "array": lambda field, value: [
            {
                "type": "value_error",
                "loc": ["body", field],
                "input": value,
                "supported": list(SUPPORTED_OUTPUT_FORMATS),
            }
        ],
    }

    verdicts = {}
    for shape, record in shapes.items():
        with serving(lying_deployment([WELL_FORMED], speaks=one_shape(record))) as url:
            verdicts[shape] = _words(url)

    for shape, words in verdicts.items():
        asked = {claim_id: words[claim_id] for claim_id in sorted(prober.REFUSALS)}
        assert asked == {claim_id: "held" for claim_id in asked}, (shape, asked)


def test_a_refusal_carrying_a_voice_header_breaks_ref_5() -> None:
    """Nothing spoke, so there is no voice to report having spoken it.

    README:115's "every synthesis response carries `x-elvenspeak-voice`" is
    bounded by the responses that *are* syntheses, and a 422 naming a voice tells
    a caller a decision the server never made.
    """

    def leaking(output_format: str | None, body: dict[str, Any]) -> Response:
        answered = conformant(output_format, body)
        answered.headers["x-elvenspeak-voice"] = str(WELL_FORMED["voice_id"])
        return answered

    with serving(lying_deployment([WELL_FORMED], speaks=leaking)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["REF-5"].word == "broken"
    assert "x-elvenspeak-voice" in verdicts["REF-5"].why


def test_refusing_a_body_field_this_build_has_never_heard_of_breaks_ref_7() -> None:
    """`extra="forbid"` is the stricter choice and the wrong one here.

    ElevenLabs adds body fields over time, so a deployment refusing one it does
    not model breaks clients that are correct against the newer API — which is
    the compatibility this whole service is.
    """

    def strict(output_format: str | None, body: dict[str, Any]) -> Response:
        unmodelled = sorted(set(body) - _MODELLED)
        if unmodelled:
            return _refusal([{"type": "extra_forbidden", "loc": ["body", unmodelled[0]]}])
        return conformant(output_format, body)

    with serving(lying_deployment([WELL_FORMED], speaks=strict)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["REF-7"].word == "broken"
    assert prober.INVENTED_FIELD in verdicts["REF-7"].why


def test_an_unmodelled_field_dropped_without_a_word_breaks_ref_7() -> None:
    """The quieter half, and the one a status-code check waves straight through.

    A deployment that keeps the request, answers 200, and says nothing has told
    the caller their whole request was honoured when part of it was ignored —
    which is rule 2 exactly inverted.
    """

    def silent(output_format: str | None, body: dict[str, Any]) -> Response:
        kept = {key: body[key] for key in body.keys() & _MODELLED}
        return conformant(output_format, kept)

    with serving(lying_deployment([WELL_FORMED], speaks=silent)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["REF-7"].word == "broken"
    assert "dropped" in verdicts["REF-7"].why


# ------------------------- a real deployment telling one lie about one voice


#: The per-voice claims read a deployment's *behaviour*, not a payload it can be
#: handed, so the twenty-four `piper-conformance-e16.5` landed are asked of a real
#: `engine_app` with one observable rewritten on its way out rather than of a
#: hand-built stand-in. [`lying_deployment`] above is the right shape for the
#: format and refusal claims, whose whole subject is a body; expressing "this
#: voice declares `speed` and then ignores it" there would have meant
#: reimplementing capability honesty, substitution, alias resolution, `model_id`
#: arbitration and language reduction in the fixture — at which point the fixture,
#: not the server, is what the tests assert against ([LAW:behavior-not-structure]).
#:
#: So the control is the real server, which the live run and
#: [`test_an_open_deployment_breaks_nothing`] both establish holds every one of
#: these, and each case below buys exactly one lie.


@dataclass(frozen=True)
class Answer:
    """One answer on its way out, and everything a lie might need to change it.

    Carries the *request* beside the response because that is what selects the
    draw: eleven of the requests a voice is asked go to one path and differ only
    in the body, so a lie keyed on the path alone could not tell the neutral draw
    from the one carrying a rate ([LAW:types-are-the-program] — the discriminator
    the tamper needs is in the type rather than re-derived per case).
    """

    path: str
    sent: dict[str, Any]
    status: int
    headers: dict[str, str]
    body: bytes


#: One deployment's lie: the answer it was about to give, and the one it gives.
Tamper = Callable[[Answer], Answer]


def tampering(tamper: Tamper, app: FastAPI | None = None) -> FastAPI:
    """A real deployment whose every answer passes through `tamper` on the way out.

    [LAW:dataflow-not-control-flow] Every deployment that lies about a voice is
    this one function carrying a different value, rather than a fixture apiece:
    what separates "reports the rate ignored" from "moves the speaker" is two
    lines over an [`Answer`], and a second server around each would bury that.

    `content-length` is dropped rather than carried, because a lie that changes
    the body changes it — a stale length is a truncated response, which every
    claim below would report as a defect that is really this helper's.
    """
    served = engine_app("piper", DECLARED_VOICES) if app is None else app

    @served.middleware("http")
    async def spoil(request: Request, call_next: Any) -> Response:
        sent = await request.body()
        answered = await call_next(request)
        given = tamper(
            Answer(
                path=request.url.path,
                sent=json.loads(sent) if sent else {},
                status=answered.status_code,
                headers={
                    name: value
                    for name, value in answered.headers.items()
                    if name.lower() != "content-length"
                },
                body=b"".join([chunk async for chunk in answered.body_iterator]),
            )
        )
        return Response(
            content=given.body, status_code=given.status, headers=given.headers
        )

    return served


def _rewritten(answer: Answer, body: Any) -> Answer:
    """`answer`, carrying `body` rendered as JSON instead of what it held."""
    return replace(answer, body=json.dumps(body).encode())


def _also_ignored(answer: Answer, name: str) -> Answer:
    """`answer`, additionally reporting `name` as a parameter it dropped."""
    named = answer.headers.get("x-elvenspeak-ignored")
    return replace(
        answer,
        headers={
            **answer.headers,
            "x-elvenspeak-ignored": name if named is None else f"{named}, {name}",
        },
    )


def _not_ignored(answer: Answer, name: str) -> Answer:
    """`answer`, no longer reporting `name` among the parameters it dropped."""
    kept = [
        part.strip()
        for part in answer.headers.get("x-elvenspeak-ignored", "").split(",")
        if part.strip() and part.strip() != name
    ]
    headers = {
        key: value
        for key, value in answer.headers.items()
        if key.lower() != "x-elvenspeak-ignored"
    }
    return replace(
        answer,
        headers=headers if not kept else {**headers, "x-elvenspeak-ignored": ", ".join(kept)},
    )


def _spoke(answer: Answer) -> bool:
    """Whether this answer is a synthesis that really carried audio."""
    return answer.status == 200 and answer.path.startswith("/v1/text-to-speech/")


def _sent_to(answer: Answer) -> str:
    """The voice id this request addressed, as the caller spelled it.

    Five of the lies below are keyed on *which voice was asked*, and two of those
    ids are [`prober.UNPRINTABLE_VOICE`] and an alias — ids that reach the path
    percent-encoded. Decoded in one place rather than at each of the five, so the
    question "does this arrive encoded or not" has one answer instead of five that
    can disagree ([LAW:parse-dont-validate] — the id is parsed out of the path
    once and every tamper below reads the parsed value, never the path).

    [`unquote`] rather than a measurement of what this uvicorn happens to do: it
    is the identity on an already-decoded path, so this is right either way and
    stays right if that changes.
    """
    return unquote(answer.path).removeprefix("/v1/text-to-speech/").split("/")[0]


def _setting(answer: Answer, name: str) -> Any:
    """What this request asked for in `voice_settings.<name>`, if anything."""
    return answer.sent.get("voice_settings", {}).get(name)


def _publishing(answer: Answer, alias: str, on: str) -> Answer:
    """`answer`, with `alias` published on voice `on` wherever this body lists it.

    The listing and the single read both, because an alias published in only one
    of them is a deployment no server could be — two lies where the discipline of
    this section is exactly one.

    `engine_app` has no way to publish an alias: aliases come from the catalogue's
    own table, which these fixtures leave empty, so the listing is the only place
    `SUB-5` can be given a subject at all.
    """
    if not answer.path.startswith("/v1/voices"):
        return answer
    body = json.loads(answer.body)
    listed = body["voices"] if isinstance(body, dict) and "voices" in body else [body]
    for entry in listed:
        if isinstance(entry, dict) and entry.get("voice_id") == on:
            entry["aliases"] = [*entry.get("aliases", []), alias]
    return _rewritten(answer, body)


def _plain_spoken() -> FastAPI:
    """The same deployment, its voices declaring no capability at all.

    `CAP-2` and `CAP-4` are the arms a voice's *own* declaration selects, and
    [`DECLARED_VOICES`] declare everything — so against the default server both
    are `unasked` and no lie told over them has a subject to lie about. This is
    the one fact those two claims read, set the other way
    ([LAW:one-type-per-behavior] — one deployment carrying a different value, not
    a second kind of fixture).
    """
    return engine_app("piper", DECLARED_VOICES, capabilities=frozenset())


def test_the_tampering_seam_changes_nothing_when_it_tells_no_lie() -> None:
    """The control for every case below, so one lie is what differs.

    Without this, a middleware that mangled every answer would turn all of them
    red for the wrong reason and read as proof the prober works. It also pins the
    one thing the helper is easy to get wrong: a body rewritten under a stale
    `content-length` is a truncated response, and a truncated response breaks
    claims that have nothing to do with the lie being told.
    """
    with serving(tampering(lambda answer: answer)) as base_url:
        words = _words(base_url)

    assert "broken" not in words.values(), words


#: The two voices [`engine_app`] offers below, named off the fixture rather than
#: spelled again: a lie that addressed a voice this deployment does not have would
#: be testing substitution instead of whatever it meant to test.
FIRST, SECOND = DECLARED_VOICES[0].id, DECLARED_VOICES[1].id


def test_discovery_that_substitutes_an_unknown_id_breaks_disc_2() -> None:
    """A read that substitutes is the failure `DISC-2` exists to catch.

    The same unknown id earns audio from a synthesis endpoint and must earn a 404
    here, because a client reading the catalogue to populate a picker would
    otherwise offer a voice nothing has.
    """

    def substituting(answer: Answer) -> Answer:
        if answer.path == f"/v1/voices/{prober.FOREIGN_VOICE}":
            return _rewritten(replace(answer, status=200), {"voice_id": FIRST})
        return answer

    with serving(tampering(substituting)) as base_url:
        verdict = _verdicts(base_url)["DISC-2"]

    assert verdict.word == "broken"
    assert "discovery substituted" in verdict.why


def test_one_voices_malformed_capabilities_still_leaves_the_id_claims_askable() -> None:
    """Each claim gated on what it reads rather than on the widest gate available.

    `DISC-2` reads a voice's id and `DISC-5` its language; neither reads
    `capabilities`, and reading both off the whole-catalogue declarations left
    them unasked over a deployment they could have judged on the spot. `DISC-1`,
    whose claim the malformed field really is, still reports it broken — which is
    the point: one bad field should cost exactly one verdict.
    """

    def unparseable(answer: Answer) -> Answer:
        if answer.path != "/v1/voices":
            return answer
        listing = json.loads(answer.body)
        listing["voices"][1]["capabilities"] = "speed"
        return _rewritten(answer, listing)

    with serving(tampering(unparseable)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["DISC-1"].word == "broken"
    assert verdicts["DISC-2"].word == "held", verdicts["DISC-2"]
    assert verdicts["DISC-5"].word == "held", verdicts["DISC-5"]


def test_a_language_that_is_not_a_tag_at_all_leaves_disc_5_unasked() -> None:
    """The precondition `DISC-5` genuinely has, kept while the borrowed one goes.

    A `language` that is not a usable string is nothing to reduce, so this claim
    has no subject — and the blocker names `DISC-1`, which owns whether a
    published field is well-typed, rather than reporting the voice broken twice.
    """

    def untyped(answer: Answer) -> Answer:
        if answer.path != "/v1/voices":
            return answer
        listing = json.loads(answer.body)
        listing["voices"][1]["language"] = 7
        return _rewritten(answer, listing)

    with serving(tampering(untyped)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["DISC-1"].word == "broken"
    assert verdicts["DISC-5"].word == "unasked", verdicts["DISC-5"]
    assert "DISC-1 is the claim" in verdicts["DISC-5"].blocker


def test_a_language_that_is_not_a_bare_family_breaks_disc_5() -> None:
    """`ES` is the spelling that really shipped, and no caller's `es` can match it.

    The voice stays listed, stays addressable and still speaks — it is only
    unreachable *by the language it claims*, which is why nothing but this claim
    notices.
    """

    def shouting(answer: Answer) -> Answer:
        if answer.path == "/v1/voices":
            listing = json.loads(answer.body)
            listing["voices"][0]["language"] = "EN"
            return _rewritten(answer, listing)
        return answer

    with serving(tampering(shouting)) as base_url:
        verdict = _verdicts(base_url)["DISC-5"]

    assert verdict.word == "broken"
    assert "reduces to 'en'" in verdict.why


def test_a_models_listing_offering_router_breaks_mod_2() -> None:
    """`router` names a deployment that synthesizes nothing.

    The regression `piper-routing-7e2.17` fixed and the one this claim watches
    for: a router advertising its own name offers callers a `model_id` no voice
    anywhere answers to.
    """

    def naming_itself(answer: Answer) -> Answer:
        if answer.path == "/v1/models":
            listed = json.loads(answer.body)
            return _rewritten(
                answer,
                [*listed, {"model_id": "router", "languages": [], "capabilities": []}],
            )
        return answer

    with serving(tampering(naming_itself)) as base_url:
        verdict = _verdicts(base_url)["MOD-2"]

    assert verdict.word == "broken"
    assert "'router'" in verdict.why


def test_a_models_listing_inventing_an_id_no_voice_answers_breaks_mod_2() -> None:
    """The union half, which the `router` check above would never reach.

    Everything else about the invented entry is made to agree — its languages and
    its capabilities — so `MOD-7` and `CAP-5` stay held and this claim is the only
    one with anything to say.
    """

    def inventing(answer: Answer) -> Answer:
        if answer.path == "/v1/models":
            listed = json.loads(answer.body)
            return _rewritten(
                answer,
                [
                    *listed,
                    {
                        "model_id": "eleven_invented_v9",
                        "languages": [],
                        "capabilities": ["speed", "timestamps"],
                    },
                ],
            )
        return answer

    with serving(tampering(inventing)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["MOD-2"].word == "broken"
    assert "invents" in verdicts["MOD-2"].why
    assert verdicts["MOD-7"].word == "held"
    assert verdicts["CAP-5"].word == "held"


def test_a_voices_own_model_reported_ignored_breaks_mod_3() -> None:
    """The header lying in the one direction rule 2 cannot afford.

    The id chose the engine that spoke, so reporting it ignored tells a caller
    their `model_id` reached nothing when it reached exactly what they named.
    """

    def disowning(answer: Answer) -> Answer:
        sent = answer.sent.get("model_id")
        if _spoke(answer) and sent not in (None, prober.UNMAPPED_MODEL):
            return _also_ignored(answer, "model_id")
        return answer

    with serving(tampering(disowning)) as base_url:
        verdict = _verdicts(base_url)["MOD-3"]

    assert verdict.word == "broken"
    assert "reported model_id ignored" in verdict.why


def test_a_router_answers_every_claim_a_direct_engine_does_and_one_more() -> None:
    """The property `piper-conformance-e16` calls the one most worth testing.

    One prober run, two deployments, the same verdicts — that identity is the
    whole reason the per-voice claims are read per voice, because behind a router
    the voices in one process come from different engines.

    `MOD-4` is the one claim that moves, and it moves the right way: a direct
    engine leaves it unasked, because every `model_id` such a deployment publishes
    is answered by every voice it offers and no observable id names an engine that
    is not speaking. A router fronting two engines is the deployment where that
    stops being true, so the claim becomes askable here and nowhere else — which
    is what keeps its `unasked` on a direct engine readable as a fact about the
    deployment rather than about a probe nothing can ever ask.
    """
    piper = tuple(
        replace(voice, id=f"piper-{voice.id}", models=frozenset({"piper"}))
        for voice in DECLARED_VOICES
    )
    kokoro = tuple(
        replace(
            voice, id=f"kokoro-{voice.id}", models=frozenset({"kokoro"}), language="es"
        )
        for voice in DECLARED_VOICES
    )

    with cluster(
        ("piper", piper, frozenset(Capability)),
        ("kokoro", kokoro, frozenset(Capability)),
    ) as consul_url:
        with serving(router_app(consul_url)) as base_url:
            verdicts = _verdicts(base_url)

    words = {name: verdict.word for name, verdict in verdicts.items()}
    assert "broken" not in words.values(), words
    assert words["MOD-4"] == "held", verdicts["MOD-4"]
    assert "422" in verdicts["MOD-4"].evidence


def test_an_engines_id_served_instead_of_refused_breaks_mod_4() -> None:
    """A caller asks for one engine, hears another, and is told nothing.

    The id here names no engine this build knows, so the deployment serves it and
    reports it ignored — correct for an id that names nothing, and wrong for one
    its own listing says another voice answers to. That is the disagreement
    `MOD-4` reads.
    """

    def claiming_an_engine_it_lacks(answer: Answer) -> Answer:
        if answer.path == "/v1/voices":
            listing = json.loads(answer.body)
            listing["voices"][1]["models"] = ["no-such-engine"]
            return _rewritten(answer, listing)
        return answer

    with serving(tampering(claiming_an_engine_it_lacks)) as base_url:
        verdict = _verdicts(base_url)["MOD-4"]

    assert verdict.word == "broken"
    assert "'no-such-engine'" in verdict.why


def test_an_unmapped_model_served_without_a_word_breaks_mod_5() -> None:
    """Every stock ElevenLabs client sends a model this service never mapped.

    Serving it is right; serving it silently is not — the caller is told the id
    they named was honoured when it steered nothing at all.
    """

    def silently(answer: Answer) -> Answer:
        if _spoke(answer) and answer.sent.get("model_id") == prober.UNMAPPED_MODEL:
            return _not_ignored(answer, "model_id")
        return answer

    with serving(tampering(silently)) as base_url:
        verdict = _verdicts(base_url)["MOD-5"]

    assert verdict.word == "broken"
    assert "steered nothing" in verdict.why


def test_a_model_id_that_moves_the_speaker_breaks_mod_6() -> None:
    """`voice_id` decides who speaks; `model_id` is only read against that.

    A server letting the model choose the speaker answers a caller who named a
    voice in a different one, which no header here reports as a substitution
    because the deployment does not think it made one.
    """

    def steering(answer: Answer) -> Answer:
        if _spoke(answer) and "model_id" in answer.sent:
            spoke = answer.headers.get("x-elvenspeak-voice")
            return replace(
                answer,
                headers={
                    **answer.headers,
                    "x-elvenspeak-voice": SECOND if spoke == FIRST else FIRST,
                },
            )
        return answer

    with serving(tampering(steering)) as base_url:
        verdict = _verdicts(base_url)["MOD-6"]

    assert verdict.word == "broken"
    assert "model_id moved the speaker" in verdict.why


def test_a_model_advertising_a_language_its_voices_cannot_say_breaks_mod_7() -> None:
    """A caller choosing a model by language reaches a voice that cannot say it."""

    def overselling(answer: Answer) -> Answer:
        if answer.path == "/v1/models":
            listed = json.loads(answer.body)
            listed[0]["languages"] = [{"language_id": "zz", "name": "zz"}]
            return _rewritten(answer, listed)
        return answer

    with serving(tampering(overselling)) as base_url:
        verdict = _verdicts(base_url)["MOD-7"]

    assert verdict.word == "broken"
    assert "advertises ['zz']" in verdict.why


# ----------------------------------------- the ten per-voice capability claims


def test_a_deployment_whose_voices_declare_nothing_swaps_both_capability_arms() -> None:
    """The control for `CAP-2` and `CAP-4`, and the property no lie can show.

    Four claims are two questions asked of two kinds of voice, and which pair has
    a subject is decided by the voices rather than by the prober. A mutation sweep
    cannot see that: every lie below is told to whichever arm this deployment
    already selected, so nothing else in this file would notice a prober that had
    quietly stopped asking the other two. Asked here as one run because the four
    verdicts have to move *together* — `CAP-1` held and `CAP-2` held at once would
    mean both arms found a subject in one voice, which is the contradiction the
    split exists to make impossible.

    `AUTH-2` is named rather than left to the sweep because this deployment is
    what found the defect it now guards: the 501 `CAP-4` *requires* of a voice
    that cannot report timings was read by `AUTH-2` as a keyless caller turned
    away, so one file's two claims demanded opposite answers to one request and a
    deployment honouring nothing beyond plain speech — a shape `_PUBLISHED_FIELDS`
    calls legitimate — was reported broken for conforming.
    """
    with serving(tampering(lambda answer: answer, _plain_spoken())) as base_url:
        words = _words(base_url)

    assert words["CAP-1"] == "unasked"
    assert words["CAP-2"] == "held"
    assert words["CAP-3"] == "unasked"
    assert words["CAP-4"] == "held"
    assert words["AUTH-2"] == "held"
    assert "broken" not in words.values(), words


def test_an_undocumented_501_still_breaks_auth_2() -> None:
    """The discriminator in `AUTH-2`'s exemption, without which it is a hole.

    `CAP-4` requires a 501 from a voice that cannot report timings, so `AUTH-2`
    cannot read every 501 as a keyless caller turned away — but exempting the
    *status* would let a deployment guard its expensive endpoints behind an
    undocumented 501 and pass the claim whole. The published sentence is what
    separates the two, so this is that status carried without it.
    """

    def guarding(answer: Answer) -> Answer:
        if answer.path.endswith("/stream/with-timestamps"):
            return replace(
                _rewritten(answer, {"detail": "Not Implemented"}), status=501
            )
        return answer

    with serving(tampering(guarding)) as base_url:
        verdict = _verdicts(base_url)["AUTH-2"]

    assert verdict.word == "broken"
    assert "answered 501 to a keyless request" in verdict.why


def test_a_rate_that_reaches_no_engine_breaks_cap_1() -> None:
    """The defect the whole per-voice pass is priced for.

    A deployment can publish `speed`, report it honoured, and hand back the same
    audio every time; every header says the rate arrived and nothing in the
    response contradicts it. Only the byte count gives it away, so the lie told
    here changes nothing *but* the byte count — the rate is accepted, unreported
    as ignored, and simply does not shorten the utterance.
    """

    def unhurried(answer: Answer) -> Answer:
        if _spoke(answer) and _setting(answer, "speed") == prober.FASTER:
            return replace(answer, body=answer.body * 2)
        return answer

    with serving(tampering(unhurried)) as base_url:
        verdict = _verdicts(base_url)["CAP-1"]

    assert verdict.word == "broken"
    assert "the rate reached no engine" in verdict.why


def test_a_declared_rate_reported_ignored_breaks_cap_1() -> None:
    """`CAP-1`'s other half, which only a routed deployment reaches in the field.

    A backend with the capability withheld reports `voice_settings.speed` ignored
    while the listing that reached the caller still declares it, so the two
    disagree in the one direction that matters: a caller read the listing to
    decide whether to send a rate and was told the opposite of what happened. The
    audio is left alone here, so the byte-count arm above stays quiet and this is
    the only thing the verdict can be about.
    """

    def withholding(answer: Answer) -> Answer:
        if _spoke(answer) and _setting(answer, "speed") == prober.FASTER:
            return _also_ignored(answer, "voice_settings.speed")
        return answer

    with serving(tampering(withholding)) as base_url:
        verdict = _verdicts(base_url)["CAP-1"]

    assert verdict.word == "broken"
    assert "reported voice_settings.speed ignored" in verdict.why


def test_a_rate_dropped_without_a_word_breaks_cap_2() -> None:
    """README:24's rule 2 exactly inverted, and the quiet way it fails.

    A voice that never declared `speed` is allowed to ignore one — what it may not
    do is accept the request, discard the rate, and answer 200 with nothing said,
    because the caller then believes their whole request was honoured.
    """

    def wordless(answer: Answer) -> Answer:
        if _spoke(answer) and _setting(answer, "speed") == prober.FASTER:
            return _not_ignored(answer, "voice_settings.speed")
        return answer

    with serving(tampering(wordless, _plain_spoken())) as base_url:
        verdict = _verdicts(base_url)["CAP-2"]

    assert verdict.word == "broken"
    assert "the rate was dropped without a word" in verdict.why


def test_an_empty_alignment_from_the_streaming_endpoint_breaks_cap_3() -> None:
    """Only the streaming half is emptied, which is what makes this worth a test.

    A deployment answering `/with-timestamps` with real timings and the streaming
    endpoint with an alignment-shaped hole satisfies every check that reads a
    status and half of one that reads the body. `CAP-3` is a claim about *both*
    endpoints, and lying on only the second is the way to find out whether it
    really asks both — the endpoint a caller reaches for under load is the one a
    prober is most likely to have skipped.
    """

    def hollow(answer: Answer) -> Answer:
        if not answer.path.endswith("/stream/with-timestamps"):
            return answer
        lines = answer.body.splitlines()
        first = json.loads(lines[0])
        first["alignment"]["characters"] = []
        return replace(
            answer, body=b"\n".join([json.dumps(first).encode(), *lines[1:]])
        )

    with serving(tampering(hollow)) as base_url:
        verdict = _verdicts(base_url)["CAP-3"]

    assert verdict.word == "broken"
    assert "/stream/with-timestamps" in verdict.why
    assert "an empty `characters` array" in verdict.why


def test_a_timestamp_refusal_that_is_not_the_published_sentence_breaks_cap_4() -> None:
    """A bare 501 is the wrong refusal even though it is the right status.

    The status alone cannot tell a caller a route this build does not serve from a
    voice that cannot report timings, and only the second is a reason to try
    another voice. So the lie keeps the 501 and changes nothing but the sentence.
    """

    def mumbling(answer: Answer) -> Answer:
        if answer.status == 501:
            return _rewritten(answer, {"detail": "Not Implemented"})
        return answer

    with serving(tampering(mumbling, _plain_spoken())) as base_url:
        verdict = _verdicts(base_url)["CAP-4"]

    assert verdict.word == "broken"
    assert "rather than the published" in verdict.why


def test_a_models_listing_overstating_the_union_breaks_cap_5() -> None:
    """Rule 2 kept per voice and broken per deployment.

    A caller reads `GET /v1/models` to decide whether to send a rate at all, so a
    union that overstates what the voices declare sends them to a voice that
    reports the parameter ignored — every per-voice claim holds and the caller was
    still misled.
    """

    def overselling(answer: Answer) -> Answer:
        if answer.path == "/v1/models":
            listed = json.loads(answer.body)
            for entry in listed:
                entry["capabilities"] = [*entry["capabilities"], "cloning"]
            return _rewritten(answer, listed)
        return answer

    with serving(tampering(overselling)) as base_url:
        verdict = _verdicts(base_url)["CAP-5"]

    assert verdict.word == "broken"
    assert "advertises capabilities" in verdict.why


def test_a_modelled_setting_dropped_without_a_word_breaks_cap_6() -> None:
    """The half of rule 2 that `REF-7` cannot reach.

    `REF-7` sends a field nobody modelled, so a deployment reporting only what its
    parser could not place keeps that claim and breaks this one. The setting
    dropped here is one this service *models* and no engine honours, which is
    exactly the gap between the two.
    """

    def wordless(answer: Answer) -> Answer:
        if _spoke(answer) and _setting(answer, prober.UNHONOURABLE_SETTING) is not None:
            return _not_ignored(answer, f"voice_settings.{prober.UNHONOURABLE_SETTING}")
        return answer

    with serving(tampering(wordless)) as base_url:
        verdict = _verdicts(base_url)["CAP-6"]

    assert verdict.word == "broken"
    assert "was dropped without a word" in verdict.why


def test_an_empty_ignored_header_where_nothing_was_dropped_breaks_cap_7() -> None:
    """Absent and empty are different answers, and the difference is the claim.

    A caller reading the header's *presence* as "something was dropped" is reading
    it exactly as README:22 documents it, so an empty header sends them hunting
    for a parameter nobody dropped. The lie names nothing — it only makes the
    header exist on the one request in the whole run that had nothing to report.
    """

    def hinting(answer: Answer) -> Answer:
        if (
            _spoke(answer)
            and set(answer.sent) == {"text"}
            and not answer.path.endswith("with-timestamps")
        ):
            return replace(
                answer, headers={**answer.headers, "x-elvenspeak-ignored": ""}
            )
        return answer

    with serving(tampering(hinting)) as base_url:
        verdict = _verdicts(base_url)["CAP-7"]

    assert verdict.word == "broken"
    assert "the header is present where nothing was dropped" in verdict.why


def test_an_unspeakable_language_dropped_without_a_word_breaks_cap_8() -> None:
    """The arm that catches a server silently ignoring `language_code`.

    Keyed on [`prober.UNSPOKEN_LANGUAGES`] rather than on a tag spelled here,
    because which family no voice speaks is the prober's fact and a second
    spelling of it would go green against a prober that picked a different one.
    """

    def wordless(answer: Answer) -> Answer:
        unspoken = answer.sent.get("language_code") in prober.UNSPOKEN_LANGUAGES
        if _spoke(answer) and unspoken:
            return _not_ignored(answer, "language_code")
        return answer

    with serving(tampering(wordless)) as base_url:
        verdict = _verdicts(base_url)["CAP-8"]

    assert verdict.word == "broken"
    assert "the preference was dropped and the caller was not told" in verdict.why


def test_an_honoured_language_reported_ignored_breaks_cap_8() -> None:
    """The other arm, and the reason one alone is half a verdict.

    A header naming every `language_code` satisfies the arm above exactly as well
    as a correct one does, so `CAP-8` asks both per voice: the tag this voice
    really speaks must come back *not* reported as dropped. Without this arm a
    deployment naming every language it is ever sent would read as conformant.
    """
    spoken = DECLARED_VOICES[0].language

    def grumbling(answer: Answer) -> Answer:
        if _spoke(answer) and answer.sent.get("language_code") == spoken:
            return _also_ignored(answer, "language_code")
        return answer

    with serving(tampering(grumbling)) as base_url:
        verdict = _verdicts(base_url)["CAP-8"]

    assert verdict.word == "broken"
    assert "a preference this voice met was reported as dropped" in verdict.why


def test_an_empty_language_reported_ignored_breaks_cap_9() -> None:
    """The bug this claim was written from, told back to the prober.

    `""` is what a form or a JS client sends for "unset"; taken literally it is a
    language no voice speaks, so every such request came back reporting
    `language_code` dropped and a caller who expressed no preference was told
    their preference was ignored.
    """

    def literal(answer: Answer) -> Answer:
        if _spoke(answer) and answer.sent.get("language_code") == "":
            return _also_ignored(answer, "language_code")
        return answer

    with serving(tampering(literal)) as base_url:
        verdict = _verdicts(base_url)["CAP-9"]

    assert verdict.word == "broken"
    assert prober.EMPTY_LANGUAGE in verdict.why
    assert "there was nothing to report as dropped" in verdict.why


def test_a_region_tagged_variant_reported_ignored_breaks_cap_10() -> None:
    """A server comparing raw strings, which is invisible to a canonical spelling.

    The bare family is honoured and the region-tagged upper-cased spelling of the
    same language is reported dropped — the exact signature of a comparison made
    before the tag was reduced, and a failure no check sending only `en` can see.
    """

    def unreduced(answer: Answer) -> Answer:
        if _spoke(answer) and str(answer.sent.get("language_code", "")).endswith(
            "_419"
        ):
            return _also_ignored(answer, "language_code")
        return answer

    with serving(tampering(unreduced)) as base_url:
        verdict = _verdicts(base_url)["CAP-10"]

    assert verdict.word == "broken"
    assert "the tag was compared before it was reduced to its family" in verdict.why


# --------------------------- the claims that require a draw to be served


#: Each of these promises a draw is *served*, and until `e16.5`'s review they read
#: it through [`prober.Asked.spoke`] alone, which cannot tell a draw the prober
#: could not compose from one the deployment refused. Both left the claim
#: `unasked` — so a voice declaring a capability and then refusing the very
#: endpoint it promises, the defect these claims exist to catch, was reported as
#: something the prober could not look at. [`prober.Asked.refusal`] is the reading
#: that separates them, and each case below refuses exactly the draw its claim
#: requires and asks for `broken`: the deployment answered, and its answer is the
#: promise not kept.


def _refusing(answer: Answer) -> Answer:
    """`answer`, turned into a refusal of a request this deployment promised."""
    return replace(answer, status=503, body=b'{"detail": "this voice is not speaking"}')


def test_a_declared_rate_refused_outright_breaks_cap_1() -> None:
    """A voice that declares `speed` and then will not be asked at one.

    The audio arm cannot see this: there is no length to read against another, so
    reading it through `spoke` alone reported a deployment contradicting its own
    listing as one that could not be asked.
    """

    def striking(answer: Answer) -> Answer:
        if _spoke(answer) and _setting(answer, "speed") == prober.FASTER:
            return _refusing(answer)
        return answer

    with serving(tampering(striking)) as base_url:
        verdict = _verdicts(base_url)["CAP-1"]

    assert verdict.word == "broken"
    assert f"declares 'speed' and then answered {prober.FASTER_DRAW} with 503" in verdict.why


def test_a_rate_refused_by_a_voice_that_never_declared_it_breaks_cap_2() -> None:
    """`CAP-2`'s arm of the same defect, which its own voices select.

    A voice omitting `speed` owes the caller the audio and the word that the rate
    was dropped. Refusing the request is neither, and it is a stricter failure
    than the one this claim was written for: the caller gets nothing at all.
    """

    def striking(answer: Answer) -> Answer:
        if _spoke(answer) and _setting(answer, "speed") == prober.FASTER:
            return _refusing(answer)
        return answer

    with serving(tampering(striking, _plain_spoken())) as base_url:
        verdict = _verdicts(base_url)["CAP-2"]

    assert verdict.word == "broken"
    assert f"omits 'speed' and then answered {prober.FASTER_DRAW} with 503" in verdict.why


def test_a_declared_timestamp_endpoint_that_refuses_breaks_cap_3() -> None:
    """The case the review named, and the one nothing else here catches.

    `CAP-4` is the claim about a voice that refuses timings, and its subject is
    the voices *omitting* the capability — so a voice declaring `timestamps` and
    refusing both endpoints falls between the two arms unless this one reports it.
    """

    def silent(answer: Answer) -> Answer:
        if answer.path.endswith("with-timestamps"):
            return _refusing(answer)
        return answer

    with serving(tampering(silent)) as base_url:
        verdict = _verdicts(base_url)["CAP-3"]

    assert verdict.word == "broken"
    assert "declares 'timestamps' and then answered POST /with-timestamps with 503" in verdict.why
    assert "which is CAP-4's answer" in verdict.why


def test_a_voices_own_model_refused_breaks_mod_3() -> None:
    """An id a voice publishes as its own and will not answer to.

    A caller reading the listing to choose a model is handed one that cannot be
    used, which is the same lie as reporting it ignored and a louder one: the
    request does not survive at all.
    """

    def disowning(answer: Answer) -> Answer:
        sent = answer.sent.get("model_id")
        if _spoke(answer) and sent not in (None, prober.UNMAPPED_MODEL):
            return _refusing(answer)
        return answer

    with serving(tampering(disowning)) as base_url:
        verdict = _verdicts(base_url)["MOD-3"]

    assert verdict.word == "broken"
    assert f"among its own models and then answered {prober.LISTED_MODEL} with 503" in verdict.why


def test_a_model_id_that_ends_the_request_breaks_mod_6() -> None:
    """The ranking inverted rather than merely unobservable.

    `MOD-6` is the claim that `voice_id` decides who speaks and `model_id` is only
    read against it. A deployment that refuses over an id naming no engine let
    that id decide the whole outcome, which is the ranking the wrong way round
    rather than a draw the prober failed to make.
    """

    def vetoing(answer: Answer) -> Answer:
        if _spoke(answer) and answer.sent.get("model_id") == prober.UNMAPPED_MODEL:
            return _refusing(answer)
        return answer

    with serving(tampering(vetoing)) as base_url:
        verdict = _verdicts(base_url)["MOD-6"]

    assert verdict.word == "broken"
    assert f"was addressed and then answered {prober.UNMAPPED_DRAW} with 503" in verdict.why
    assert "outranked the voice" in verdict.why


def test_a_published_alias_refused_outright_breaks_sub_5() -> None:
    """An alias this deployment lists and then will not answer to at all.

    The mis-routing arm cannot see this one: a refused request reached no voice,
    so there is no id to read against the one that published the alias, and
    reading it through `spoke` alone reported a listing contradicting itself as
    something that could not be asked. `SUB-5` promises the alias *reaches* the
    voice, which is the same promise `MOD-3` makes about a published model id.
    """

    def disowning(answer: Answer) -> Answer:
        if _spoke(answer) and _sent_to(answer) == "eleven-legacy-id":
            return _refusing(answer)
        return _publishing(answer, "eleven-legacy-id", FIRST)

    with serving(tampering(disowning)) as base_url:
        verdict = _verdicts(base_url)["SUB-5"]

    assert verdict.word == "broken"
    assert f"is published on voice {FIRST!r} and then answered " in verdict.why
    assert f"{prober._alias_draw('eleven-legacy-id')} with 503" in verdict.why


def test_a_setting_no_engine_honours_refused_breaks_cap_6() -> None:
    """README:24's rule 2 is a promise about a request that is *served*.

    A parameter this service cannot honour is dropped and named back in
    `x-elvenspeak-ignored`, so a deployment refusing over one hands the caller
    neither the audio nor the word. Read through `spoke` alone that was reported
    as a draw this claim could not read.
    """

    def strict(answer: Answer) -> Answer:
        if _spoke(answer) and _setting(answer, prober.UNHONOURABLE_SETTING) is not None:
            return _refusing(answer)
        return answer

    with serving(tampering(strict)) as base_url:
        verdict = _verdicts(base_url)["CAP-6"]

    assert verdict.word == "broken"
    assert f"answered {prober.UNHONOURABLE_DRAW} with 503" in verdict.why
    assert "rule 2" in verdict.why


def test_an_unspoken_language_refused_breaks_cap_8() -> None:
    """The same rule read over a language rather than over a voice setting.

    A language the addressed voice does not speak is a preference to drop and
    report, so refusing it is rule 2 broken. This claim sweeps every voice and
    `_of_every` lets a `Blocked` out of one discard the whole sweep, so before
    this the first refusing voice took every other voice's verdict with it.
    """

    def monolingual(answer: Answer) -> Answer:
        if _spoke(answer) and answer.sent.get("language_code") in prober.UNSPOKEN_LANGUAGES:
            return _refusing(answer)
        return answer

    with serving(tampering(monolingual)) as base_url:
        verdict = _verdicts(base_url)["CAP-8"]

    assert verdict.word == "broken"
    assert f"answered {prober.UNSPOKEN_DRAW} with 503" in verdict.why
    assert "never one to refuse over" in verdict.why


def test_an_empty_language_refused_breaks_cap_9() -> None:
    """The literal reading this claim was written from, made louder.

    `""` is what a form or a JS client sends for "unset". A deployment reporting
    it dropped has taken it for a value, which is this claim's own bug; one that
    refuses over it has done the same thing and denied the caller audio as well.
    `PLAIN` is deliberately not read this way — it carries no `language_code` at
    all, so refusing it says nothing about how an unset tag is read and is
    `FMT-1`'s subject instead.
    """

    def rejecting(answer: Answer) -> Answer:
        if _spoke(answer) and answer.sent.get("language_code") == "":
            return _refusing(answer)
        return answer

    with serving(tampering(rejecting)) as base_url:
        verdict = _verdicts(base_url)["CAP-9"]

    assert verdict.word == "broken"
    assert f"answered {prober.EMPTY_LANGUAGE} with 503" in verdict.why
    assert "for no preference at all" in verdict.why


def test_a_voices_own_language_refused_breaks_cap_8() -> None:
    """A voice that will not answer to the language its own listing names.

    `CAP-8`'s second arm asks each voice for exactly that language. Refusing it is
    the lie `MOD-3` catches over a published model id and `SUB-5` over a published
    alias — the listing offers something the voice does not honour — and this arm
    was not one the adversarial review named.
    """

    def disowning(answer: Answer) -> Answer:
        lang = answer.sent.get("language_code")
        own = (
            isinstance(lang, str)
            and lang.strip() != ""
            and lang == lang.lower()
            and "_" not in lang
            and lang not in prober.UNSPOKEN_LANGUAGES
        )
        if _spoke(answer) and own:
            return _refusing(answer)
        return answer

    with serving(tampering(disowning)) as base_url:
        verdict = _verdicts(base_url)["CAP-8"]

    assert verdict.word == "broken"
    assert "was asked for exactly that" in verdict.why
    assert f"answered {prober.OWN_FAMILY} with 503" in verdict.why


def test_a_region_tagged_spelling_refused_breaks_cap_10() -> None:
    """A spelling this deployment should have reduced, rejected instead.

    `CAP-10`'s subject is that the mangled tag and the bare family are one
    request. A deployment refusing the region-tagged spelling has read it as a
    value it may reject rather than as a tag to reduce — this claim broken more
    loudly than the reported-ignored case it was written for.
    """

    def literal(answer: Answer) -> Answer:
        lang = answer.sent.get("language_code")
        if _spoke(answer) and isinstance(lang, str) and lang.endswith("_419"):
            return _refusing(answer)
        return answer

    with serving(tampering(literal)) as base_url:
        verdict = _verdicts(base_url)["CAP-10"]

    assert verdict.word == "broken"
    assert f"answered {prober.OWN_VARIANT} with 503" in verdict.why
    assert "rather than to refuse over" in verdict.why


# ------------------------------------------- the six substitution claims


def test_a_substitution_that_does_not_say_what_was_asked_for_breaks_sub_1() -> None:
    """A substitution a caller cannot detect is the failure `SUB-1` exists for.

    The audio is real and the voice that spoke is named; only the record of what
    was *asked for* is missing, which is precisely the state where a caller
    believes they were answered by the voice they addressed. `SUB-2` sees this
    lie too — every lie about that header does, since the two claims read the same
    pair — so it is asserted here rather than left to look like an accident.
    """

    def unrecorded(answer: Answer) -> Answer:
        if _sent_to(answer) == prober.FOREIGN_VOICE:
            return replace(
                answer,
                headers={
                    name: value
                    for name, value in answer.headers.items()
                    if name.lower() != "x-elvenspeak-voice-requested"
                },
            )
        return answer

    with serving(tampering(unrecorded)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["SUB-1"].word == "broken"
    assert "was not what spoke" in verdicts["SUB-1"].why
    assert verdicts["SUB-2"].word == "broken"


def test_a_voice_requested_header_on_a_direct_hit_breaks_sub_2() -> None:
    """`SUB-2`'s own arm, told where `SUB-1` has nothing to read.

    The header is added to answers that substituted nothing, so `SUB-1` — which
    reads only the request addressed at a voice nothing offers — stays held and
    this is the only claim with anything to say. A `voice-requested` on every
    response tells a caller nothing, which is the same defect as one that never
    appears: the header stops meaning a substitution happened.
    """

    def crying_wolf(answer: Answer) -> Answer:
        if _spoke(answer) and _sent_to(answer) == FIRST:
            return replace(
                answer,
                headers={**answer.headers, "x-elvenspeak-voice-requested": FIRST},
            )
        return answer

    with serving(tampering(crying_wolf)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["SUB-2"].word == "broken"
    assert "the header no longer means a substitution happened" in verdicts["SUB-2"].why
    assert verdicts["SUB-1"].word == "held"


def test_a_substitute_the_listing_does_not_offer_breaks_sub_3() -> None:
    """A voice a caller can hear and never address again.

    The fifth of the five claims a router misreporting its fleet passes: the
    substitution is reported honestly in every other respect, and the voice it
    names simply is not in `GET /v1/voices`, which makes what the caller heard
    unreproducible.
    """

    def ghostly(answer: Answer) -> Answer:
        if _spoke(answer):
            return replace(
                answer,
                headers={**answer.headers, "x-elvenspeak-voice": "ghost-voice"},
            )
        return answer

    with serving(tampering(ghostly)) as base_url:
        verdict = _verdicts(base_url)["SUB-3"]

    assert verdict.word == "broken"
    assert "GET /v1/voices does not offer" in verdict.why


def test_a_404_naming_no_voice_breaks_sub_4() -> None:
    """Refusing is allowed; refusing anonymously is not.

    A deployment with substitution switched off owes a 404, and `engine_app`
    hardcodes `Substitution.FIRST_OFFERED`, so the refusal has to be bought with a
    lie rather than with a setting. The body names nothing, which leaves a caller
    unable to tell this from a path the build does not serve.
    """

    def anonymous(answer: Answer) -> Answer:
        if _sent_to(answer) == prober.FOREIGN_VOICE:
            return replace(_rewritten(answer, {"detail": "Not Found"}), status=404)
        return answer

    with serving(tampering(anonymous)) as base_url:
        verdict = _verdicts(base_url)["SUB-4"]

    assert verdict.word == "broken"
    assert "which names no voice" in verdict.why


def test_a_404_naming_the_voice_it_refused_holds_sub_4_and_unasks_sub_1() -> None:
    """The arm selection `SUB-1` and `SUB-4` share, which neither alone can show.

    One request, two documented promises, and which of them has a subject is the
    deployment's choice rather than the prober's. A prober that had quietly
    stopped asking `SUB-4` would be indistinguishable from this one in every other
    test here — the lie is the same 404 as above with a body that names what it
    refused, and the pair must move in opposite directions together.
    """

    def named(answer: Answer) -> Answer:
        if _sent_to(answer) == prober.FOREIGN_VOICE:
            return replace(
                _rewritten(
                    answer, {"detail": f"unknown voice {prober.FOREIGN_VOICE!r}"}
                ),
                status=404,
            )
        return answer

    with serving(tampering(named)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["SUB-4"].word == "held"
    assert verdicts["SUB-1"].word == "unasked"
    assert "SUB-4 is the arm it selects" in verdicts["SUB-1"].blocker


def test_a_deployment_publishing_no_alias_leaves_sub_5_unasked() -> None:
    """Zero aliases all resolving correctly is a check that cannot fail.

    `docs/conformance-claims.md` said an empty alias set made this claim hold
    trivially, which contradicted the same document's rejection of checks that
    cannot fail and the `HEALTH-2` precedent already in `prober.py`;
    `piper-conformance-e16.5` settled it in the document's favour of the code, and
    this is the assertion that keeps the two agreeing.
    """
    with serving(tampering(lambda answer: answer)) as base_url:
        verdict = _verdicts(base_url)["SUB-5"]

    assert verdict.word == "unasked"
    assert "no voice here publishes an alias" in verdict.blocker


def test_an_alias_that_reaches_another_voice_breaks_sub_5() -> None:
    """An alias table is exactly the kind of map that drifts.

    The alias is published on the second voice, where an unknown id does *not*
    land — this deployment substitutes to the first offered one — so the listing
    promises a mapping the deployment does not make. Published on the first voice
    the same lie would hold, which is what the assertion on `SUB-5`'s held
    wording below the break is for: the ids are foreign by definition and nothing
    but this check ever reads them.
    """
    with serving(
        tampering(lambda answer: _publishing(answer, "eleven-legacy-id", SECOND))
    ) as base_url:
        verdict = _verdicts(base_url)["SUB-5"]

    assert verdict.word == "broken"
    assert "the listing promises a mapping this deployment does not make" in verdict.why


def test_an_alias_reaching_the_voice_that_published_it_holds_sub_5() -> None:
    """The same lie moved one voice over, so the break above is about the mapping.

    Without this, `SUB-5` breaking on a published alias would be equally
    consistent with a probe that reports `broken` for *any* alias it finds.
    """
    with serving(
        tampering(lambda answer: _publishing(answer, "eleven-legacy-id", FIRST))
    ) as base_url:
        verdict = _verdicts(base_url)["SUB-5"]

    assert verdict.word == "held"
    assert "reached the voice that published it" in verdict.evidence


def test_an_unprintable_voice_id_answered_500_breaks_sub_6() -> None:
    """The id reaching the response headers unescaped, which is a 500 on the way out.

    A header value must be latin-1 on the wire, so a server echoing this id
    unencoded raises inside its own response assembly — after the status is
    chosen — and fails a request it had already decided to serve. The lie is that
    500, told to a request the real server answers correctly.
    """

    def unescaped(answer: Answer) -> Answer:
        if _sent_to(answer) == prober.UNPRINTABLE_VOICE:
            return replace(
                _rewritten(answer, {"detail": "Internal Server Error"}), status=500
            )
        return answer

    with serving(tampering(unescaped)) as base_url:
        verdict = _verdicts(base_url)["SUB-6"]

    assert verdict.word == "broken"
    assert "reached the response headers unescaped" in verdict.why


# ------------------------------------------- the timestamp endpoints' body shape
#
# Each lie below leaves a real deployment answering real timings and changes one
# thing about the shape, because that is the failure worth catching: a body that
# is absent or unparseable breaks these claims through `_timed` and would prove
# only that the parser runs. The control is the same real server, which the live
# run and [`test_an_open_deployment_breaks_nothing`] establish holds all five.


def _timestamped(answer: Answer) -> list[dict[str, Any]]:
    """Every timestamped object `answer` carries, parsed."""
    return [json.loads(line) for line in answer.body.splitlines()]


def _relined(answer: Answer, objects: list[dict[str, Any]]) -> Answer:
    """`answer`, carrying `objects` one per line in place of what it held."""
    return replace(
        answer, body=b"\n".join(json.dumps(one).encode() for one in objects)
    )


#: The two per-character timelines an alignment publishes, spelled out rather
#: than read off `prober` — a test that took these names from the thing under
#: test could not notice it renaming one ([LAW:behavior-not-structure]).
_TIME_ARRAYS = ("character_start_times_seconds", "character_end_times_seconds")


def _streamed(answer: Answer) -> bool:
    """Whether `answer` came from the streaming timestamp endpoint."""
    return answer.path.endswith(prober.STREAMED_TIMESTAMPS)


def _plain_timestamped(answer: Answer) -> bool:
    """Whether `answer` came from the non-streaming timestamp endpoint."""
    return answer.path.endswith(prober.WITH_TIMESTAMPS) and not _streamed(answer)


def test_a_with_timestamps_answer_carrying_no_fidelity_header_breaks_time_2() -> None:
    """The timings survive; only the header saying what they are worth is dropped.

    A caller handed word-exact timings and a caller handed a span spread evenly
    over characters receive the same floats, and `x-elvenspeak-alignment` is the
    only place the non-streaming endpoint can tell them apart. So the lie keeps
    every number and removes the one value that says what the numbers mean.
    """

    def unlabelled(answer: Answer) -> Answer:
        if not _plain_timestamped(answer):
            return answer
        return replace(
            answer,
            headers={
                name: value
                for name, value in answer.headers.items()
                if name.lower() != prober.ALIGNMENT_HEADER
            },
        )

    with serving(tampering(unlabelled)) as base_url:
        verdict = _verdicts(base_url)["TIME-2"]

    assert verdict.word == "broken"
    assert f"carrying no {prober.ALIGNMENT_HEADER} header" in verdict.why


def test_a_fidelity_header_outside_the_published_two_breaks_time_2() -> None:
    """A header that is present and means nothing, which absence alone cannot catch.

    Without this, `TIME-2` would be satisfied by any deployment that sends the
    header at all — and a value outside the published two is exactly as unusable
    to a caller branching on it as no value.
    """

    def invented(answer: Answer) -> Answer:
        if not _plain_timestamped(answer):
            return answer
        return replace(
            answer,
            headers={**answer.headers, prober.ALIGNMENT_HEADER: "approximate"},
        )

    with serving(tampering(invented)) as base_url:
        verdict = _verdicts(base_url)["TIME-2"]

    assert verdict.word == "broken"
    assert "'approximate', which is neither of the two values" in verdict.why


def test_a_fidelity_header_on_the_streaming_endpoint_breaks_time_3() -> None:
    """The header is wrong *because* it is on this endpoint, not because it is odd.

    `api.py:998` withholds it deliberately: fidelity is settled per sentence and
    can differ between the objects of one response, so a single header reports at
    most one of several answers and would have to be sent before any of them were
    known. The lie sends a value that is perfectly legal on the other endpoint.
    """

    def headered(answer: Answer) -> Answer:
        if not _streamed(answer):
            return answer
        return replace(
            answer,
            headers={**answer.headers, prober.ALIGNMENT_HEADER: "word-exact"},
        )

    with serving(tampering(headered)) as base_url:
        verdict = _verdicts(base_url)["TIME-3"]

    assert verdict.word == "broken"
    assert f"carrying {prober.ALIGNMENT_HEADER}" in verdict.why


def test_a_streamed_object_whose_fidelity_is_unpublished_breaks_time_3() -> None:
    """The other half of `TIME-3`: each object must say what its own timings are worth.

    Only the second object is spoiled, so a probe reading the first and reporting
    on the rest cannot pass this.
    """

    def invented(answer: Answer) -> Answer:
        if not _streamed(answer):
            return answer
        objects = _timestamped(answer)
        objects[-1]["alignment_fidelity"] = "approximate"
        return _relined(answer, objects)

    with serving(tampering(invented)) as base_url:
        verdict = _verdicts(base_url)["TIME-3"]

    assert verdict.word == "broken"
    assert "`alignment_fidelity` is not published" in verdict.why


def test_a_timeline_covering_half_its_audio_breaks_time_4() -> None:
    """Every time scaled together, so the timeline stays ordered and stops early.

    The failure this catches is a server whose timings are internally consistent
    and describe a different utterance than the one it sent — subtitles that run
    out halfway. Scaling rather than truncating is what keeps the end times
    ascending, so this cannot be passed by a probe that only checks the order.
    """

    def halved(answer: Answer) -> Answer:
        if not _plain_timestamped(answer):
            return answer
        objects = _timestamped(answer)
        for one in objects:
            for field in _TIME_ARRAYS:
                one["alignment"][field] = [
                    second * 0.5 for second in one["alignment"][field]
                ]
        return _relined(answer, objects)

    with serving(tampering(halved)) as base_url:
        verdict = _verdicts(base_url)["TIME-4"]

    assert verdict.word == "broken"
    assert "do not account for the samples they arrived with" in verdict.why


def test_an_end_time_that_goes_backwards_breaks_time_4() -> None:
    """`TIME-4`'s other half, and the half a total duration check cannot see.

    Two adjacent end times are swapped, which leaves the first and last untouched
    — so the timeline still covers exactly its own audio and only its interior
    reverses. A caller stepping through it word by word goes backwards.
    """

    def reversed_pair(answer: Answer) -> Answer:
        if not _plain_timestamped(answer):
            return answer
        objects = _timestamped(answer)
        field = "character_end_times_seconds"
        for one in objects:
            times = one["alignment"][field]
            times[1], times[2] = times[2], times[1]
        return _relined(answer, objects)

    with serving(tampering(reversed_pair)) as base_url:
        verdict = _verdicts(base_url)["TIME-4"]

    assert verdict.word == "broken"
    assert "go backwards" in verdict.why


def test_a_time_4_parse_fault_names_the_endpoint_that_sent_it() -> None:
    """`TIME-4` asks both endpoints, so a fault that names neither is half a report.

    A refusal carries its own draw, but the parser only ever saw a body — so
    without the endpoint put back, a maintainer reading "answered an `alignment`
    whose ... is not an array of numbers" cannot tell which of the two produced
    it. Only the non-streaming endpoint is spoiled here, so naming the other one
    would fail this just as silence does.
    """

    def unreadable(answer: Answer) -> Answer:
        if not _plain_timestamped(answer):
            return answer
        objects = _timestamped(answer)
        for one in objects:
            one["alignment"]["character_end_times_seconds"] = ["soon"]
        return _relined(answer, objects)

    with serving(tampering(unreadable)) as base_url:
        verdict = _verdicts(base_url)["TIME-4"]

    assert verdict.word == "broken"
    assert prober.WITH_TIMESTAMPS in verdict.why
    assert prober.STREAMED_TIMESTAMPS not in verdict.why


def test_a_later_streamed_object_whose_times_reverse_breaks_time_4() -> None:
    """`TIME-4` is asked of every object, not only the one its timeline begins at.

    The streaming endpoint is the only place that can be shown. Every other
    `TIME-4` test here tampers the non-streaming one, which the parser always
    turns into exactly one object — so all of them would pass unchanged against a
    loop narrowed to `objects[:1]`, and the claim's own reason for reading a
    stream at all is that a cumulative offset slips in the objects after the
    first. Only the last object reverses here and the first arrives untouched, so
    the `begins` reading that opens the claim still passes.
    """

    def reversed_late(answer: Answer) -> Answer:
        if not _streamed(answer):
            return answer
        objects = _timestamped(answer)
        times = objects[-1]["alignment"]["character_end_times_seconds"]
        times[1], times[2] = times[2], times[1]
        return _relined(answer, objects)

    with serving(tampering(reversed_late)) as base_url:
        verdict = _verdicts(base_url)["TIME-4"]

    assert verdict.word == "broken"
    assert "go backwards" in verdict.why


def test_a_normalized_alignment_that_is_not_the_alignment_breaks_time_5() -> None:
    """`alignment` stays real, so only a caller reading the other field is broken.

    This is the failure that passes every check pointed at `alignment` — which is
    the field a prober is most likely to have asked about — while handing the
    SDK's own documented reader an alignment covering nothing.
    """

    def hollow(answer: Answer) -> Answer:
        if not _plain_timestamped(answer):
            return answer
        objects = _timestamped(answer)
        for one in objects:
            one["normalized_alignment"] = {
                "characters": [],
                "character_start_times_seconds": [],
                "character_end_times_seconds": [],
            }
        return _relined(answer, objects)

    with serving(tampering(hollow)) as base_url:
        verdict = _verdicts(base_url)["TIME-5"]

    assert verdict.word == "broken"
    assert "`normalized_alignment` that is not its `alignment`" in verdict.why


def test_a_later_streamed_object_whose_normalization_is_hollow_breaks_time_5() -> None:
    """Both of `TIME-5`'s loops, which the endpoint above cannot reach either of.

    The test above spoils the non-streaming endpoint's only object, so a claim
    that read just `_TIMESTAMP_ENDPOINTS[0]`, or just `objects[0]`, would still
    report it `broken`. Spoiling the streamed run's last object is what makes
    both loops falsifiable — and it is the failure this claim exists for: the
    deployment that got `normalized_alignment` right on the endpoint a prober is
    most likely to ask, and wrong on the one a caller reaches for under load.
    """

    def hollow_late(answer: Answer) -> Answer:
        if not _streamed(answer):
            return answer
        objects = _timestamped(answer)
        objects[-1]["normalized_alignment"] = {
            "characters": [],
            "character_start_times_seconds": [],
            "character_end_times_seconds": [],
        }
        return _relined(answer, objects)

    with serving(tampering(hollow_late)) as base_url:
        verdict = _verdicts(base_url)["TIME-5"]

    assert verdict.word == "broken"
    assert "`normalized_alignment` that is not its `alignment`" in verdict.why


def test_streamed_objects_that_each_restart_at_zero_break_time_6() -> None:
    """Every object correct alone, and the sequence sliding against its own audio.

    The exact bug `api.py:979` carries `elapsed` forward to avoid: a server that
    aligns each sentence against its own audio and forgets the offset answers
    objects that each cover their own samples perfectly — so `TIME-4` still holds
    — and lay on top of each other rather than end to end. Only a reading that
    puts consecutive objects side by side finds it.
    """

    def restarted(answer: Answer) -> Answer:
        if not _streamed(answer):
            return answer
        objects = _timestamped(answer)
        for one in objects:
            offset = one["alignment"]["character_start_times_seconds"][0]
            for field in _TIME_ARRAYS:
                one["alignment"][field] = [
                    second - offset for second in one["alignment"][field]
                ]
        return _relined(answer, objects)

    with serving(tampering(restarted)) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["TIME-6"].word == "broken"
    assert "sliding against their own audio" in verdicts["TIME-6"].why
    assert verdicts["TIME-4"].word == "held", (
        "each object still covers exactly its own audio, so this lie must be "
        "invisible to TIME-4 — otherwise the break above is not about the seam "
        "between objects"
    )


def test_a_declared_timestamp_endpoint_that_refuses_breaks_every_time_claim() -> None:
    """A refused draw is these claims being false, never these claims being unaskable.

    The reading five review rounds of `piper-conformance-e16.5` got wrong, kept
    here because the mistake is invisible in a green run: `Asked.spoke` raises
    `Blocked` on any non-200 and `_finding` turns that into `unasked`, so a
    deployment refusing the very request a claim tests reports as one the prober
    could not look at. Every one of these draws is a promise the voice made by
    publishing `timestamps`, so a refusal is the deployment answering.

    All of them are asserted together rather than one per test because the defect
    arrives one claim at a time — five instances of it were missed across three
    rounds — and a list that must be exhaustive is what the next claim added here
    will fail against. `CAP-3` is that next claim, and it arrived by the predicted
    route: it read a refusal correctly but through its own inline copy of the
    rule, so it sat outside the list this test holds. It now routes through
    [`_refused_own_endpoint`] like the other five, which is the "six claims" that
    helper's own docstring always named.
    """
    refusing = {"CAP-3", "TIME-2", "TIME-3", "TIME-4", "TIME-5", "TIME-6"}

    def unavailable(answer: Answer) -> Answer:
        if not answer.path.endswith("with-timestamps"):
            return answer
        return replace(
            _rewritten(answer, {"detail": "Service Unavailable"}), status=503
        )

    with serving(tampering(unavailable)) as base_url:
        verdicts = _verdicts(base_url)

    for claim_id in sorted(refusing):
        assert verdicts[claim_id].word == "broken", (
            f"{claim_id} reported {verdicts[claim_id].word!r} against a "
            "deployment that refused the endpoint its own voice promised — a "
            "refusal read through spoke() alone becomes unasked, which is this "
            "epic's documented defect"
        )
        assert "contradicted its own declaration" in verdicts[claim_id].why


def test_a_start_time_that_goes_backwards_breaks_time_4() -> None:
    """`character_end_times_seconds` is left untouched, which is the whole point.

    A deployment that reverses two adjacent *start* times keeps its end times
    non-decreasing and keeps the span between its first start and its last end
    matching its audio exactly — so every other reading in `TIME-4` is satisfied,
    and a caller drawing each character when its start time arrives watches the
    clock run backwards. Reading `ends` alone reported this `held`.
    """

    def reversed_starts(answer: Answer) -> Answer:
        if not _plain_timestamped(answer):
            return answer
        objects = _timestamped(answer)
        for one in objects:
            times = one["alignment"]["character_start_times_seconds"]
            times[1], times[2] = times[2], times[1]
        return _relined(answer, objects)

    with serving(tampering(reversed_starts)) as base_url:
        verdict = _verdicts(base_url)["TIME-4"]

    assert verdict.word == "broken"
    assert "in `character_start_times_seconds` that go backwards" in verdict.why


def test_a_malformed_timing_array_breaks_cap_3() -> None:
    """The boundary `CAP-3` gained by reading through `_timed`, pinned deliberately.

    `characters` is left real and non-empty — the reading `CAP-3` used to make —
    and only the times beside it are spoiled. A body like this once held, because
    the old reader stopped at `characters`. It breaks now, and that is this
    claim's own argument carried the rest of the way: times that cannot be read
    are an alignment-shaped hole by exactly the reasoning that rejects an empty
    `characters` array, and `held` over them is a check that cannot fail.

    Pinned here rather than left implicit so that narrowing the shared parser
    back to its old tolerance fails a test, instead of quietly returning `CAP-3`
    to reporting `held` over garbage.
    """

    def untimed(answer: Answer) -> Answer:
        if not _plain_timestamped(answer):
            return answer
        objects = _timestamped(answer)
        for one in objects:
            one["alignment"]["character_end_times_seconds"] = ["soon", "later"]
        return _relined(answer, objects)

    with serving(tampering(untimed)) as base_url:
        verdict = _verdicts(base_url)["CAP-3"]

    assert verdict.word == "broken"
    assert "`character_end_times_seconds` is not an array of numbers" in verdict.why


def test_an_audio_base64_that_is_not_base64_breaks_cap_3() -> None:
    """The other half of the widening `probe_cap_3`'s docstring claims is pinned.

    That docstring names two bodies it now refuses — "a malformed timing array or
    an `audio_base64` that is not base64" — and says the suite holds it to both.
    Only the timing array was covered, so half the sentence was a promise about
    tests that did not exist, and loosening the base64 check would have restored
    the `held`-over-garbage tolerance without turning anything red.

    `characters` and both timelines are left real and ordered; only the audio the
    alignment claims to describe is spoiled. `TIME-4` has to read that audio to
    weigh a timeline against it, so a body that cannot produce any is an
    alignment-shaped hole by the same argument that rejects an empty
    `characters`.
    """

    def unreadable_audio(answer: Answer) -> Answer:
        if not _plain_timestamped(answer):
            return answer
        objects = _timestamped(answer)
        for one in objects:
            one["audio_base64"] = "not base64 at all!!"
        return _relined(answer, objects)

    with serving(tampering(unreadable_audio)) as base_url:
        verdict = _verdicts(base_url)["CAP-3"]

    assert verdict.word == "broken"
    assert "`audio_base64` that is not base64" in verdict.why


def test_a_timeline_that_does_not_lay_against_its_text_breaks_cap_3() -> None:
    """The length branch, pinned — every element still a perfectly good number.

    This is the branch the type check cannot reach: dropping one end time leaves
    an array of real, ordered, finite floats that simply cannot be laid against
    the characters it is supposed to time, so the caller has a timeline for every
    character but one and no way to know which. `test_a_malformed_timing_array_
    breaks_cap_3` corrupts values and so only ever exercises `_number`.

    Conceded as unpinned in a previous round and pinned here rather than conceded
    again: the same reviewer quoting my own concession back is the correct
    outcome for a gap left open twice.
    """

    def short_a_time(answer: Answer) -> Answer:
        if not _plain_timestamped(answer):
            return answer
        objects = _timestamped(answer)
        for one in objects:
            one["alignment"]["character_end_times_seconds"].pop()
        return _relined(answer, objects)

    with serving(tampering(short_a_time)) as base_url:
        verdict = _verdicts(base_url)["CAP-3"]

    assert verdict.word == "broken"
    assert "cannot be laid against the text it is supposed to time" in verdict.why


def test_a_streamed_body_carrying_one_object_leaves_time_6_unasked() -> None:
    """`TIME-6`'s `Blocked` arm, which is the one arm a green run never shows.

    A single object is a deployment declining to split a two-sentence text, and
    nothing documented obliges it to — so there is no pair to lay end to end and
    no promise broken. `held` there would be the check that cannot fail this
    epic exists to delete, and `broken` would report a promise nobody made.

    Worth its own test because the failure mode is invisible: swap the `raise
    Blocked` for `return None` and every run stays green while `TIME-6` quietly
    reports `held` over a response it never compared.
    """

    def only_the_first(answer: Answer) -> Answer:
        if not _streamed(answer):
            return answer
        objects = _timestamped(answer)
        return _relined(answer, objects[:1])

    with serving(tampering(only_the_first)) as base_url:
        verdict = _verdicts(base_url)["TIME-6"]

    assert verdict.word == "unasked"
    assert "no consecutive pair to lay end to end" in verdict.blocker


def test_a_missing_audio_base64_breaks_cap_3() -> None:
    """The field's other failure: absent entirely rather than unreadable.

    `_one_timed` refuses a missing or non-string `audio_base64` on a different
    branch from the one that refuses unparseable base64, and `TIME-4` needs the
    bytes either way — a timeline weighed against audio that never arrived is
    weighed against nothing.
    """

    def no_audio(answer: Answer) -> Answer:
        if not _plain_timestamped(answer):
            return answer
        objects = _timestamped(answer)
        for one in objects:
            del one["audio_base64"]
        return _relined(answer, objects)

    with serving(tampering(no_audio)) as base_url:
        verdict = _verdicts(base_url)["CAP-3"]

    assert verdict.word == "broken"
    assert "no `audio_base64` string" in verdict.why


def test_an_empty_timestamp_body_never_reaches_the_timestamp_parser() -> None:
    """Review asked for a test of `_timed`'s empty-body branch. This is why not.

    A 200 carrying nothing is refused a layer earlier, by `Asked.spoke`'s own
    precondition, so the branch below it cannot be reached from any claim — the
    verdict is `unasked` with "answered 200 carrying no audio at all", never the
    parser's "carrying no alignment at all". A test asserting the parser's
    message here would fail, and one asserting `broken` would be asserting
    something this deployment cannot produce.

    Pinned in the shape it actually has, because the branch is not dead: it is
    what stops [`_timed`] ever handing back an empty tuple, and `probe_time_4`
    now reads `objects[0]` on the strength of that.
    """

    def nothing_at_all(answer: Answer) -> Answer:
        if not _plain_timestamped(answer):
            return answer
        return replace(answer, body=b"")

    with serving(tampering(nothing_at_all)) as base_url:
        verdict = _verdicts(base_url)["CAP-3"]

    assert verdict.word == "unasked"
    assert "answered 200 carrying no audio at all" in verdict.blocker


def test_two_objects_on_one_line_breaks_time_3() -> None:
    """`TIME-3`'s "one object per line" is enforced by the parser, and that shows.

    Round-3 review read `probe_time_3.fault`, found no `len(objects)` check, and
    concluded the promise was unfalsifiable. It is not: [`_timed`] splits the body
    on lines and parses each one, so two objects sharing a line leave trailing
    data that `json.loads` refuses, and the claim reports `broken`. The mechanism
    lives one call away from the fault that depends on it, which is exactly why it
    earns a test named for the property rather than a comment.

    Note what is *not* asserted here. Merging the two sentences into a single
    object is not this violation — one object on one line satisfies "per line"
    literally — and nothing documented obliges the endpoint to split sentences,
    which is why `TIME-6` reports that response `unasked` rather than `broken`.
    """

    def crammed(answer: Answer) -> Answer:
        if not _streamed(answer):
            return answer
        objects = _timestamped(answer)
        return replace(
            answer, body=b" ".join(json.dumps(one).encode() for one in objects)
        )

    with serving(tampering(crammed)) as base_url:
        verdict = _verdicts(base_url)["TIME-3"]

    assert verdict.word == "broken"
    assert "answered a body that is not JSON" in verdict.why


def test_a_timeline_shifted_off_its_own_audio_breaks_time_4() -> None:
    """Every other reading in `TIME-4` is a difference, so none of them sees this.

    A uniform offset added to every time in the response leaves each character's
    span exact, leaves [`Timed.span`] exact — it subtracts two shifted values —
    and leaves `TIME-6`'s gaps exact for the same reason. The caller is handed
    subtitles that are uniformly late against the very audio the times arrived
    with, and before this check every claim reported `held`.

    The shift is the shape of a real bug rather than an invented one: `api.py:979`
    carries an `elapsed` accumulator forward between the sentences of one
    response, and an engine that fails to reset it between top-level requests
    produces exactly this.
    """

    def shifted(answer: Answer) -> Answer:
        if not _plain_timestamped(answer):
            return answer
        objects = _timestamped(answer)
        for one in objects:
            for field in _TIME_ARRAYS:
                one["alignment"][field] = [
                    second + 37.0 for second in one["alignment"][field]
                ]
        return _relined(answer, objects)

    with serving(tampering(shifted)) as base_url:
        verdict = _verdicts(base_url)["TIME-4"]

    assert verdict.word == "broken"
    assert "begins at 37.000s rather than at the start of the audio" in verdict.why


@pytest.mark.parametrize(
    "unusable",
    [float("nan"), float("inf"), float("-inf"), 10**400],
    ids=["nan", "infinity", "-infinity", "wider-than-a-float"],
)
def test_a_time_no_comparison_can_answer_breaks_time_4(unusable: float) -> None:
    """Each of these is shaped like a time and answers no question a claim asks.

    `json` reads all four off the wire: `NaN`, `Infinity` and `-Infinity` are
    non-standard tokens `json.dumps` itself writes for those floats, and Python's
    `int` has no width at all.

    All four are refused the same way and in the same place — `_number`'s range
    comparison, inside `_one_timed`, before any claim does arithmetic — which is
    why one assertion covers the four and why the message names the array rather
    than the reading that would have tripped over the value.

    What differs is only what each would do *were `_number` to admit it*, and
    that is the whole argument for refusing them at the crossing rather than
    inland. Measured by dropping the range clause and rerunning this test:

    - `NaN` — verdict `held`. It answers `False` to `<` and to `>` alike, so the
      reversal check and the length check both go quiet over a scrambled
      timeline. The silent one, and the reason this belongs in the parser.
    - `Infinity` and `-Infinity` — verdict `broken`, but from the reversal check
      ("the first infs followed by 0.013605s"), which reports a timeline that
      reverses rather than a time that cannot be read. Right word, wrong account.
    - an integer wider than a `float` — no verdict at all: `OverflowError` out of
      `float(second)` in `_one_timed`, which ends the run instead of judging the
      deployment.

    Rejecting all four at `_number` is what lets [`Timed`]'s stamp mean what its
    docstring says, and lets the claims downstream go on comparing without ever
    asking.
    """

    def unusable_time(answer: Answer) -> Answer:
        if not _plain_timestamped(answer):
            return answer
        objects = _timestamped(answer)
        for one in objects:
            one["alignment"]["character_end_times_seconds"][1] = unusable
        return _relined(answer, objects)

    with serving(tampering(unusable_time)) as base_url:
        verdict = _verdicts(base_url)["TIME-4"]

    assert verdict.word == "broken"
    assert "`character_end_times_seconds` is not an array of numbers" in verdict.why


# --------------------------------------------- the routes this service refuses


def bare(_method: str, _path: str, status: int) -> Response:
    """The two fall-throughs as they stood before `piper-conformance-e16.6`.

    Verbatim, because this is the deployment `ROUTE-1` and `ROUTE-2` exist to
    catch, and it is the one every build of this service was until that ticket
    landed.
    """
    detail = "Not Found" if status == 404 else "Method Not Allowed"
    return Response(
        content=json.dumps({"detail": detail}).encode(),
        status_code=status,
        media_type="application/json",
    )


def test_a_deployment_that_falls_through_to_starlette_breaks_both_claims() -> None:
    """The old server, probed by the claims written against it."""
    with serving(lying_deployment([WELL_FORMED], refuses=bare)) as base_url:
        verdicts = _verdicts(base_url)

    for claim in ("ROUTE-1", "ROUTE-2"):
        assert verdicts[claim].word == "broken", claim
        assert "does not name the method" in verdicts[claim].why


def test_a_refusal_naming_nothing_it_serves_is_broken() -> None:
    """Quoting the request back is half the promise, and the cheaper half.

    A body echoing the method and path tells a caller what it already sent. The
    list is the part that tells them where to go instead, so a refusal without
    one leaves them exactly where the bare 404 did.
    """

    def wordier(method: str, path: str, status: int) -> Response:
        return Response(
            content=json.dumps({"detail": f"{method} {path} is not served"}).encode(),
            status_code=status,
            media_type="application/json",
        )

    with serving(lying_deployment([WELL_FORMED], refuses=wordier)) as base_url:
        verdicts = _verdicts(base_url)

    for claim in ("ROUTE-1", "ROUTE-2"):
        assert verdicts[claim].word == "broken", claim
        assert "naming none of" in verdicts[claim].why


def test_a_deployment_answering_an_unrouted_path_at_all_is_broken() -> None:
    """A 200 where a refusal was documented, which is the answer-shaped void.

    A deployment answering every path is not a lenient one: a client that
    mistyped a URL gets a success back and never learns the endpoint it meant
    was never reached.
    """

    def cheerful(_method: str, _path: str, _status: int) -> Response:
        return Response(content=b"{}", status_code=200, media_type="application/json")

    with serving(lying_deployment([WELL_FORMED], refuses=cheerful)) as base_url:
        verdicts = _verdicts(base_url)

    for claim in ("ROUTE-1", "ROUTE-2"):
        assert verdicts[claim].word == "broken", claim
        assert "answered 200 rather than the documented" in verdicts[claim].why


def test_both_route_claims_are_asked_of_a_deployment_that_answers_nothing_else() -> None:
    """The two claims with no precondition, shown to have none.

    Every other claim here rests on a listing, a voice or a key, so a deployment
    that lost its catalogue leaves them `unasked`. A refusal needs none of that,
    and a prober that quietly let these two inherit a precondition would report a
    silent deployment as conforming on the one question it could still answer.
    """
    catalogueless = lying_deployment([], health_body={"voices": []}, health_status=503)
    with serving(catalogueless) as base_url:
        verdicts = _verdicts(base_url)

    for claim in ("ROUTE-1", "ROUTE-2"):
        assert verdicts[claim].word == "held", (claim, verdicts[claim].why)


@pytest.mark.parametrize("refusal", [401, 403])
def test_a_guard_in_front_of_the_router_blocks_both_claims_rather_than_breaking_them(
    refusal: int,
) -> None:
    """A refused key hides the router's verdict; it is not a verdict.

    Both statuses written out rather than read from [`prober.AUTH_REFUSALS`],
    for the reason [`WELL_FORMED`] is: a test drawing its cases from the constant
    under test follows it wherever it goes, so narrowing the prober's reading of
    a guard to one status would leave this passing on the half that still works.

    Needing no catalogue is not the same as reaching the router. A guard
    answering before any route is consulted — this fixture's middleware, or the
    authenticating proxy a deployment sits behind in the wild — refuses the
    prober's key first, and the 404 these claims came to read never happens.
    Reading that refusal as the router's own would report a sound deployment
    broken for a key the operator mistyped, which is the one direction a
    conformance prober must never fail in.
    """
    guarded = guarded_deployment(
        lambda path, sent: path != "/health" and sent != "probe-key", status=refusal
    )
    with serving(guarded) as base_url:
        verdicts = _verdicts(base_url, key="not-the-key")

    for claim in ("ROUTE-1", "ROUTE-2"):
        assert verdicts[claim].word == "unasked", (claim, verdicts[claim].why)
        assert "--key" in verdicts[claim].blocker, claim
        assert str(refusal) in verdicts[claim].blocker, claim
