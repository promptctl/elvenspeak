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
from pathlib import Path
from typing import Any

import pytest
from conftest import DECLARED_VOICES
from fastapi import FastAPI, Request, Response
from fleet import engine_app, serving

from elvenspeak import prober
from elvenspeak.api import MAX_TEXT_LENGTH
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
LANDED_ISSUES = ("e16.3", "e16.4")

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


def lying_deployment(
    voices: list[dict[str, Any]],
    health_body: Any = HEALTHY,
    health_status: int = 200,
    speaks: Speaks = conformant,
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
    """
    app = FastAPI()

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


#: A row of README's endpoint table: the method and the path, in backticks.
_ENDPOINT_ROW = re.compile(r"^\|\s*`(GET|POST) (/\S+)`\s*\|", re.M)


def published_endpoints() -> dict[str, set[str]]:
    """README's endpoint table, as `method -> paths`."""
    text = (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")
    endpoints: dict[str, set[str]] = {"GET": set(), "POST": set()}
    for row in _ENDPOINT_ROW.finditer(text):
        endpoints[row.group(1)].add(row.group(2))
    return endpoints


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
    ) -> Response | None:
        if isinstance(body.get("text"), str) and not body["text"].strip():
            return _refusal("request exceeds the maximum context window")
        return _refused(output_format, body)

    with serving(lying_deployment([WELL_FORMED], speaks=refusing_without_naming)) as url:
        verdicts = _verdicts(url)

    assert verdicts["REF-2"].word == "broken"
    assert "naming none of" in verdicts["REF-2"].why
    assert verdicts["REF-4"].word == "held", "the rows that do name `text` are untouched"


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
