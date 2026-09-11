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

import re
from pathlib import Path
from typing import Any

import pytest
from conftest import DECLARED_VOICES
from fastapi import FastAPI, Response
from fleet import engine_app, serving

from elvenspeak import prober
from elvenspeak.prober import Broken, Claim, Finding, Held, Target, Unasked

#: The document that owns the claim ids and their evidence classes. Read from
#: the repository rather than restated here: this file's whole job in the first
#: three tests is to be the thing that notices the two disagreeing, and a copy of
#: the answer could not ([LAW:one-source-of-truth]).
CLAIMS_DOC = Path(__file__).parent.parent / "docs" / "conformance-claims.md"

#: The issue whose claims `elvenspeak.prober` is expected to ask today. Named
#: because the document's own "Probed by" column is what decides the set, so the
#: day `piper-conformance-e16.4` lands its claims, this list grows by editing the
#: document and the prober — never this test.
THIS_ISSUE = "e16.3"

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


def test_every_claim_the_document_assigns_this_issue_is_probed() -> None:
    """The prober asks everything it was made responsible for, and says so."""
    owed = {
        claim_id
        for claim_id, (_, probed_by) in documented().items()
        if THIS_ISSUE in probed_by
    }
    asked = {claim.id for claim in prober.CLAIMS}

    assert owed - asked == set(), (
        f"{CLAIMS_DOC.name} says {sorted(owed - asked)} are probed by "
        f"{THIS_ISSUE} and nothing here asks them"
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


# ------------------------------------- the deployments that lie past the parse


def lying_deployment(
    voices: list[dict[str, Any]],
    health_voices: tuple[str, ...] = ("v1",),
    health_status: int = 200,
) -> FastAPI:
    """A server answering whatever it is told to, however untrue.

    Hand-built rather than `create_app` with a doctored engine, because what has
    to be expressible here is a response this service *cannot* produce. A stand-in
    that could only emit well-formed answers would make every test below pass by
    construction ([LAW:verifiable-goals] — the failure needs a shape it can take).
    """
    app = FastAPI()

    @app.get("/health")
    def health(response: Response) -> dict:
        response.status_code = health_status
        return {"voices": list(health_voices)}

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
    def speak(voice_id: str) -> Response:
        return Response(content=b"\0\0" * 512, media_type="audio/pcm")

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
    with serving(lying_deployment([WELL_FORMED], health_voices=("v1", "ghost"))) as url:
        verdicts = _verdicts(url)

    assert verdicts["HEALTH-2"].word == "broken"
    assert "ghost" in verdicts["HEALTH-2"].why


def test_health_answering_200_with_no_voices_is_broken() -> None:
    """The 2026-09-02 incident, asked as a claim.

    A router that discovered no engines registered in Consul as passing and
    served silence. The status line and the body have to agree, and this is the
    direction that puts a dead deployment into rotation.
    """
    with serving(lying_deployment([WELL_FORMED], health_voices=())) as base_url:
        verdicts = _verdicts(base_url)

    assert verdicts["HEALTH-1"].word == "broken"
    # And the claim that would have had to speak those voices says it could not,
    # rather than passing because there was nothing to check.
    assert verdicts["HEALTH-2"].word == "unasked"


def test_health_answering_503_while_offering_voices_is_broken() -> None:
    """The other direction: a server withdrawn from rotation that can still speak."""
    with serving(lying_deployment([WELL_FORMED], health_status=503)) as base_url:
        assert _words(base_url)["HEALTH-1"] == "broken"


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
