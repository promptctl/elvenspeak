"""What this surface answers a request it does not route.

Two fall-throughs, not one, and a fix for either alone leaves the other bare: a
catch-all matches unknown *paths*, and a known path asked under an unmatched
*method* never reaches one. Both are probed here, against the same house shape
every other refusal in this service uses — a message quoting what was sent, and
a list of the legal values beside it.

The negative half matters as much: the 422s, 501s and deliberate 404s this
service writes for itself must come back unchanged. They and the router's
verdicts travel as the same exception type down to a subclass, so a handler
claiming one class too many would rewrite every refusal in
`elvenspeak/api.py` as a fall-through and no other test here would notice.
"""

from __future__ import annotations

import pytest
from conftest import README, published_endpoints
from fastapi.testclient import TestClient
from test_api import VOICE, served, settings_for

from elvenspeak import create_app
from elvenspeak.api import NON_GOALS
from elvenspeak.engine import Capability

#: Starlette's own two answers, verbatim on the wire. The ticket's whole subject
#: is that neither of these names the route, the position this project has taken
#: on it, or anything this deployment does serve.
BARE = ('{"detail":"Not Found"}', '{"detail":"Method Not Allowed"}')

#: Methods to throw at every path, including one no route anywhere declares.
#: `FOO` is not filler: a catch-all route registered against a *list* of methods
#: leaves exactly this hole, because an undeclared method makes that route a
#: partial match like any other and the bare 405 wins.
METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "FOO")


@pytest.fixture(scope="module")
def client():
    with served(settings_for()) as started:
        yield started


def refusal(response) -> dict:
    """The refusal body, asserted to be the house shape on the way out."""
    detail = response.json()["detail"]
    assert set(detail) == {"message", "served"}, detail
    return detail


def test_an_unknown_path_is_refused_naming_what_was_asked(client) -> None:
    """404 is still the status; the body is what changes.

    The path is quoted back because a client that reached a wrong URL usually
    built it from a base URL and a suffix, and the assembled string is the one
    thing its author never sees.
    """
    detail = refusal(client.get("/v1/nope"))

    assert "GET /v1/nope" in detail["message"]
    assert client.get("/v1/nope").status_code == 404


def test_a_wrong_method_keeps_its_405_and_its_allow_header(client) -> None:
    """The router's verdict, reshaped rather than recomputed.

    405 and `Allow` are how a caller learns the path exists and what it takes,
    which a 404 would destroy — so the handler passes both through rather than
    answering every fall-through the same way. `POST /v1/voices` is the shape a
    client attempting to create a voice arrives in.
    """
    response = client.post("/v1/voices")

    assert response.status_code == 405
    assert response.headers["allow"] == "GET"
    assert "POST /v1/voices" in refusal(response)["message"]


def test_a_method_no_route_declares_at_all_is_refused_too(client) -> None:
    """The half-closed fall-through, probed directly.

    Nothing in this app declares `FOO`, so no route can match it fully and the
    request arrives by the partial-match arm — the same arm `POST /v1/voices`
    takes, which is why closing one closes the other.
    """
    assert "FOO /v1/voices" in refusal(client.request("FOO", "/v1/voices"))["message"]


def test_deleting_a_voice_is_answered_as_the_non_goal_it_is(client) -> None:
    """The route README has already taken a position on, answering with it.

    "Method Not Allowed" reads as an oversight a later release might fix. This
    one says the opposite, which is the only thing a caller can act on.
    """
    message = refusal(client.delete(f"/v1/voices/{VOICE}"))["message"]

    assert "non-goal" in message
    assert "no account" in message


def test_a_refusal_names_every_endpoint_the_readme_publishes(client) -> None:
    """The advertised surface, held against the documented one.

    Read off the router, so this is the first thing in the suite holding
    `elvenspeak.api`'s routes against README's table at all — `tests/test_prober.py`
    deliberately holds only the prober's tuples against it, because the prober
    judges builds other than this one ([LAW:one-source-of-truth]).

    A subset rather than an equality: the list also carries `/docs` and the
    schema, which README's table does not publish and a caller who guessed a
    path wrong is the likeliest person to want.
    """
    published = published_endpoints()
    served_here = set(refusal(client.get("/v1/nope"))["served"])

    assert {
        f"{method} {path}"
        for method, paths in published.items()
        for path in paths
    } <= served_here


def test_the_advertised_list_is_this_deployments_own_routes() -> None:
    """Read off the router, which a list kept beside it could not be.

    A hand-kept table — or a second copy of README's — would answer identically
    for the deployment it was written against, and would go on answering that
    after a route was added or taken away. Serving one path the default build
    does not, and finding it advertised, is what tells the two apart
    ([LAW:one-source-of-truth]).
    """
    settings = settings_for()
    app = create_app(settings, settings.engine.open())

    @app.get("/v1/only-here")
    def _extra() -> dict:
        return {}

    with TestClient(app) as client:
        assert "GET /v1/only-here" in refusal(client.get("/v1/nope"))["served"]


def test_a_refusal_never_advertises_the_request_it_just_refused(client) -> None:
    """The one entry the list must not hold, whichever fall-through produced it.

    A list built from anything but this app's own routes — a hand-kept table, a
    copy of README — could advertise the endpoint whose refusal it is attached
    to, and the caller would read that as a server contradicting itself inside
    one response.
    """
    for method, path in (("GET", "/v1/nope"), ("POST", "/v1/voices")):
        detail = refusal(client.request(method, path))
        assert f"{method} {path}" not in detail["served"]


def test_no_answer_on_this_surface_is_starlettes_own(client) -> None:
    """The ticket's mechanical check, swept over the whole surface.

    Every path this app routes — with `{voice_id}` filled — plus a path it does
    not, under every method including one no route declares. Both of Starlette's
    bare bodies are matched verbatim, because it is the exact string a client
    receives that carries no route, no position and no alternative.
    """
    paths = [
        route.path.replace("{voice_id}", VOICE) for route in client.app.routes
    ] + ["/v1/nope", "/", "/v1/voices/nosuch/settings"]

    for path in paths:
        for method in METHODS:
            body = client.request(method, path, follow_redirects=False).text
            assert body not in BARE, f"{method} {path}"


def test_the_refusals_this_service_writes_are_not_rewritten_as_fall_throughs() -> None:
    """The negative half, and the reason the handler names two exception classes.

    Every one of these travels as a subclass of what the router raises, so a
    handler registered one class too wide would answer all four with "this
    deployment does not serve …" — true of none of them, and the endpoint each
    one is about would vanish from the answer.
    """
    guarded = served(settings_for(api_key="secret"))
    keyless = guarded.get("/v1/voices")
    assert keyless.status_code == 401
    assert keyless.json()["detail"] == "invalid xi-api-key"

    with served(settings_for()) as client:
        unknown_voice = client.get("/v1/voices/nosuch")
        assert unknown_voice.status_code == 404
        assert unknown_voice.json()["detail"] == "unknown voice 'nosuch'"

        bad_format = client.post(
            f"/v1/text-to-speech/{VOICE}",
            json={"text": "Compatibility is measurable."},
            params={"output_format": "mp3_9999_9"},
        )
        assert bad_format.status_code == 422
        assert "supported" in bad_format.json()["detail"]

    withheld = settings_for(withheld=frozenset({Capability.TIMESTAMPS}))
    with served(withheld) as client:
        refused = client.post(
            f"/v1/text-to-speech/{VOICE}/with-timestamps",
            json={"text": "Compatibility is measurable."},
        )
        assert refused.status_code == 501
        assert (
            refused.json()["detail"]
            == f"this service cannot {Capability.TIMESTAMPS.value}"
        )


def test_a_trailing_slash_still_redirects_to_the_path_it_means(client) -> None:
    """What a catch-all route would have cost, pinned so nobody pays it quietly.

    A route matching every path is a full match for `/v1/voices/` too, which
    leaves the router nothing to redirect: a client whose base URL ends in a
    slash would go from working to refused, and the refusal would look correct.
    Closing the fall-through at the exception instead keeps this.
    """
    redirected = client.get("/v1/voices/", follow_redirects=False)

    assert redirected.status_code == 307
    assert redirected.headers["location"].endswith("/v1/voices")


def test_every_non_goal_is_one_this_project_states_in_prose() -> None:
    """The table and README's non-goals section, held together.

    Two maps of one position — what this service will never serve — and prose is
    the one a machine cannot check, so the check runs from the side that can.
    Without it the server could refuse an endpoint as a documented non-goal that
    no document names ([LAW:one-source-of-truth]).
    """
    published = README.read_text(encoding="utf-8")

    for method, path, because in NON_GOALS:
        assert f"{method} {path}" in published, (method, path)
        assert because.strip(), (method, path)
