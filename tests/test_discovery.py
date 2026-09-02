"""What the router is allowed to conclude from a service catalog.

Driven against a real Consul-shaped server on loopback (`tests/fleet.py`), for
the same reason the router's own tests are: the module's whole job is reading
somebody else's HTTP, and a patched `urlopen` would prove only that the patch
matches the code that calls it.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fleet import serving

from elvenspeak import discovery
from elvenspeak.provisioning import ConfigError


def catalog_serving(catalog: dict, health: dict, unhealthy: dict | None = None) -> FastAPI:
    """A Consul answering exactly what it is told to, however wrong.

    `unhealthy` holds instances that exist but are failing their check. They are
    withheld when the lookup filters to passing and returned when it does not,
    which is what makes the filter observable — a stub that always returned
    everything would let `?passing=true` be dropped without a test noticing.
    """
    failing = unhealthy or {}
    app = FastAPI()

    @app.get("/v1/catalog/services")
    def services() -> dict:
        return catalog

    @app.get("/v1/health/service/{name}")
    def instances(name: str, passing: bool = False):
        listed = list(health.get(name, []))
        return listed if passing else listed + list(failing.get(name, []))

    return app


def instance(host: str, port: int, service_address: str = "") -> dict:
    return {
        "Node": {"Address": host},
        "Service": {"Address": service_address, "Port": port},
    }


def test_only_services_carrying_the_tag_are_engines():
    """[LAW:one-source-of-truth] The cluster states it; this does not guess it.

    A name pattern would make this module decide from the shape of a string which
    services are its business, and `elvenspeak-lookalike` is what that mistake
    looks like: named like ours, tagged as nothing, and not ours.
    """
    app = catalog_serving(
        catalog={
            "elvenspeak-piper": [discovery.ENGINE_TAG],
            "elvenspeak-lookalike": [],
            "gitea": ["vcs"],
        },
        health={"elvenspeak-piper": [instance("10.0.0.4", 29280)]},
    )
    with serving(app) as consul:
        found = discovery.engines(consul)

    assert [backend.service for backend in found] == ["elvenspeak-piper"]
    assert found[0].base_url == "http://10.0.0.4:29280"


def test_a_service_with_its_own_address_is_reached_at_that_address():
    """Consul leaves `Service.Address` empty unless the service overrode it.

    Both readings are the documented shape of one answer rather than a field that
    might be missing, so they are taken together in the one place that has both.
    """
    app = catalog_serving(
        catalog={"elvenspeak-piper": [discovery.ENGINE_TAG]},
        health={
            "elvenspeak-piper": [
                instance("10.0.0.4", 29280, service_address="192.168.7.218")
            ]
        },
    )
    with serving(app) as consul:
        (backend,) = discovery.engines(consul)

    assert backend.base_url == "http://192.168.7.218:29280"


def test_every_healthy_instance_of_one_service_is_a_backend():
    """A service scaled to two allocations is two places to send a request."""
    app = catalog_serving(
        catalog={"elvenspeak-piper": [discovery.ENGINE_TAG]},
        health={
            "elvenspeak-piper": [
                instance("10.0.0.5", 29281),
                instance("10.0.0.4", 29280),
            ]
        },
    )
    with serving(app) as consul:
        found = discovery.engines(consul)

    # Sorted, because the first voice a router offers becomes the fallback voice
    # of every deployment that named none, and a fleet that reordered between
    # boots would move it.
    assert [backend.base_url for backend in found] == [
        "http://10.0.0.4:29280",
        "http://10.0.0.5:29281",
    ]


def test_an_instance_that_is_not_passing_its_check_is_not_a_backend():
    """Healthy, not merely registered — and the filter is what says which.

    Each engine's own check passes only once its voices are open, so an instance
    that is still loading its models is registered and cannot speak. Routing to
    it would 503 every clause of a conversation.

    This fails if `?passing=true` is ever dropped from the lookup URL, which was
    previously unprotected: the stub returned every instance regardless, so the
    filter could have been deleted with the whole suite still green.
    """
    app = catalog_serving(
        catalog={"elvenspeak-piper": [discovery.ENGINE_TAG]},
        health={"elvenspeak-piper": [instance("10.0.0.4", 29280)]},
        unhealthy={"elvenspeak-piper": [instance("10.0.0.9", 29289)]},
    )
    with serving(app) as consul:
        found = discovery.engines(consul)

    assert [backend.base_url for backend in found] == ["http://10.0.0.4:29280"]


def test_a_cluster_running_no_engines_is_an_empty_answer_not_a_failure():
    """Nobody deployed one yet. That is a fact, and the router reports it by
    having no voices and failing its own healthcheck — see `test_router`."""
    with serving(catalog_serving(catalog={"gitea": ["vcs"]}, health={})) as consul:
        assert discovery.engines(consul) == ()


def test_a_consul_that_cannot_be_reached_raises():
    """[LAW:no-silent-failure] The answer-shaped void this module must not return.

    An empty fleet and an unreachable agent would arrive as the same value, and
    the router would boot into a healthy-looking service that can speak nothing.
    """
    with pytest.raises(ConfigError) as raised:
        discovery.engines("http://127.0.0.1:1")
    assert "consul service catalog" in str(raised.value)


@pytest.mark.parametrize(
    "health, expected",
    [
        ({"elvenspeak-piper": [{"Node": {"Address": "10.0.0.4"}}]}, "Service and Node"),
        (
            {"elvenspeak-piper": [{"Node": {}, "Service": {"Port": 29280}}]},
            "named no address and port",
        ),
        (
            {"elvenspeak-piper": [{"Node": {"Address": "10.0.0.4"}, "Service": {}}]},
            "named no address and port",
        ),
    ],
    ids=["no-service-object", "no-address-anywhere", "no-port"],
)
def test_an_answer_that_is_not_an_address_refuses_the_boot(health, expected):
    """A backend the router cannot reach is worth more as a refusal than a guess.

    [LAW:parse-dont-validate] `Backend` exists because a `base_url` that was
    assembled from whatever was in the response would push the failure to
    whichever request first used it.
    """
    app = catalog_serving(
        catalog={"elvenspeak-piper": [discovery.ENGINE_TAG]}, health=health
    )
    with serving(app) as consul:
        with pytest.raises(ConfigError) as raised:
            discovery.engines(consul)

    assert expected in str(raised.value)
