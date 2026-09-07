"""The smoke script's parsers, held against output real runtimes actually printed.

`smoke.py` proves an image serves by running it, so almost all of it is only
provable with a container runtime — which a CI runner executing this suite does
not have, and which no unit test should require. What *is* provable here is the
part where the mistakes live: reading an image's declared port and healthcheck
out of two runtimes that spell an image config differently.

Every fixture string below was captured from a real `image inspect`, not
composed to match the parser. That distinction is the whole value of the file:
a fixture invented alongside the regex it feeds can only confirm that the regex
matches itself, which is the failure `tests/test_dockerfile.py` was written
about — a check that shares an assumption with the thing it checks.

The two runtimes are two maps of one territory, so the sharpest test here is not
either parser alone but that they agree: `test_both_runtimes_read_one_image_the
_same_way` describes a single image in both spellings and requires one answer.
"""

from __future__ import annotations

import json

import pytest

from smoke import (
    DOCKER,
    PODMAN,
    ImageConfig,
    SmokeFailure,
    _apple_image_config,
    _declared_port,
    _docker_image_config,
    select_runtime,
)

#: Verbatim from `container image inspect registry.sanctuary.gdn/elvenspeak-piper:2026.09.07.1`
#: on 2026-09-07 — Apple's runtime renders the healthcheck as Go's `%+v` of the
#: struct, and this is the only place it exposes it at all. Kept exactly as
#: printed because its two hard properties are both accidental-looking: the
#: command contains `]` (from `os.environ['PORT']`) and it contains `%s`.
PIPER_HEALTHCHECK_HISTORY = (
    "HEALTHCHECK {Test:[CMD-SHELL python -c \"import urllib.request,os; "
    "urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ['PORT'])\"] "
    "Interval:30s Timeout:5s StartPeriod:2m0s StartInterval:0s Retries:0}"
)

#: The command inside it — what both parsers must hand back, byte for byte.
PIPER_HEALTHCHECK = (
    "python -c \"import urllib.request,os; "
    "urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ['PORT'])\""
)


def apple_inspect(env: list[str], history: list[str]) -> str:
    """An Apple `image inspect` payload, shaped as the real one is."""
    return json.dumps(
        [
            {
                "id": "33284f32e77d",
                "variants": [
                    {
                        "config": {
                            "architecture": "amd64",
                            "config": {"Env": env, "Cmd": ["uv", "run", "--no-dev", "main.py"]},
                            "history": [{"created_by": line} for line in history],
                        }
                    }
                ],
            }
        ]
    )


def docker_inspect(env: list[str], test: list[str] | None) -> str:
    """A docker/podman payload in the shape `Runtime.inspect_image`'s template emits."""
    healthcheck = None if test is None else {"Test": test, "Interval": 30000000000}
    return json.dumps({"Env": env, "Healthcheck": healthcheck})


def test_the_apple_runtime_healthcheck_survives_the_bracket_inside_it():
    """The command ends `os.environ['PORT'])"`, so it closes a bracket of its own.

    A non-greedy match to the first `]` truncates it mid-expression and yields a
    command that is still a plausible-looking string — it would be `exec`d, fail
    for a reason having nothing to do with the image, and report the image
    unhealthy. Anchoring on the `Interval:` that follows the real closing bracket
    is what makes that unavailable, and this is the case that proves it.
    """
    config = _apple_image_config(apple_inspect(["PORT=5001"], [PIPER_HEALTHCHECK_HISTORY]))
    assert config.healthcheck == PIPER_HEALTHCHECK


def test_both_runtimes_read_one_image_the_same_way():
    """[LAW:one-source-of-truth] Two spellings of one image config, one answer.

    The parsers exist only because `docker` and Apple's `container` disagree
    about how to print an image; nothing about the *image* differs. If they ever
    return different answers for the same artifact, then which runtime ran the
    smoke test decides what was proved, and the check stops meaning one thing.
    """
    env = ["PATH=/usr/local/bin", "ELVENSPEAK_ENGINE=piper", "PORT=5001"]
    assert _apple_image_config(apple_inspect(env, [PIPER_HEALTHCHECK_HISTORY])) == (
        _docker_image_config(docker_inspect(env, ["CMD-SHELL", PIPER_HEALTHCHECK]))
    )


def test_the_port_comes_from_the_image_rather_than_from_this_repository():
    """An image built with a different PORT is polled on that port, not on 5001."""
    env = ["PORT=8080"]
    assert _apple_image_config(apple_inspect(env, [PIPER_HEALTHCHECK_HISTORY])).port == 8080
    assert _docker_image_config(docker_inspect(env, ["CMD-SHELL", "true"])).port == 8080


@pytest.mark.parametrize(
    "env",
    [[], ["PATH=/usr/local/bin"], ["PORTAL=5001"]],
    ids=["no environment", "an environment without PORT", "a name PORT is a prefix of"],
)
def test_an_image_that_names_no_port_is_refused(env):
    """[LAW:no-silent-failure] No default: a guessed port polls the wrong endpoint.

    Defaulting to 5001 here would turn "this image does not say where it serves"
    into "this image is unhealthy", which is a different sentence about a
    different problem.
    """
    with pytest.raises(SmokeFailure, match="declares no PORT"):
        _declared_port(env)


def test_a_port_that_is_not_a_number_is_refused():
    with pytest.raises(SmokeFailure, match="not a port number"):
        _declared_port(["PORT=highest"])


def test_an_environment_value_containing_an_equals_sign_survives():
    """Only the first separator splits, so a value with `=` in it is not truncated."""
    assert _declared_port(["PIPER_VOICES=a=b", "PORT=5001"]) == 5001


@pytest.mark.parametrize(
    ("apple_history", "docker_test"),
    [([], None), (["RUN echo hello"], None)],
    ids=["no history at all", "history naming no healthcheck"],
)
def test_an_image_declaring_no_healthcheck_is_refused_by_both(apple_history, docker_test):
    """Half this script's question is the image's own healthcheck; without one there
    is no question to ask, and reporting such an image "proven" would be a lie."""
    with pytest.raises(SmokeFailure, match="declares no HEALTHCHECK"):
        _apple_image_config(apple_inspect(["PORT=5001"], apple_history))
    with pytest.raises(SmokeFailure, match="declares no HEALTHCHECK"):
        _docker_image_config(docker_inspect(["PORT=5001"], docker_test))


def test_an_exec_form_healthcheck_is_refused_rather_than_guessed_at():
    """`["CMD", "prog", "arg"]` is space-joined beyond recovery in Apple's dump.

    Refusing it in both parsers keeps the two runtimes answering alike. Accepting
    it under `docker` only would mean an image that smokes clean in CI and cannot
    be smoked at all on the machine of whoever is asked to reproduce the failure.
    """
    with pytest.raises(SmokeFailure, match="only the shell form"):
        _docker_image_config(docker_inspect(["PORT=5001"], ["CMD", "wget", "-q", "http://x/y"]))
    with pytest.raises(SmokeFailure, match="only the shell form"):
        _apple_image_config(
            apple_inspect(
                ["PORT=5001"],
                ["HEALTHCHECK {Test:[CMD wget -q http://x/y] Interval:30s}"],
            )
        )


def test_the_last_healthcheck_declared_is_the_one_that_binds():
    """A later HEALTHCHECK replaces an earlier one, which is what the runtime does."""
    config = _apple_image_config(
        apple_inspect(
            ["PORT=5001"],
            [
                "HEALTHCHECK {Test:[CMD-SHELL /bin/false] Interval:30s}",
                PIPER_HEALTHCHECK_HISTORY,
            ],
        )
    )
    assert config.healthcheck == PIPER_HEALTHCHECK


def test_a_parsed_config_is_the_whole_answer():
    """[LAW:parse-dont-validate] Nothing downstream re-reads the inspect output."""
    assert _docker_image_config(
        docker_inspect(["PORT=5001"], ["CMD-SHELL", PIPER_HEALTHCHECK])
    ) == ImageConfig(port=5001, healthcheck=PIPER_HEALTHCHECK)


def test_podman_is_docker_with_a_different_binary_name():
    """The one thing that may differ between them, and the ones that may not.

    They are one instance apart on purpose: `podman` accepts docker's argv for
    every call this script makes, verified against both. A future edit that gave
    podman its own `running_names` or `read_config` would be re-describing a
    runtime that has not diverged.
    """
    assert PODMAN.binary == "podman" and DOCKER.binary == "docker"
    assert PODMAN.running_names == DOCKER.running_names
    assert PODMAN.inspect_image == DOCKER.inspect_image
    assert PODMAN.read_config is DOCKER.read_config


def test_asking_for_a_runtime_that_is_not_installed_names_what_is(monkeypatch):
    """[LAW:no-silent-failure] No falling through to whatever else is lying around.

    Silently running podman when docker was asked for would smoke a different
    artifact path than the one the caller meant to test, and say nothing.
    """
    monkeypatch.setattr("smoke.shutil.which", lambda binary: binary == "podman")
    with pytest.raises(SmokeFailure, match="wanted docker, found podman"):
        select_runtime("docker")
    assert select_runtime(None) is PODMAN


def test_no_runtime_at_all_is_refused_naming_every_candidate(monkeypatch):
    monkeypatch.setattr("smoke.shutil.which", lambda binary: None)
    with pytest.raises(SmokeFailure, match="none installed"):
        select_runtime(None)

