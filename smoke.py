#!/usr/bin/env python3
"""Run a built image and prove it serves, or say why it did not.

Nothing in this repository has ever run a container. The publish job builds the
image, pushes it, checks the registry holds the digest it pushed and that the
config blob carries the right labels — every one of those proves the bytes
arrived and the metadata is stamped, and not one of them starts the process.
The suite proves the *program*: it runs in-process on the runner against
`create_app`. The gap is between the program and the *artifact*, and until this
file there was no check in it at all — which is how chatterbox shipped in an
image that has never run anywhere and no step told anyone.

What publishes green today without this: a baked asset landing at a path the
engine does not read, an `open()` that raises on the image's own environment,
`uv run --no-dev main.py` failing on a dependency present at build and absent at
runtime, a `USER elvenspeak` that cannot read what root baked.

WHAT IT ASKS THE IMAGE. Three questions, and they are not the same question:

  1. Does the service answer `/health` with 200, from outside the container?
  2. Does the image's *own* `HEALTHCHECK` — the command an orchestrator runs —
     exit 0, executed inside the container?
  3. Does it speak? [`speaks`] asks every property the engine seam promises,
     over HTTP, of the running container.

The first two are read from the image rather than from this repository. `PORT` is read from
the image's environment and the healthcheck out of its config, so this file
holds no second copy of either ([LAW:one-source-of-truth]). Reading them from
the Dockerfile instead was the obvious shortcut and is the wrong one: it would
prove a property of the *source* while claiming to have proved the artifact,
which is the exact gap this file exists to close. A published image and a
working tree are routinely different commits — the machine this was written on
had `2026.09.07.1` cached against a tree two commits ahead.

WHY THE LOGS ARE ALWAYS PRINTED, not only on failure. A smoke test that fails
without saying why costs more than no smoke test, and "print them if it went
wrong" is a branch that can itself be wrong. Printing them on every path is one
operation with no condition on it ([LAW:dataflow-not-control-flow]), it is less
code than the guarded version, and on the passing path it is the only account
anyone has of what the artifact said at boot. The container's lifetime is owned
by a context manager, so "started but never removed" and "failed without logs"
are not states this file can reach ([LAW:no-ambient-temporal-coupling]).

RUNTIMES. `docker` is what CI has; Apple's `container` is what a macOS dev box
has, and it is the only runtime on which this file could be *run* while it was
being written — a version verified nowhere is fiction, so both are supported and
both were exercised. They agree on far more than they disagree: `run`, `logs`,
`exec` and `rm --force` are byte-identical across all three CLIs here, and
"which containers are running" differs only in argv (`ps --format '{{.Names}}'`
against `ls -q`), both emitting one name per line. So a runtime is one type with
three instances differing in *data* ([LAW:one-type-per-behavior]) — `docker` and
`podman` differ in nothing but the binary's name — and the single genuine
divergence, how an image's config is spelled, is a pure function per runtime.

THE ROUTER IS THE ONE IMAGE THIS CANNOT ASK ALONE. It synthesizes nothing and
discovers what it fronts, so started by itself it has no voices and answers
`/health` 503 for as long as anyone waits. `--fleet` runs [`fleetstub`] beside it
— in a container started from the image under test, so nothing is built and
nothing is pulled — and hands the image the address to discover it at. Every
other engine is given an empty list of companions and runs the identical
sequence below ([LAW:dataflow-not-control-flow]).

Standard library only, deliberately. This runs on a CI runner before anything is
installed and on a laptop with no virtualenv activated; `python3 smoke.py` has
to be the whole invocation, so nothing here may import from `elvenspeak` or from
any dependency of it. [`fleetstub`] is neither, and is held to the same rule for
its own reasons, so importing it costs that guarantee nothing. Importing nothing
buys nothing if the *syntax* outruns the `python3` already on the machine — that
dies at parse time, before any of the failure messages below can be reached — so
the floor is pinned and checked by `tests/test_smoke.py` rather than promised
here.
"""

from __future__ import annotations

import argparse
import http.client
import json
import pathlib
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace

import fleetstub
import speaks

#: How long a container gets to answer `/health` before this gives up. The
#: image's own HEALTHCHECK allows a 120s start period, and piper reached 200 in
#: about nine seconds on the machine this was measured on — but chatterbox loads
#: a much larger model, so the cap is the slow engine's number and not the fast
#: one's. It is a cap rather than a wait: a container that never becomes healthy
#: has to fail rather than hold the runner until the job's own timeout.
DEFAULT_TIMEOUT = 180.0

#: Seconds between probes. Short enough that a fast engine is not billed a full
#: interval of idling, long enough that a slow one is not probed thousands of
#: times.
PROBE_INTERVAL = 2.0

#: Ceiling on any single runtime CLI call. Distinct from DEFAULT_TIMEOUT, which
#: bounds the whole wait: this one bounds one `docker`/`container` invocation, so
#: a wedged daemon surfaces as a named failure instead of a hung job.
CLI_TIMEOUT = 120.0

#: How long the stub fleet gets to answer. Distinct from DEFAULT_TIMEOUT, which is
#: sized for a container loading a model: this one binds a socket and answers from
#: memory, so a wait longer than this is a fleet the runner cannot route to rather
#: than one still starting — and the image under test still has its own full
#: budget to spend after it.
FLEET_TIMEOUT = 30.0

#: How long one *listing* request may take of a server that has already answered
#: `/health` 200. Distinct from DEFAULT_TIMEOUT for the reason FLEET_TIMEOUT is:
#: that one is a budget for a container still loading a model, and by the time
#: conformance runs the loading is over and the questions are answered from a
#: catalogue held in memory. `speaks.budget` covers the slow half — a synthesis
#: on cpu — and is that file's to decide, because it is the file that knows which
#: requests those are and what text each one carries.
CONFORM_TIMEOUT = 30.0

#: The variable a router reads to learn where to discover the engines it fronts.
#: Held equal to `elvenspeak.router.CONSUL_URL` by `tests/test_smoke.py`, because
#: this file cannot import it: a stub filling a variable the image no longer reads
#: would present as the empty fleet `--fleet` exists to prevent, and present it as
#: a timeout rather than as a name that does not match.
CONSUL_URL_VAR = "ROUTER_CONSUL_URL"


class SmokeFailure(Exception):
    """The image did not prove it serves, and this says which question it failed.

    One type for every way this can end badly, because every one of them means
    the same thing to the caller — the artifact is not proven, do not publish it
    — and they differ only in the sentence a human reads. The alternative, a
    family of exception classes nobody catches separately, would be structure
    standing in for a string.
    """


@dataclass(frozen=True)
class ImageConfig:
    """What running an image requires, read from the image and nowhere else.

    Both fields are answers to questions this file must not guess at. `port` is
    the port the server inside the container binds — hardcoding 5001 here would
    be a second copy of a value the Dockerfile already owns, free to drift the
    day an image is built with a different one. `healthcheck` is the shell
    command the image declares, which is the thing an orchestrator will run and
    therefore the thing worth asserting.
    """

    port: int
    healthcheck: str


def _declared_port(env: Iterable[str]) -> int:
    """The PORT the image's own environment names.

    `Env` arrives as `NAME=value` strings in both runtimes' spelling of an image
    config. A value containing `=` is legal and survives, because only the first
    separator is split on.
    """
    declared = dict(entry.partition("=")[::2] for entry in env)
    port = declared.get("PORT")
    if port is None:
        raise SmokeFailure(
            "the image declares no PORT in its environment, so there is no "
            "endpoint to poll — this file will not guess one"
        )
    if not port.isdigit():
        raise SmokeFailure(f"the image declares PORT={port!r}, which is not a port number")
    return int(port)


def _shell_healthcheck(test: Sequence[str] | None) -> str:
    """The shell command out of an OCI healthcheck's `Test`, or a loud refusal.

    `HEALTHCHECK CMD <shell string>` — the form this project's Dockerfile uses,
    and the form `docker` records as `["CMD-SHELL", ...]` — is the only shape
    accepted. Exec form (`["CMD", "prog", "arg"]`) is refused rather than
    half-handled: Apple's runtime prints the healthcheck as a Go struct dump in
    which an exec-form argv is space-joined and no longer separable, so
    supporting it would mean parsing it correctly under one runtime and
    plausibly-wrongly under the other. A refusal that names what it found is
    worth more than a command reassembled with the argument boundaries guessed
    ([LAW:parse-dont-validate], [LAW:no-silent-failure]).
    """
    declared = list(test or ())
    if len(declared) == 2 and declared[0] == "CMD-SHELL":
        return declared[1]
    if declared in ([], ["NONE"]):
        # `HEALTHCHECK NONE` and no HEALTHCHECK at all are different things to
        # write and the same thing to be told: there is no declared health for
        # this image, so there is no second question to ask of it.
        raise SmokeFailure(
            "the image declares no HEALTHCHECK, so there is nothing to assert — "
            "an image whose health nobody defined cannot be proven healthy"
        )
    raise SmokeFailure(
        f"the image's HEALTHCHECK is {declared!r}, and only the shell form "
        "(CMD-SHELL) can be run faithfully here"
    )


def _docker_image_config(stdout: str) -> ImageConfig:
    """Parse the config of an image as `docker` and `podman` report it.

    The `--format` template this reads is written to emit a shape *this* file
    defines rather than either runtime's own: asked for `{{json .Config}}`,
    `podman` returns an object with no `Healthcheck` key at all while resolving
    `.Config.Healthcheck` perfectly well, and `docker` nests it where `podman`
    hoists it to the top level. Naming both fields explicitly in the template
    settles that at the argv, so this function sees one shape and needs to know
    nothing about which CLI produced it.
    """
    raw = json.loads(stdout)
    return ImageConfig(
        port=_declared_port(raw["Env"] or ()),
        healthcheck=_shell_healthcheck((raw["Healthcheck"] or {}).get("Test")),
    )


#: Apple's runtime exposes an image's healthcheck only inside `history[].created_by`,
#: as Go's `%+v` rendering of the struct: `HEALTHCHECK {Test:[CMD-SHELL <cmd>]
#: Interval:30s ...}`. The slice is matched greedily and anchored on the
#: `Interval:` that follows the closing bracket, because the command itself
#: contains `]` — this project's healthcheck ends `os.environ['PORT'])"`, so a
#: non-greedy match to the first bracket truncates it mid-expression. `Interval`
#: is the next field of a fixed Go struct, which is what makes it a safe anchor.
_APPLE_HEALTHCHECK = re.compile(r"Test:\[(?P<test>.*)\] Interval:")


def _apple_healthcheck_test(history: Iterable[str]) -> list[str]:
    """The last declared healthcheck out of an Apple history dump, spelled as OCI does.

    This reads the shape and decides nothing. Whether a given shape can actually
    be run is `_shell_healthcheck`'s single call to make ([LAW:single-enforcer]) —
    it was briefly this function's too, and the cost was two copies of the
    sentence an image with no healthcheck gets told, free to drift into two
    different explanations of one situation.

    A later `HEALTHCHECK` supersedes an earlier one, so the last is the one that
    binds. A line this cannot read is handed on whole rather than discarded: it
    will be refused, and the refusal quotes what was actually there.
    """
    declared = [line for line in history if line.startswith("HEALTHCHECK")]
    if not declared:
        return []
    found = _APPLE_HEALTHCHECK.search(declared[-1])
    if not found:
        return [declared[-1]]
    kind, _, command = found.group("test").partition(" ")
    return [kind, command] if command else [kind]


def _apple_image_config(stdout: str) -> ImageConfig:
    """Parse the config of an image as Apple's `container` reports it.

    One variant is read, not all of them: both fields come from `ENV` and
    `HEALTHCHECK` instructions, which are architecture-independent by
    construction, so a multi-architecture index cannot disagree with itself about
    them in an image built from one Dockerfile.
    """
    variant = json.loads(stdout)[0]["variants"][0]["config"]
    history = [entry.get("created_by", "") for entry in variant["history"]]
    return ImageConfig(
        port=_declared_port(variant["config"]["Env"] or ()),
        healthcheck=_shell_healthcheck(_apple_healthcheck_test(history)),
    )


@dataclass(frozen=True)
class Runtime:
    """One container CLI, described by the two ways they differ and nothing else.

    `run`, `logs`, `exec` and `rm --force` are spelled identically by every CLI
    here and so appear nowhere in this type — a field per verb would be three
    copies of one string. What is left is the pair that genuinely diverges: how
    to list running containers, and how to read an image's config.
    """

    name: str
    binary: str
    #: Argv after the binary that prints the name of every *running* container,
    #: one per line. `docker ps` needs a format to stop printing a table; Apple's
    #: `ls -q` prints names already, because there a container's name is its id.
    running_names: tuple[str, ...]
    #: Argv after the binary, before the image reference, that prints the image's
    #: configuration in whatever shape `read_config` expects.
    inspect_image: tuple[str, ...]
    read_config: Callable[[str], ImageConfig]


DOCKER = Runtime(
    name="docker",
    binary="docker",
    running_names=("ps", "--format", "{{.Names}}"),
    inspect_image=(
        "image",
        "inspect",
        "--format",
        '{"Env":{{json .Config.Env}},"Healthcheck":{{json .Config.Healthcheck}}}',
    ),
    read_config=_docker_image_config,
)

#: Identical to docker in every respect but the name of the binary, which is the
#: whole reason `Runtime` is data rather than a class hierarchy: this instance
#: costs one line and cannot drift from the behaviour it shares.
PODMAN = replace(DOCKER, name="podman", binary="podman")

APPLE = Runtime(
    name="container",
    binary="container",
    running_names=("ls", "-q"),
    inspect_image=("image", "inspect"),
    read_config=_apple_image_config,
)

#: Ordered by preference, and the order is the point: CI has `docker`, so a
#: machine with several installed smokes the image the same way CI will.
RUNTIMES = (DOCKER, PODMAN, APPLE)


def select_runtime(requested: str | None) -> Runtime:
    """The runtime to use, or a refusal naming what was wanted and what is here.

    Asking for one that is not installed and finding none at all are the same
    failure — no usable runtime — so they share an arm and a message rather than
    branching into two that say the same thing.
    """
    available = [runtime for runtime in RUNTIMES if shutil.which(runtime.binary)]
    usable = [r for r in available if r.name == requested] if requested else available
    if not usable:
        wanted = requested or "any of " + ", ".join(r.name for r in RUNTIMES)
        raise SmokeFailure(
            f"no usable container runtime: wanted {wanted}, "
            f"found {', '.join(r.name for r in available) or 'none installed'}"
        )
    return usable[0]


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run a runtime command, converting a timeout into the one failure type.

    The base of three: the only place a runtime subprocess is spawned, so
    `TimeoutExpired` has exactly one place it could escape from and does not. What
    reaches a caller is a return code, which the two rungs above read differently
    because a non-zero exit does not mean the same thing to them.
    """
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=CLI_TIMEOUT)
    except subprocess.TimeoutExpired as expired:
        raise SmokeFailure(
            f"`{' '.join(argv)}` did not return within {CLI_TIMEOUT:.0f}s"
        ) from expired


def _capture(argv: Sequence[str]) -> str:
    """Run a runtime command and return its stdout, or fail loudly with its stderr.

    The only way past this function is exit 0, so no caller downstream inspects a
    return code ([LAW:parse-dont-validate]).
    """
    done = _run(argv)
    if done.returncode != 0:
        raise SmokeFailure(f"`{' '.join(argv)}` exited {done.returncode}\n{done.stderr.strip()}")
    return done.stdout


def _attempt(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run a cleanup command that must not raise, whatever happens to it.

    Cleanup runs while another failure is already on its way up, so a second one
    raised from here would replace the verdict the caller came for and skip
    whatever cleanup stood after it. The failure arm is therefore turned back into
    data: a timeout returns 124 — what `timeout(1)` reports — carrying its own
    sentence as stderr, which is the shape both callers already print. A wedged
    daemon is then reported through the same path as any other bad exit rather
    than through an arm written for it ([LAW:dataflow-not-control-flow]).
    """
    try:
        return _run(argv)
    except SmokeFailure as timed_out:
        return subprocess.CompletedProcess(argv, 124, stdout="", stderr=str(timed_out))


def _free_port() -> int:
    """A port on the loopback interface that nothing holds right now.

    Asking the kernel beats picking a constant: the publish job runs one leg per
    engine, and two legs that both hardcoded a port could not run on one runner.
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_config(runtime: Runtime, image: str) -> ImageConfig:
    """Ask a runtime to describe an image, and refuse an answer of the wrong shape.

    Both `read_config` implementations walk into the runtime's JSON by index and
    key, which is the right way to read a shape that is a contract and the wrong
    way to meet one that is not. This is the single place that crossing is made
    ([LAW:single-enforcer]), so a runtime that answers with an empty `variants`, a
    missing key or something that is not JSON at all reaches the caller as the one
    failure type quoting what it actually printed, rather than as a traceback past
    `main`'s handler. The four names are the ways walking into JSON of an
    unexpected shape goes wrong and not one name more: a bare `Exception` here
    would also catch a genuine bug in either parser and report it as the runtime's
    fault, which is a wrong diagnosis printed confidently.

    `SmokeFailure` passes through because it descends from `Exception` directly and
    is none of the four — load-bearing and invisible: rebased on `ValueError` it
    would be swallowed here, and the refusals naming the exact healthcheck or port
    they found would come back out wearing this generic sentence instead.
    """
    stdout = _capture([runtime.binary, *runtime.inspect_image, image])
    try:
        return runtime.read_config(stdout)
    except (LookupError, ValueError, TypeError, AttributeError) as malformed:
        raise SmokeFailure(
            f"`{runtime.binary} {' '.join(runtime.inspect_image)}` described {image} in a "
            f"shape this cannot read ({malformed!r}), so neither its port nor its "
            f"healthcheck could be established:\n{stdout.strip()}"
        ) from malformed


@dataclass(frozen=True)
class Probe:
    """One observation of the health endpoint: whether it served, and what it said.

    `detail` is filled on both arms and always means the same thing — what this
    probe saw — so a timeout can report the last real observation instead of
    "no answer", which is the difference between a diagnosable failure and a
    shrug.
    """

    served: bool
    detail: str


def _probe(url: str) -> Probe:
    """Ask `/health` once, turning every outcome into an observation.

    Errors are caught here and nowhere else, and none of them is suppressed —
    each becomes the text of an observation that a failure message will print
    ([LAW:no-silent-failure]). A non-200 is not an immediate failure because it
    can be a server still coming up; it becomes one when the deadline passes,
    and by then it is quoted.
    """
    try:
        with urllib.request.urlopen(url, timeout=PROBE_INTERVAL) as answer:
            body = answer.read().decode("utf-8", "replace").strip()
            return Probe(served=answer.status == 200, detail=f"HTTP {answer.status} {body}")
    except urllib.error.HTTPError as status:
        return Probe(served=False, detail=f"HTTP {status.code} {status.reason}")
    # `URLError` and `TimeoutError` are both `OSError`, so the one name covers a
    # refused connection, a reset and a timeout alike. `HTTPException` is not, and
    # is what a half-written response from a server still coming up arrives as —
    # left out, it escapes as a traceback that reads like a bug in this file
    # rather than the boot-time symptom it is.
    except (OSError, http.client.HTTPException) as unreachable:
        return Probe(served=False, detail=f"unreachable: {unreachable!r}")


def _await_serving(runtime: Runtime, name: str, url: str, timeout: float) -> str:
    """Poll until the container serves, dies, or runs out of time.

    Liveness is checked before the endpoint on every pass, so a container that
    exits at boot fails in one interval with "exited before it served" rather
    than after the full timeout with "no answer" — the same red, a different
    afternoon.

    Reading "not running" as "it exited" rests on a contract rather than on
    timing: every runtime here returns from `run -d` only once the container has
    been started, so by the time this loop can ask, a name that is absent is a
    process that has already gone. Were that a race instead of a contract, the
    symptom would be this failure raised against an empty log block
    ([LAW:no-ambient-temporal-coupling]).
    """
    deadline = time.monotonic() + timeout
    seen = "no probe completed"
    while True:
        if name not in _capture([runtime.binary, *runtime.running_names]).split():
            raise SmokeFailure(f"{name} exited before it served {url} — last probe: {seen}")
        probe = _probe(url)
        seen = probe.detail
        if probe.served:
            return probe.detail
        if time.monotonic() >= deadline:
            raise SmokeFailure(
                f"{url} did not answer 200 within {timeout:.0f}s — last probe: {seen}"
            )
        time.sleep(PROBE_INTERVAL)


def _print_logs(runtime: Runtime, name: str) -> None:
    """Print everything the container said, on every path, success included.

    Deliberately not `_capture`: a failure to read logs must not replace the
    failure that made them worth reading. The exit code is printed instead, so a
    `logs` call that went wrong is visible rather than an empty block that reads
    like a quiet boot.
    """
    done = _attempt([runtime.binary, "logs", name])
    print(f"--- {name} logs (`{runtime.binary} logs` exited {done.returncode}) ---", flush=True)
    print(done.stdout, end="", flush=True)
    print(done.stderr, end="", flush=True)
    print(f"--- end {name} logs ---", flush=True)


@contextmanager
def _container(runtime: Runtime, name: str, argv: Sequence[str]) -> Iterator[None]:
    """Own one container's whole lifetime: start it, then always log it and remove it.

    Starting is inside the manager because `run -d` is create-then-start, so a
    start that fails after a successful create leaves a named container behind —
    with the `run` outside, that was the one container neither logged nor removed
    by the `finally` written to guarantee both.

    The two ways in are kept apart because they leave the host in different states
    and only one of them has anything to say. A `run` that failed may have created
    nothing at all, so its sweep is silent: the caller already has the runtime's
    own stderr in the failure going up, and warning that a container "was left
    behind" when none was ever created would send someone hunting one that does
    not exist. Past a `run` that succeeded there is certainly one, so its logs are
    the point and a removal that fails is worth saying out loud — what that leaves
    behind is a container holding a port on the runner.
    """
    sweep = [runtime.binary, "rm", "--force", name]
    try:
        _capture(argv)
    except SmokeFailure:
        _attempt(sweep)
        raise
    try:
        yield
    finally:
        _print_logs(runtime, name)
        removed = _attempt(sweep)
        if removed.returncode != 0:
            print(
                f"warning: {name} was left behind — `{runtime.binary} rm --force` "
                f"exited {removed.returncode}: {removed.stderr.strip()}",
                file=sys.stderr,
                flush=True,
            )


def _fleet_address(url: str) -> str:
    """The address the stub fleet says a sibling container reaches it at.

    The one place an answer from [`fleetstub`] is crossed into this file, so a
    stub that came up but replied with something else — an old copy of the file
    with no `/address`, a proxy in the way — reaches the caller as a named failure
    instead of a `KeyError` past `main`'s handler ([LAW:parse-dont-validate]).
    """
    try:
        with urllib.request.urlopen(url, timeout=PROBE_INTERVAL) as answer:
            found = json.load(answer)["base_url"]
            # Into the handler below, so a wrong answer and no answer fail alike.
            if not isinstance(found, str) or not found:
                raise TypeError(f"base_url was {found!r}")
            return found
    # The same four names `_read_config` crosses on, plus the two a socket fails
    # with. `TypeError` is the one that looks redundant and is not: an answer that
    # is a JSON list or string takes a string subscript and raises it, not the
    # `LookupError` a missing key would.
    except (
        OSError,
        http.client.HTTPException,
        ValueError,
        LookupError,
        TypeError,
        AttributeError,
    ) as unread:
        raise SmokeFailure(
            f"the stub fleet answered {url} with something that is not an address "
            f"({unread!r}), so the image has nowhere to be pointed"
        ) from unread


@contextmanager
def _serving_fleet(runtime: Runtime, image: str, platform: str | None) -> Iterator[str]:
    """Run [`fleetstub`] beside the image, yielding the env entry that finds it.

    STARTED FROM THE IMAGE UNDER TEST, with its entrypoint replaced by the
    `python3` its own base layer already carries, and this repository's file
    handed to it as source on the command line. Nothing is built, nothing is
    pulled and nothing is mounted: the image is already in the local store because
    the step above built it there, and a mount or a copy would assume the runner's
    filesystem is the daemon's, which is not a thing this file is entitled to
    assume. On a registry with no other elvenspeak image there is also nothing
    else to borrow.

    THE PORT IS PUBLISHED TO LOOPBACK FOR THIS PROCESS, not for the image under
    test — which reaches the stub over the container network at the address the
    stub reports. Two addresses for one server, because the two callers are on
    different sides of it. Publishing is what lets the wait below be the same wait
    every container here gets, rather than a sleep long enough to usually work
    ([LAW:no-ambient-temporal-coupling]).
    """
    host_port = _free_port()
    name = f"elvenspeak-fleet-{uuid.uuid4().hex[:8]}"
    probe = f"http://127.0.0.1:{host_port}{fleetstub.ADDRESS_PATH}"
    argv = [
        runtime.binary,
        "run",
        "-d",
        "--name",
        name,
        "-p",
        f"127.0.0.1:{host_port}:{fleetstub.DEFAULT_PORT}",
        *(("--platform", platform) if platform else ()),
        "--entrypoint",
        "python3",
        image,
        "-c",
        pathlib.Path(fleetstub.__file__).read_text(encoding="utf-8"),
        "--port",
        str(fleetstub.DEFAULT_PORT),
    ]

    print(f"smoke: {runtime.name} running the stub fleet as {name}", flush=True)
    with _container(runtime, name, argv):
        _await_serving(runtime, name, probe, FLEET_TIMEOUT)
        found = _fleet_address(probe)
        print(f"smoke: the stub fleet reports itself at {found}", flush=True)
        yield f"{CONSUL_URL_VAR}={found}"


def smoke(
    runtime: Runtime,
    image: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    memory: str | None = None,
    platform: str | None = None,
    env: Sequence[str] = (),
    fleet: bool = False,
) -> None:
    """Run `image` and return only if it served and its own healthcheck passed.

    Returns nothing: there is no verdict to inspect, because the failure arm is
    an exception and the success arm is having got here at all.

    `fleet` asks for a discoverable fleet to exist first, which only the router
    needs. It is resolved here into environment entries — nothing below this line
    knows a router exists, and the sequence an image is put through is the same
    one whether the list it produced was empty or not.
    """
    with ExitStack() as running:
        supplied = (
            [running.enter_context(_serving_fleet(runtime, image, platform))]
            if fleet
            else []
        )
        env = (*env, *supplied)
        config = _read_config(runtime, image)
        host_port = _free_port()
        name = f"elvenspeak-smoke-{uuid.uuid4().hex[:8]}"
        base = f"http://127.0.0.1:{host_port}"
        url = f"{base}/health"

        # `--rm` is not used and must not be added: with `-d` it removes the
        # container the instant it exits, taking the logs of exactly the boot
        # failure this exists to explain. Removal is the context manager's job,
        # after the logs have been read.
        argv = [
            runtime.binary,
            "run",
            "-d",
            "--name",
            name,
            "-p",
            f"127.0.0.1:{host_port}:{config.port}",
            *(("--platform", platform) if platform else ()),
            *(("-m", memory) if memory else ()),
            *[argument for value in env for argument in ("-e", value)],
            image,
        ]

        print(
            f"smoke: {runtime.name} running {image} as {name}, "
            f"container port {config.port} published on {host_port}",
            flush=True,
        )
        # Entered on the one stack rather than in a nested `with`, so that a fleet
        # started above is torn down after this container and not before it: the
        # image under test is logged and removed while the thing it discovered is
        # still answering, which is the order that keeps a shutdown from looking
        # like a lost backend.
        running.enter_context(_container(runtime, name, argv))
        served = _await_serving(runtime, name, url, timeout)
        print(f"smoke: {url} answered {served}", flush=True)

        # `_run`, not `_capture`: a non-zero exit here is the answer to the
        # question, phrased below, rather than a runtime call that went wrong.
        checked = _run([runtime.binary, "exec", name, "/bin/sh", "-c", config.healthcheck])
        if checked.returncode != 0:
            raise SmokeFailure(
                f"the image's own HEALTHCHECK exited {checked.returncode} against a "
                f"server that answered 200 — the check and the endpoint disagree\n"
                f"  command: {config.healthcheck}\n"
                f"  stderr: {checked.stderr.strip()}"
            )
        print(f"smoke: the image's own HEALTHCHECK exited 0 ({config.healthcheck})", flush=True)

        # Unconditional, and last. Every image gets the same questions asked of it
        # because `speaks.py` reads what a voice can do from that voice rather than
        # from a table of engines ([LAW:dataflow-not-control-flow]) — an engine
        # added tomorrow is conformant here without being named anywhere. Last
        # because it is the only step that costs inference: an image that does not
        # boot, or whose own healthcheck disagrees with its endpoint, says so in
        # seconds rather than after a minute of synthesis.
        speaks.conform(base, CONFORM_TIMEOUT)
        print(f"smoke: {image} answered every question `speaks.py` asks", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a container image and prove it serves.",
        epilog=(
            "Exits 0 only when /health answered 200, the image's own HEALTHCHECK "
            "exited 0 inside the running container, and every property speaks.py "
            "asks held. The container's logs are printed either way."
        ),
    )
    parser.add_argument("image", help="the image reference to run, as the runtime resolves it")
    parser.add_argument(
        "--runtime",
        choices=[runtime.name for runtime in RUNTIMES],
        help="which container CLI to use (default: the first of %s that is installed)"
        % ", ".join(runtime.name for runtime in RUNTIMES),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="seconds to wait for /health to answer 200 (default: %(default)s)",
    )
    parser.add_argument(
        "--memory",
        help="memory limit for the container, in the runtime's own spelling (e.g. 3072m)",
    )
    parser.add_argument(
        "--platform",
        help="platform to run, for an image whose architecture is not the host's "
        "(e.g. linux/amd64)",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="an environment variable for the container; repeatable",
    )
    parser.add_argument(
        "--fleet",
        action="store_true",
        help="run a discoverable stub fleet beside the image and set "
        f"{CONSUL_URL_VAR} to it — what a router needs before it has any voices",
    )
    args = parser.parse_args(argv)

    try:
        smoke(
            select_runtime(args.runtime),
            args.image,
            timeout=args.timeout,
            memory=args.memory,
            platform=args.platform,
            env=args.env,
            fleet=args.fleet,
        )
    # Both verdicts, printed one way: `speaks.py` refuses with its own type because
    # it cannot import this file's, and an image that serves but cannot speak is
    # the same red to whoever reads this line. [LAW:single-enforcer]
    except (SmokeFailure, speaks.ConformanceFailure) as failure:
        print(f"smoke: FAILED — {failure}", file=sys.stderr, flush=True)
        return 1
    print("smoke: ok", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
