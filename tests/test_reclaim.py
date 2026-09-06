"""What the suite's own reclaim has to be true of, asserted without a model.

[LAW:verifiable-goals] `conftest.reclaim` is the whole of what stands between
this suite and the OOM killer on the build runner, and its correctness is
otherwise observable only on a 7.9 GB machine, four test files and forty minutes
into a run that then dies with no traceback. The property is cheap to state and
cheap to check, so it is checked here in milliseconds instead: an app nobody
holds any more must not still be holding an engine.

A stand-in engine proves it exactly as well as a 3412 MiB Chatterbox one, because
what pins the engine is the closure its handlers were built from, not its weight.
"""

from __future__ import annotations

import os
import subprocess
import sys
import weakref
from pathlib import Path

from conftest import (
    DECLARED_VOICES,
    DeclaredEngine,
    DeclaredPrepared,
    declaring,
    reclaim,
)

from elvenspeak import create_app
from elvenspeak.engine import Capability
from elvenspeak.engines import ENGINES
from elvenspeak.settings import Settings
from elvenspeak.voices import Substitution


def _served_then_dropped() -> weakref.ref:
    """Serves an engine through a real app, and returns only a weak handle to it.

    Everything strong is local to this frame, so returning is what drops both the
    app and the engine — no `del`, and nothing for a later reader to preserve by
    accident. The reference that outlives the call is deliberately one that
    cannot itself keep the engine alive.
    """
    engine = DeclaredEngine(declaring(frozenset(Capability), DECLARED_VOICES))
    create_app(
        Settings(
            engine=DeclaredPrepared(),
            engine_name="declared",
            known_engines=frozenset(ENGINES) | {"declared"},
            withheld=frozenset(),
            fallback=Substitution.FIRST_OFFERED,
            api_key=None,
            host="127.0.0.1",
            port=0,
        ),
        engine,
    )
    return weakref.ref(engine)


def test_an_app_the_suite_has_finished_with_stops_holding_its_engine():
    """[LAW:no-ambient-temporal-coupling] The reclaim owns this, or nothing does.

    `create_app` builds each handler as a closure over the engine it serves, and
    `fastapi.dependencies.models` memoizes what kind of callable each handler is
    against the handler itself, 4096 entries deep. So a FastAPI app outlives
    every reference the suite holds to it, and carries its engine along — which
    for `test_chatterbox.py` is a 3412 MiB model riding under every unrelated
    test that follows.

    This is the regression test for three CI cycles spent on the wrong theory.
    Runs 24, 25 and 26 each moved `malloc_trim` to a different moment and each
    measured the same unmoved floor, because the model was never free memory
    glibc was declining to return — it was memory Python could still reach, and
    no allocator call hands back a live object.

    Asserted as reachability rather than as resident bytes: RSS is a fact about
    an allocator and a platform, and it is what made the real cause ambiguous for
    three runs. Whether the engine is reachable is the same answer on every
    machine, which is why this one runs on a workstation in milliseconds.
    """
    held = _served_then_dropped()

    reclaim()

    assert held() is None, (
        "an app the suite dropped is still holding its engine — if FastAPI moved "
        "the caches `conftest._endpoint_memos` drops, the build runner is about "
        "to be OOM-killed carrying a model no test is using"
    )


#: The conftest whose hooks are under test, and the root that makes it importable.
_CONFTEST = Path(__file__).with_name("conftest.py")
_ROOT = Path(__file__).parents[1]

#: [LAW:no-shared-mutable-globals] The outer run's instrumentation, which the
#: nested one must not inherit. The `tests` job exports both: `PYTEST_ADDOPTS`
#: would load `tests.rsscurve` in the child too -- `_ROOT` is on its PYTHONPATH
#: and `tests` needs no `__init__.py` to import from there -- and that plugin's
#: `pytest_configure` reopens `RSS_CURVE_OUT` with mode "w". Measured: the child
#: truncates the parent's curve and leaves its own two throwaway nodeids in it,
#: destroying the row the outer run had already written. This file sorts late,
#: so an OOM would be explained by a file describing a different suite.
_NOT_INHERITED = ("PYTEST_ADDOPTS", "RSS_CURVE_OUT")

#: Loaded by path under a name of its own, so the throwaway suite drives the real
#: hooks rather than a copy of them: a `hookimpl` marker lives on the function
#: object, so re-exporting the functions carries wrapper-vs-plain across with
#: them, which is the whole property being measured.
#:
#: `logstart` and `logfinish` are this probe's own instrumentation, not part of
#: what is under test. They are here because "a reclaim happened later" is not
#: the claim -- every phase reclaims, so an unbounded search finds a reclaim
#: belonging to the next phase and passes through the bug. These two bracket
#: each test, which is what makes the windows below closed.
_PROBE_CONFTEST = """\
import importlib.util
import sys

_spec = importlib.util.spec_from_file_location("reclaim_hooks_under_test", {conftest!r})
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

pytest_runtest_setup = _module.pytest_runtest_setup
pytest_runtest_teardown = _module.pytest_runtest_teardown

_reclaim = _module.reclaim


def _note(what):
    with open({log!r}, "a") as handle:
        handle.write(what + "\\n")


def _recorded():
    _note("reclaim")
    _reclaim()


_module.reclaim = _recorded


def pytest_runtest_logstart(nodeid, location):
    _note("start:" + nodeid.rsplit("::", 1)[-1])


def pytest_runtest_logfinish(nodeid, location):
    _note("finish:" + nodeid.rsplit("::", 1)[-1])
"""

#: Two scopes, because a test releases more than one kind of thing: pytest tears
#: the module scope down inside the teardown of the module's last test, once it
#: knows the next test is elsewhere, so both land in the same window here.
_PROBE_FIRST = """\
import pytest


def _note(what):
    with open({log!r}, "a") as handle:
        handle.write(what + "\\n")


@pytest.fixture(scope="module")
def held_for_the_module():
    yield object()
    _note("module-let-go")


@pytest.fixture
def held_for_the_test():
    yield object()
    _note("test-let-go")


def test_holds_both(held_for_the_module, held_for_the_test):
    pass
"""

#: A second test in a second module, so there is a setup that follows a teardown
#: -- the moment the setup hook is answerable for, and a body to close its window.
_PROBE_SECOND = """\
def _note(what):
    with open({log!r}, "a") as handle:
        handle.write(what + "\\n")


def test_lets_the_previous_module_go():
    _note("second-body")
"""


def test_a_reclaim_follows_every_release_rather_than_preceding_it(tmp_path):
    """[LAW:behavior-not-structure] The hooks' ordering, measured by running them.

    `test_an_app_the_suite_has_finished_with_stops_holding_its_engine` calls
    `reclaim` directly, so it says nothing about *when* the suite calls it -- and
    when is the part that cost three CI rounds. Run 24 died with the fix already
    written: a plain `pytest_runtest_teardown` in a conftest is registered after
    the builtin hooks and so runs *before* them, trimming a heap whose fixtures
    had not been torn down yet. Both that and the second hook at setup are
    recorded in `conftest`'s docstrings, where nothing enforces them, and a
    `wrapper=True` quietly dropped to a plain hookimpl reintroduces the OOM while
    every other test in this file still passes.

    Each assertion is a *closed* window -- release to the next phase boundary,
    not "somewhere later". Every phase reclaims, so an open-ended search finds
    the next phase's reclaim and passes through the very bug this is here for.
    That is not hypothetical: the first draft of this test asserted exactly that
    and survived having the teardown hook flattened to a plain one.
    """
    log = tmp_path / "order"
    (tmp_path / "conftest.py").write_text(
        _PROBE_CONFTEST.format(conftest=str(_CONFTEST), log=str(log))
    )
    (tmp_path / "test_first.py").write_text(_PROBE_FIRST.format(log=str(log)))
    (tmp_path / "test_second.py").write_text(_PROBE_SECOND.format(log=str(log)))

    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
        cwd=tmp_path,
        env={
            **{k: v for k, v in os.environ.items() if k not in _NOT_INHERITED},
            "PYTHONPATH": str(_ROOT),
        },
        capture_output=True,
        text=True,
        # Bounded, because an unbounded hang here is indistinguishable from the
        # OOM this file exists to make legible: it would surface as the whole
        # `tests` job hitting its "something hung" bound, naming nothing. The
        # nested run measures 0.38s on a workstation; 120 is room for a 2-cpu
        # runner sharing one daemon with three other jobs, and still two minutes
        # rather than twelve. [LAW:no-silent-failure]
        timeout=120,
    )
    assert run.returncode == 0, f"the probe suite itself failed:\n{run.stdout}\n{run.stderr}"

    order = log.read_text().split()

    def window(opens, closes):
        assert opens in order and closes in order, (
            f"the probe never reached {opens!r}..{closes!r}, so it measured nothing: {order}"
        )
        return order[order.index(opens) : order.index(closes)]

    assert "reclaim" in window("test-let-go", "finish:test_holds_both"), (
        "a test's fixtures were let go and its own teardown ended without a "
        "reclaim -- the teardown hook has stopped being a `wrapper=True` "
        "hookimpl that yields first, so it now runs before the teardown it "
        f"exists to collect, on a heap still holding everything. Run 24: {order}"
    )

    assert "reclaim" in window("start:test_lets_the_previous_module_go", "second-body"), (
        "a test was set up and reached its body without a reclaim -- the setup "
        "hook is gone or no longer wraps, so whatever the previous module let go "
        "of is never handed back and this test allocates on top of it. This is "
        f"the second hook, which runs 24 and 25 both needed: {order}"
    )
