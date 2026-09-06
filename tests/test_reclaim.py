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

import weakref

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
