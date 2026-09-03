"""The engines this build can run, and which one it runs by default.

The one module in the package that names a concrete engine. Both entry points —
`main.py` and `python -m elvenspeak.bake` — import it and hand it to
[`Settings.from_env`]; nothing in the ElevenLabs surface imports it, and
[`elvenspeak.settings`] takes it as an argument rather than reaching for it, so
`api` still cannot reach an engine library through its own configuration import.
`tests/test_encoding.py` checks that rather than trusting this paragraph.

[LAW:one-source-of-truth] Choosing an engine used to be a line of code in each
entry point, which is two answers to one question with nothing keeping them the
same. It is a lookup in this table now, made once, and both entry points work
from the result.

Adding an engine is a line here, a same-named extra in `pyproject.toml`
(`tests/test_packaging.py` fails without it), a leg in the publish matrix
(`tests/test_workflow.py` fails without it), and a case in
`test_conformance.ENGINES` so the contract suite drives it.

Two further edits are owed only by an engine that *has* a third-party library:
its import name in `conftest.ENGINE_LIBRARIES` and that same name in
`test_encoding`'s positive control, which proves the seam check can fail. The
router has no library — it needs nothing beyond the base dependencies — so it
appears in neither, and adding it to either would assert something untrue.
"""

from __future__ import annotations

from . import chatterbox, kokoro, piper, router
from .provisioning import Registry

#: [LAW:one-source-of-truth] The key is the engine's name; no entry repeats it.
#:
#: The first entry is the default, so an unset `ELVENSPEAK_ENGINE` always names a
#: real engine — a separate default setting could name one that is not here.
#: Piper is first because it is roughly twenty-five times faster: measured on
#: this class of machine, Piper runs at RTF ~0.03 against Kokoro's ~0.77. Kokoro
#: sounds considerably better, which is a deployment's choice to make and not one
#: to inherit by being listed.
#:
#: `chatterbox` is third because it is the slow one and the one with a
#: hardware requirement: it clones a single speaker and then speaks any of 23
#: languages in that voice, which is what neither of the two above can do — their
#: per-language voices are different people — and it costs an accelerator and an
#: RTF of ~0.8 on CUDA or ~3 on Apple's GPU to say so. It also names no default
#: device, so a deployment that has not said what hardware it has does not boot.
#: Listed after the two that run anywhere, for the same reason Kokoro is listed
#: after Piper: what a deployment inherits by leaving `ELVENSPEAK_ENGINE` unset
#: should be the cheapest thing that works, not the best thing that might not.
#:
#: `router` is last and is an engine like any other: it satisfies the same
#: protocol, is selected the same way, and is published as its own image. What it
#: has instead of a model file is a fleet of other elvenspeak servers — which
#: [`elvenspeak.engine`] always allowed, and this is the entry that collects on
#: it.
ENGINES: Registry = {
    "piper": piper.configure,
    "kokoro": kokoro.configure,
    "chatterbox": chatterbox.configure,
    "router": router.configure,
}
