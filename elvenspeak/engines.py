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

Adding an engine is a line here — plus, in `tests/`, a line in
`test_conformance.ENGINES` so it must pass the contract suite, a line in
`test_encoding._ENGINE_LIBRARIES` so the seam check can see its library, and a
line in that module's positive control.
"""

from __future__ import annotations

from . import piper
from .provisioning import Registry

#: [LAW:one-source-of-truth] The key is the engine's name; no entry repeats it.
#:
#: The first entry is the default, so an unset `ELVENSPEAK_ENGINE` always names a
#: real engine — a separate default setting could name one that is not here.
#: Piper is first because it is roughly twenty times faster than the alternatives
#: measured so far; a better-sounding engine is a deployment's choice to make,
#: not one to inherit by being listed.
ENGINES: Registry = {"piper": piper.configure}
