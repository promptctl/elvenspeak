"""ElevenLabs-compatible text-to-speech, served from local voices.

# The whole public surface, and why it is all here

Two things import this package: an entry point that serves the API, and a
project that supplies its own engine and gets the API for free. The second is
the one this list exists for. Everything an outside engine has to name is
exported here, so its obligation is one import line and the answer to "what do I
have to implement" is "the two protocols in this list" rather than a tour of the
package's modules.

An outside engine implements [`Engine`] — voices and the two synthesis calls,
with what each voice can do declared on the [`Voice`] itself — and, if it wants a
deployment to be able to select it by name,
[`Prepared`] and a [`Configure`] that turns an environment into one. It then
builds its own [`Registry`], hands it to [`Settings.from_env`], and passes the
result to [`create_app`]: nothing here has to be edited and nothing has to be
forked, because the registry is an argument rather than an import. There is a
worked example, executed on every test run, in `tests/test_supplied_engine.py`.

Re-exported rather than left to `elvenspeak.engine` and
`elvenspeak.provisioning`. An alias has one definition and so cannot drift from
it, which is what makes this different from a second copy — while a partial list
is a map that lies by omission, since a reader takes what a package root exports
for what a package offers. `tests/test_packaging.py` derives the expected set
from those two modules, so a name added to the seam cannot stay unexported.
"""

from .api import create_app
from .engine import (
    Capability,
    Engine,
    Prosody,
    Silence,
    Speech,
    TimedSpeech,
    Timing,
    Voice,
    spoken_language,
)
from .provisioning import ConfigError, Configure, Prepared, Registry, flag
from .settings import Settings

__all__ = [
    # Serving the ElevenLabs surface over an engine.
    "create_app",
    "Settings",
    # What an engine is: the interface every endpoint is answered through.
    "Engine",
    "Capability",
    "Voice",
    "Prosody",
    "Speech",
    "TimedSpeech",
    "Timing",
    # The spelling `Voice.language` is held in. An engine reads it to answer for
    # a language of its own — the constructor stamps the field either way, so
    # nothing has to call this to be correct.
    "spoken_language",
    # ...and what an engine raises when it made no audio at all, which an outside
    # engine has to be able to raise by name rather than reinvent.
    "Silence",
    # How a deployment gets one: configured, then acquired and opened.
    "Prepared",
    "Configure",
    "Registry",
    "ConfigError",
    "flag",
]
