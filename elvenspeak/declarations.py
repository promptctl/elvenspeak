"""The file an engine declares its ElevenLabs compatibility in.

One engine, one file, named after that engine's key in
[`elvenspeak.engines.ENGINES`]. Two modules ask different questions of it —
[`elvenspeak.voices`] asks which local voice answers for a foreign voice id, and
[`elvenspeak.models`] asks which foreign model ids reach this engine — and they
read one file because it is one declaration: what this engine answers for that
is not its own.

[LAW:effects-at-boundaries] The one place a declaration is read off disk, and
the one place that knows where declarations live. Both readers take the result
as a value, so resolution stays pure and a test can hand either of them a table
without a file existing.

[LAW:one-source-of-truth] Split out of `voices.load_aliases`, which owned the
directory path while it was the only reader. A second reader with its own copy
of that path is a second answer to "where does an engine declare", free to keep
reading a directory the first one has moved on from.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path

from .provisioning import ConfigError

#: One file per engine, each named after that engine's registry key. An engine
#: that declares nothing has no file: nothing is registered centrally to add
#: one, so the filename is the whole of the wiring.
_DIRECTORY = Path(__file__).parent / "aliases"


def _read(engine_name: str) -> Mapping[str, object]:
    """Everything in `engine_name`'s declaration file, or nothing if it has none.

    [LAW:parse-dont-validate] The checkpoint where a file becomes a table, so
    every way that can fail has to leave by the arm every other startup problem
    uses. [`voice_aliases`] and [`model_ids`] each prove the *shape* of what comes
    back and raise [`ConfigError`] naming the file — but a file that never loads
    reaches neither, and the failure escaped past
    [`elvenspeak.settings.reported_or_exit`], which catches `ConfigError` and
    nothing else. One misplaced bracket therefore answered an operator with a
    traceback where every other typo in the same file answers with a sentence.

    `ValueError` rather than `TOMLDecodeError`, because syntax is not the only way
    the parse refuses: `tomllib.load` decodes as UTF-8 before it parses anything,
    so a file saved in another encoding raises `UnicodeDecodeError` — no kind of
    `TOMLDecodeError`, and straight back to being the traceback. Naming the two
    would be an enumeration a third door reopens; that call does nothing but turn
    bytes into a table or refuse, so the base they share is exactly the set of
    ways it can say it could not.

    Opening is inside the block too, and answers in its own sentence. A file that
    cannot be opened and a file that cannot be parsed are not one fact: they send
    the operator to a mount or a permission bit, or to the contents, and one
    message covering both would be the two collapsed into a single value that
    names neither. Both quote what the runtime said — `tomllib` gives the line and
    column it gave up at, the OS gives the errno — because neither is something
    this module can improve on.

    `exists()` still answers the ordinary absence first, since an engine that
    declares nothing has no file and that is not a failure to report.
    """
    declared = _DIRECTORY / f"{engine_name}.toml"
    if not declared.exists():
        return {}
    try:
        with declared.open("rb") as handle:
            return tomllib.load(handle)
    except OSError as unopenable:
        raise ConfigError(
            [f"{engine_name}.toml: could not be opened ({unopenable})"]
        ) from None
    except ValueError as unparseable:
        raise ConfigError(
            [f"{engine_name}.toml: not valid TOML ({unparseable})"]
        ) from None


def voice_aliases(engine_name: str) -> Mapping[str, str]:
    """Foreign voice ids mapped onto the local voice ids they reach.

    Returned unfiltered. Whether a target is a voice the engine actually has is
    [`elvenspeak.voices.load_aliases`]' question, asked where the voice list is.

    [LAW:parse-dont-validate] Shape proved here, for the same reason [`model_ids`]
    proves its own: the annotation is a theorem, and until something checks it a
    table written as `elevenlabs = "oops"` reaches `load_aliases` and dies there
    on a bare `AttributeError`. [`elvenspeak.settings.reported_or_exit`] catches
    [`ConfigError`] and nothing else, so that is the difference between an
    operator reading which file they mistyped and reading a traceback.

    Keys go unchecked because TOML has no other kind: a table's keys are strings
    or the parse already failed, and a guard that cannot fire says nothing.
    """
    declared = _read(engine_name).get("elevenlabs", {})
    if not isinstance(declared, Mapping) or not all(
        isinstance(target, str) for target in declared.values()
    ):
        raise ConfigError(
            [
                f"{engine_name}.toml: elevenlabs must be a table mapping each "
                f"foreign voice id to one local voice id"
            ]
        )
    return declared


def model_ids(engine_name: str) -> tuple[str, ...]:
    """Foreign `model_id` values this engine answers for.

    A list rather than a mapping, because the only engine these could reach is
    the one the file is named after — a value beside each id would be that name
    written a second time, free to name some other engine
    ([LAW:one-source-of-truth]).

    [LAW:no-silent-failure] A malformed declaration is a [`ConfigError`] at
    startup rather than a table that quietly answers for nothing. Every other
    bad-configuration path reports this way, and an engine silently answering
    for no model id is exactly the failure `tests/test_aliases.py` exists to
    catch statically — it must not be reachable at runtime either.
    """
    declared = _read(engine_name).get("elevenlabs_models", [])
    if not isinstance(declared, list) or not all(
        isinstance(item, str) for item in declared
    ):
        raise ConfigError(
            [f"{engine_name}.toml: elevenlabs_models must be a list of model ids"]
        )
    return tuple(declared)
