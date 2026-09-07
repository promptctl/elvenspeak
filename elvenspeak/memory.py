"""What memory limit this process is running under, if any.

The question this module answers is not "how much memory is there" but "how much
is THIS process allowed", and the two differ by the whole of what a container is.
`/proc/meminfo` reports the host and is the wrong number on every deployment that
matters here; the cgroup reports the confinement and is the right one.

It exists because [`settings`] said it could not be known. Its
[`Settings.concurrent_syntheses`] comment read "the right unit is the deployment's
own memory limit, which this process cannot know", and a CPU-derived default was
justified by that sentence. The sentence was false. What is true, and is what that
comment had generalised from, is that CPython ignores cgroup `cpu.max` — a fact
about CORES that says nothing about memory, since a confined process reads its own
memory ceiling out of the same filesystem it reads everything else from.

The cost of the false version was measured: elvenspeak-piper's default stayed as
wide as its 4-core host (8), and an eight-way burst OOM-killed it against a
2048 MiB limit.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from pathlib import Path

#: cgroup v2 first, then v1. A host presents one layout or the other, so this is a
#: search for the one that is here rather than a preference between two that both
#: are. Matched to the pair `.gitea/workflows/publish-image.yaml` already reads
#: when it explains an exit 137, and held equal to it by `tests/test_memory.py`
#: ([LAW:one-source-of-truth]) — two spellings of where the limit lives is the
#: kind of drift that ends with one of them reading a file the kernel stopped
#: writing.
LIMIT_FILES = (
    Path("/sys/fs/cgroup/memory.max"),
    Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
)

#: cgroup v1 has no word for "uncapped" and writes a number instead: the counter's
#: maximum, rounded down to a page multiple. Anything at or above it is the kernel
#: saying "nothing", not a deployment saying "eight exbibytes", and reading it as a
#: real ceiling would make every v1 host look generously provisioned rather than
#: unconfined. v2 says `max` in words and needs no such threshold.
_V1_UNLIMITED = 0x7FFFFFFFFFFFF000


class Unconfined(Enum):
    """That this process runs under no memory limit of its own.

    [LAW:types-are-the-program] A distinguished value rather than `None`, for the
    same reason [`voices.Substitution`] is one: the caller must handle it, and the
    two obvious sentinels both read as a real number somewhere. `None` invites
    `limit or 0`, and 0 is a limit that nothing fits in — an unconfined laptop
    would refuse every concurrency there is. The type makes the arithmetic
    impossible to reach instead of making it wrong.
    """

    #: No cgroup file capped this process: neither layout is present, or the one
    #: that is says it is uncapped. A development machine, a bare-metal host, and
    #: a container run without `--memory` are all this answer, and it is a real
    #: answer rather than a failure to find one.
    UNCONFINED = "no memory limit of this process's own"


#: A byte count, or that there is none. What [`limit`] returns.
Limit = int | Unconfined


def _parsed(text: str) -> Limit:
    """One cgroup limit file's contents as a limit.

    Pure, and separated from the read so that every shape a real kernel writes can
    be driven directly ([LAW:effects-at-boundaries]) — the alternative is a test
    that proves the arithmetic against a fake filesystem and proves nothing about
    the strings.

    Anything that is not a plain number is uncapped: that is `max` from v2, and it
    is also whatever a layout this was not written against decides to say. Reading
    an unrecognised word as a limit is the failure that bites, since it can only
    ever invent a ceiling that was never set.
    """
    stripped = text.strip()
    if not stripped.isdigit():
        return Unconfined.UNCONFINED
    value = int(stripped)
    return Unconfined.UNCONFINED if value >= _V1_UNLIMITED else value


def limit(files: Iterable[Path] = LIMIT_FILES) -> Limit:
    """This process's own memory ceiling, or that it has none.

    The first readable file decides. A file that is not there is not an error —
    exactly one of these layouts exists on a given host, and neither exists on
    macOS — which is why absence continues the search and an exhausted search is
    [`Unconfined.UNCONFINED`] rather than a raise.

    That is the one place this could go quietly wrong, so it does not go quietly:
    a file present but unreadable would also read as unconfined, and unconfined is
    the answer that declines to refuse anything. `create_app` logs which file was
    consulted and what it said, so a deployment that believes it is capped and is
    being read as unconfined says so at boot rather than at the OOM
    ([LAW:no-silent-failure]).
    """
    for path in files:
        try:
            text = path.read_text()
        except OSError:
            continue
        return _parsed(text)
    return Unconfined.UNCONFINED
