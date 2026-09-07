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

import logging
from collections.abc import Iterable
from enum import Enum
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

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

#: Where this process's own cgroup is named, relative to those mounts.
PROC_SELF_CGROUP = Path("/proc/self/cgroup")


def _own(text: str) -> tuple[Path, ...]:
    """The limit files for the cgroup `/proc/self/cgroup` says this process is in.

    The mount root answers for this process only when it has a cgroup namespace of
    its own — Docker's default on a v2 host. Under `--cgroupns=host`, on a v1 host,
    or under Nomad's exec and raw_exec drivers, the root files are the HOST's and
    typically say `max`, while the task's real limit sits in a subtree. That is the
    worst failure available to this module: the read SUCCEEDS, the answer is
    uncapped, no warning fires because nothing went wrong, and a genuinely confined
    deployment is served as unconfined by the code written to stop exactly that.

    So the subtree is asked first and the root second. It is a generalisation
    rather than a replacement: with a private namespace the v2 line reads `0::/`,
    the resolved path collapses onto the root file, and nothing changes where
    nothing was wrong.

    v2 writes one line, `0::<path>`. v1 writes one per controller,
    `<id>:<controllers>:<path>`, and only the one listing `memory` is ours.
    """
    found: list[Path] = []
    for line in text.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        _, controllers, path = fields
        relative = path.strip().lstrip("/")
        if not relative:
            # The root, which LIMIT_FILES already names.
            continue
        if not controllers:
            found.append(Path("/sys/fs/cgroup") / relative / "memory.max")
        elif "memory" in controllers.split(","):
            found.append(
                Path("/sys/fs/cgroup/memory") / relative / "memory.limit_in_bytes"
            )
    return tuple(found)


def _candidates(proc_self_cgroup: Path | None = None) -> tuple[Path, ...]:
    """Every file that might hold this process's limit, most specific first.

    Absent `/proc/self/cgroup` — macOS, and anywhere else without procfs — is not a
    failure: it means there is no subtree to resolve, and the roots are all there
    ever was to ask.

    The default is resolved on the call rather than bound to the signature, so the
    module's paths are read when they are used and not snapshotted at import
    ([LAW:no-ambient-temporal-coupling]) — a default bound at definition time is a
    copy of the world taken before anyone asked for it, and it silently outlives
    every later change to the original.
    """
    source = PROC_SELF_CGROUP if proc_self_cgroup is None else proc_self_cgroup
    try:
        text = source.read_text()
    except OSError:
        return LIMIT_FILES
    return _own(text) + LIMIT_FILES

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


def limit(files: Iterable[Path] | None = None) -> Limit:
    """This process's own memory ceiling, or that it has none.

    The first readable file decides, and [`_candidates`] orders them so that this
    process's own cgroup is asked before the mount root — see [`_own`] for why the
    root is not automatically ours. A file that is not there is not an error —
    exactly one layout exists on a given host, most subtree paths do not exist, and
    neither exists on macOS — which is why absence continues the search and an
    exhausted search is [`Unconfined.UNCONFINED`] rather than a raise.

    [LAW:no-silent-failure] Absence and unreadability are told apart, because they
    are the same answer and not the same event. Both end as unconfined, and
    unconfined is the answer that declines to refuse anything — so a limit that
    exists and could not be read is a deployment that believes it is capped being
    served as uncapped, which is the OOM this module exists to prevent, arrived at
    through the module itself. It is a warning naming the file. A layout that is
    simply not on this host is the ordinary case and says so at INFO.

    Reported here rather than by the caller: the read is the boundary, so the
    account of what was read belongs beside it ([LAW:effects-at-boundaries]), and
    reporting from one caller would leave the next one free to read in silence.
    """
    for path in _candidates() if files is None else files:
        try:
            text = path.read_text()
        except FileNotFoundError:
            continue
        except OSError as error:
            _LOGGER.warning(
                "cannot read %s, so this process is treated as having no memory "
                "limit; if it does have one, nothing here will refuse a synthesis "
                "ceiling too wide for it: %s",
                path,
                error,
            )
            continue
        limit = _parsed(text)
        _LOGGER.info("memory limit: %s says %r", path, text.strip())
        return limit
    _LOGGER.info("memory limit: none -- no cgroup layout here caps this process")
    return Unconfined.UNCONFINED
