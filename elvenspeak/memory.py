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
from pathlib import Path, PurePosixPath

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
    """Every cgroup limit file that binds this process, leaf first then ancestors.

    Two reasons the mount root is not simply this process's limit.

    It may not be OURS. The root holds our confinement only when we have a cgroup
    namespace of our own — Docker's default on a v2 host, and not guaranteed
    anywhere else. Under `--cgroupns=host`, on a v1 host, or under Nomad's exec and
    raw_exec drivers, the root files are the HOST's and typically say `max` while
    the task's real limit sits in a subtree.

    And the leaf may not be the WHOLE of it. The kernel enforces the minimum of
    `memory.max` along the entire chain, so a leaf that says `max` under a slice
    carrying the real ceiling is still confined by that slice. Reading only the
    leaf would report unconfined for a process the kernel is actively capping.

    Both are the same failure and it is the worst one available here: the read
    SUCCEEDS, the answer is uncapped, nothing warns because nothing went wrong, and
    a genuinely confined deployment is served as unconfined by the module written
    to stop precisely that. So every file that binds us is returned, and [`limit`]
    takes the tightest — which is what the kernel does.

    v2 writes one line, `0::<path>`. v1 writes one per controller,
    `<id>:<controllers>:<path>`, and only the one listing `memory` is ours.
    """
    found: list[Path] = []
    for line in text.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        _, controllers, path = fields
        if not controllers:
            mount, name = Path("/sys/fs/cgroup"), "memory.max"
        elif "memory" in controllers.split(","):
            mount, name = Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes"
        else:
            continue
        # The leaf, then each ancestor up to but excluding the mount itself, which
        # LIMIT_FILES already names.
        # The kernel appends this to a v2 line when the cgroup has been removed
        # while a task still names it. Left on, it becomes part of the directory
        # name and the leaf silently resolves to nothing.
        relative = PurePosixPath(path.strip().removesuffix(" (deleted)"))
        for parent in (relative, *relative.parents):
            if parent == PurePosixPath("/"):
                continue
            found.append(mount / str(parent).lstrip("/") / name)
    return tuple(found)


def _candidates() -> tuple[Path, ...]:
    """Every file that might hold this process's limit, most specific first.

    Absent `/proc/self/cgroup` — macOS, and anywhere else without procfs — is not a
    failure: it means there is no subtree to resolve, and the roots are all there
    ever was to ask.

    [`PROC_SELF_CGROUP`] is read through the module on every call rather than bound
    into a default argument ([LAW:no-ambient-temporal-coupling]): a default is
    evaluated once at import, which is a copy of the world taken before anyone
    asked for it, and it outlives every later change to the original.
    """
    try:
        text = PROC_SELF_CGROUP.read_text()
    except FileNotFoundError:
        return LIMIT_FILES
    except OSError as error:
        # [LAW:no-silent-failure] The same split [`limit`] makes, at the step that
        # decides which files it will even consult. Unreadable here means the
        # subtree cannot be resolved, so only the roots are asked -- and if this
        # process is in a host namespace those are someone else's and typically
        # uncapped, which is the silent misdetection this module exists to close,
        # reached one layer up from where it was closed.
        _LOGGER.warning(
            "cannot read %s, so this process's own cgroup cannot be resolved and "
            "only the mount roots will be consulted; if those are not this "
            "process's own, a real memory limit will go unseen: %s",
            PROC_SELF_CGROUP,
            error,
        )
        return LIMIT_FILES
    return _own(text) + LIMIT_FILES

#: Above which a number is the kernel saying "nothing", not a deployment saying
#: "four exbibytes". cgroup v1 has no word for uncapped and writes the counter's
#: maximum instead — `LONG_MAX` rounded down to a page multiple, so its exact value
#: DEPENDS ON PAGE SIZE: 0x7FFFFFFFFFFFF000 at 4 KiB, and smaller at 16 or 64 KiB,
#: which aarch64 and ppc64le kernels really use.
#:
#: So this is a plausibility bound and not the sentinel itself. Matching the 4 KiB
#: spelling exactly would read a 64 KiB host's "unlimited" as a real 8 exbibyte
#: ceiling — and that is the one direction that refuses a deployment which was
#: fine, loudly and wrongly, on a machine nobody tested. No real limit approaches
#: this, so no page size can slip past it.
_V1_UNLIMITED = 1 << 62


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

    THE TIGHTEST OF EVERY FILE THAT BINDS US, not the first one found, because
    that is what the kernel enforces: `memory.max` applies down the whole chain, so
    a process is held to the smallest limit any of its cgroups declares. Taking the
    first would report a leaf's `max` while an ancestor was doing the capping.

    Erring tight is deliberate and is the only safe direction. Too tight refuses a
    boot that would have been fine — loudly, with the number in the message, and an
    operator can act on it. Too loose is the SIGKILL with no traceback that this
    module exists to prevent. So a candidate that cannot be read is skipped rather
    than treated as unbounded, and the search continues past it.

    A file that is not there is not an error — exactly one layout exists on a given
    host, most ancestor paths do not exist, and neither exists on macOS — which is
    why absence is quiet and an exhausted search is [`Unconfined.UNCONFINED`]
    rather than a raise. A file that is there and cannot be READ is different: it
    may hold the limit that binds us, so it is a warning naming the file
    ([LAW:no-silent-failure]).

    Reported here rather than by the caller: the read is the boundary, so the
    account of what was read belongs beside it ([LAW:effects-at-boundaries]), and
    reporting from one caller would leave the next one free to read in silence.
    """
    resolved = files is None
    consulted = tuple(_candidates() if resolved else files)
    # Whether any candidate was READ, which is not the same question as whether one
    # yielded a number: a file that exists and says `max` answers "unconfined"
    # truthfully, and must not be reported as a file that could not be found.
    read_any = False
    tightest: Limit = Unconfined.UNCONFINED
    for path in consulted:
        try:
            text = path.read_text()
        except FileNotFoundError:
            continue
        except OSError as error:
            _LOGGER.warning(
                "cannot read %s, so any limit it holds is not being applied; if "
                "this process is confined, nothing here will refuse a synthesis "
                "ceiling too wide for it: %s",
                path,
                error,
            )
            continue
        read_any = True
        found = _parsed(text)
        if not isinstance(found, int):
            continue
        _LOGGER.info("memory limit: %s says %r", path, text.strip())
        tightest = found if not isinstance(tightest, int) else min(tightest, found)
    if isinstance(tightest, int):
        # Which one BINDS, said once. A deep cgroup emits a line per readable
        # ancestor, and an operator reading four numbers still has to be told
        # which of them the kernel will hold this process to.
        _LOGGER.info("memory limit: %d bytes is the tightest that binds", tightest)
    elif resolved and consulted != LIMIT_FILES and not read_any:
        # [LAW:no-silent-failure] Only on the resolved path, because that is where
        # the evidence comes from: `/proc/self/cgroup` positively stated this
        # process is in a non-root cgroup, and NOT ONE of that cgroup's limit files
        # could be read. It says "could not be read" rather than "does not exist"
        # because `read_any` is false for absence AND for a failed read, and a
        # candidate that exists but raises has already logged the real errno just
        # above -- claiming absence here would contradict that line, and of the two
        # the false one is the one that sounds conclusive.
        #
        # Two pieces of evidence -- confined, and unreadable -- and
        # the honest report of them is not "nothing caps this process". It happens
        # when /sys/fs/cgroup is not mounted in this mount namespace, or is mounted
        # somewhere these paths do not name, and the cost of saying nothing is the
        # OOM this module exists to prevent.
        #
        # Gated on nothing having been read, not on nothing having been found. A
        # task with no memory stanza reads its files fine and each says `max`: that
        # is unconfined, correctly detected, and warning at it would cry wolf on a
        # working deployment -- after which the next operator skims past the line
        # in the one case it was written for.
        _LOGGER.warning(
            "this process is in a non-root cgroup but none of its limit files "
            "could be read (%s), so it is being treated as unconfined; if it is "
            "not, nothing here will refuse a synthesis ceiling too wide for it",
            ", ".join(str(path) for path in consulted),
        )
    else:
        _LOGGER.info("memory limit: none -- nothing here caps this process")
    return tightest
