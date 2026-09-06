"""Record RSS and peak RSS around every test, so the suite's high-water mark has
a nodeid attached instead of being one number for the whole process.

This exists because `tests` is being SIGKILLed on the act_runner with no
traceback, and a whole-process figure cannot say which test raised the mark.
`ru_maxrss` is monotonic within a process -- it only ever rises -- so the test
whose `peak_after` exceeds its predecessor's is, exactly, the one that raised it.
`rss_before` is the baseline that test was handed, so `peak_after - rss_before`
is the transient it needed on top of what it inherited.

Appended and flushed per test rather than assembled at session finish, because
the run this explains is one that gets killed: a report written at the end would
be lost in precisely the case it was built for.

That flush is necessary and was not sufficient, which run 22 measured. The OOM
killer takes the whole job container, so the `if: always()` step that reads this
file never ran and the log ended mid-suite at 23% with no report -- a file
flushed inside a container is durable only against the process dying, never
against the container dying with it. So every rise in the high-water mark is
also written to the terminal reporter, which pytest holds open from before
capture and act_runner streams out of the container line by line. A row that has
already left the box cannot be killed with it.

Rows on their own went blind after the mark stopped rising, which run 23
measured: the last row was `test_the_engine_declares_nothing_it_cannot_do` at
5798.2 MiB, and the job then died four minutes and ten seconds later having
printed nothing at all -- because `ru_maxrss` is monotonic, so no later test can
report a *rise*, and the killer strikes between one test's `logstart` and its
`logfinish` rather than at either. A test that dies mid-call is exactly the test
the report needed to name and the only one it structurally could not. So above
`_DANGER_MIB` every test is bracketed by a line on the way in and a line on the
way out: the last `enter` with no `peak` after it names the victim, and the pair
around a test that survives is what a synthesis costs on top of a loaded model --
the figure run 23 died without reporting.

Loaded only when something passes `-p tests.rsscurve`, so the suite it measures is
byte-for-byte the suite both gates already run -- `tests/test_merge_gate.py`
holds the two `run:` lines equal, and an instrument that edited them would be
changing the thing it was brought in to observe.
"""

import csv
import gc
import os
import resource
import subprocess

_LINUX = os.uname().sysname == "Linux"

#: ru_maxrss is KiB on Linux and bytes on Darwin. Read off the platform rather
#: than assumed: the first draft of this file assumed one and reported a 56 MiB
#: process as using 57 GiB, which a known-answer check caught and a plausible
#: number would not have.
_PEAK_SCALE = 1024 if _LINUX else 1024 * 1024

#: Above this the process is close enough to the runner's ceiling that every test
#: is worth two lines. Read off run 23: everything before `test_chatterbox.py`
#: ran under 858 MiB and every test that holds a model ran above 5100, so any
#: floor between those two separates the tests that cannot kill the job from the
#: tests that do -- and 3000 sits far enough from both that a runner with more or
#: less memory than that one does not move which side a test falls on.
_DANGER_MIB = 3000


def _peak_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _PEAK_SCALE


def _rss_mib():
    """Current resident set, from /proc where there is one.

    `ps` costs a fork per test and needs procps in the image; `statm` field 2 is
    resident pages and is always there. The fallback is for the workstation this
    was developed on, which has no /proc.
    """
    if _LINUX:
        with open("/proc/self/statm") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1024 / 1024
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, check=True
    )
    return int(out.stdout.strip()) / 1024


def _path():
    """Where to write. Absent means misconfigured, and says so rather than
    measuring a run whose numbers nobody will find."""
    try:
        return os.environ["RSS_CURVE_OUT"]
    except KeyError:
        raise RuntimeError(
            "rsscurve is loaded but RSS_CURVE_OUT names no file, so this run "
            "would be instrumented and unreadable"
        ) from None


_before = {}
_peak = 0.0
_config = None


def pytest_configure(config):
    global _config
    _config = config
    with open(_path(), "w", newline="") as handle:
        csv.writer(handle).writerow(
            ("nodeid", "rss_before_mib", "rss_after_mib", "peak_after_mib")
        )


def pytest_runtest_logstart(nodeid, location):
    rss = _before[nodeid] = _rss_mib()
    _report_entering_the_danger_zone(rss, nodeid)


def _report_entering_the_danger_zone(rss, nodeid):
    """Name the test on the way in, while there is still a process to name it.

    This is the half `logfinish` cannot cover: the OOM killer lands inside a
    test, so the row written after that test never exists, and run 23 ended with
    four minutes of silence naming nobody. The last `enter` line with no `peak`
    line after it is the victim, and it costs one line per test above the floor.
    """
    if rss <= _DANGER_MIB:
        return
    _write_line(f"[rss] enter {rss:9.1f} MiB  {nodeid}")


def pytest_runtest_logfinish(nodeid, location):
    before = _before.pop(nodeid, float("nan"))
    after, peak = _rss_mib(), _peak_mib()
    # [LAW:one-source-of-truth] Quoted by `csv`, because a nodeid is not a field
    # this file gets to assume the shape of: `[en,es,-read3]` is a real id in
    # `test_chatterbox.py`, and joining on commas shifted every number after it
    # for whoever read the row back. The flush stays -- rows outliving the SIGKILL
    # is the whole reason this file is written a row at a time.
    with open(_path(), "a", newline="") as handle:
        csv.writer(handle).writerow(
            (nodeid, f"{before:.1f}", f"{after:.1f}", f"{peak:.1f}")
        )
        handle.flush()
    _report_if_worth_a_line(nodeid, before, after, peak)


def _report_if_worth_a_line(nodeid, before, after, peak):
    """Say so, in the log, whenever this test is one somebody will want back.

    Two tests are: the one that raised the high-water mark, and the one that ran
    near the ceiling. Every test would be 625 lines of noise; `ru_maxrss` only
    ever rises, so the tests that raise it are a handful and the 50 MiB floor
    keeps ordinary allocation churn off that list. The second rule exists because
    the first goes quiet at exactly the wrong moment -- once a load has set the
    mark at 5798 MiB, a synthesis costing 600 MiB on top of a resident model
    raises nothing and prints nothing, and that synthesis is what killed run 23.
    Above `_DANGER_MIB` the `before -> after` pair is the measurement, whether or
    not the mark moved.

    The reporter is asked for here rather than held from `pytest_configure`,
    because at that point it does not exist yet: a `-p` plugin configures before
    the terminal plugin registers, so a cached one is a cached `None` -- which is
    what the first draft cached, and it raised INTERNALERROR on the first test.
    The plugin manager owns this; keeping a copy meant keeping a wrong one.
    """
    global _peak
    rose = peak > _peak + 50
    if rose:
        _peak = peak
    if not (rose or after > _DANGER_MIB):
        return
    _write_line(
        f"[rss] peak {peak:9.1f} MiB  (rss {before:.1f} -> {after:.1f})"
        f"{_models_still_live(after)}  {nodeid}"
    )


def _models_still_live(rss):
    """How many Chatterbox models Python can still reach, above the floor.

    The one thing four runs of rows cannot distinguish: whether the 3412 MiB
    riding under `test_conformance.py` is a model something still holds, or free
    memory too fragmented for `malloc_trim` to hand back. Those have opposite
    fixes -- drop a reference, or stop asking the allocator for the impossible --
    and runs 24, 25 and 26 spent three cycles not saying which. Counting by type
    name rather than by import, so this stays true whether or not
    `chatterbox-tts` is installed in the environment being measured.

    Walking the whole heap is far too expensive to do 625 times, so it is asked
    only where the answer matters, on the same threshold everything else here
    uses.
    """
    if rss <= _DANGER_MIB:
        return ""
    live = sum(
        1
        for obj in gc.get_objects()
        if type(obj).__name__ in ("ChatterboxEngine", "ChatterboxMultilingualTTS")
    )
    return f"  live={live}"


def _write_line(text):
    """One spelling of "put this where the container cannot take it with it".

    [LAW:one-source-of-truth] Both reporting rules write through here, so the
    lazy `getplugin` -- which is the whole of what the first draft got wrong --
    is resolved in one place rather than once per rule.
    """
    _config.pluginmanager.getplugin("terminalreporter").write_line(text)
