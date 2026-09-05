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

Loaded only when something passes `-p rsscurve`, so the suite it measures is
byte-for-byte the suite both gates already run -- `tests/test_merge_gate.py`
holds the two `run:` lines equal, and an instrument that edited them would be
changing the thing it was brought in to observe.
"""

import os
import resource
import subprocess

_LINUX = os.uname().sysname == "Linux"

#: ru_maxrss is KiB on Linux and bytes on Darwin. Read off the platform rather
#: than assumed: the first draft of this file assumed one and reported a 56 MiB
#: process as using 57 GiB, which a known-answer check caught and a plausible
#: number would not have.
_PEAK_SCALE = 1024 if _LINUX else 1024 * 1024


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
    with open(_path(), "w") as handle:
        handle.write("nodeid,rss_before_mib,rss_after_mib,peak_after_mib\n")


def pytest_runtest_logstart(nodeid, location):
    _before[nodeid] = _rss_mib()


def pytest_runtest_logfinish(nodeid, location):
    before = _before.pop(nodeid, float("nan"))
    after, peak = _rss_mib(), _peak_mib()
    with open(_path(), "a") as handle:
        handle.write(f"{nodeid},{before:.1f},{after:.1f},{peak:.1f}\n")
        handle.flush()
    _report_a_rise(nodeid, before, after, peak)


def _report_a_rise(nodeid, before, after, peak):
    """Say so, in the log, whenever this test raised the high-water mark.

    Every test would be 625 lines of noise; `ru_maxrss` only ever rises, so the
    tests that raise it are the whole finding and there are a handful of them.
    The 50 MiB floor is what keeps ordinary allocation churn off the list.

    The reporter is asked for here rather than held from `pytest_configure`,
    because at that point it does not exist yet: a `-p` plugin configures before
    the terminal plugin registers, so a cached one is a cached `None` -- which is
    what the first draft cached, and it raised INTERNALERROR on the first test.
    The plugin manager owns this; keeping a copy meant keeping a wrong one.
    """
    global _peak
    if peak <= _peak + 50:
        return
    _peak = peak
    reporter = _config.pluginmanager.getplugin("terminalreporter")
    reporter.write_line(
        f"[rss] peak {peak:9.1f} MiB  (rss {before:.1f} -> {after:.1f})  {nodeid}"
    )
