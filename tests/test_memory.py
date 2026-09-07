"""What this process reads as its own memory ceiling.

The number every concurrency decision downstream is made from, so the shapes a
real kernel writes are driven directly rather than approximated. Each case here
is a string cgroup actually produces: v2's `max`, v1's page-rounded sentinel, a
plain byte count, and neither layout being present at all.

[FRAMING:representation] The dangerous direction is one-way. Reading a limit as
unconfined loses a refusal that should have fired — quietly, since unconfined is
the answer that declines to refuse anything, and the symptom arrives later as a
SIGKILL with no traceback. Reading unconfined as a limit merely refuses a boot
that would have been fine, loudly, with the number in the message. So the tests
that matter most are the ones pinning that a real limit stays a real limit.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from elvenspeak import memory

WORKFLOW = Path(__file__).parent.parent / ".gitea" / "workflows" / "publish-image.yaml"


@pytest.fixture(autouse=True)
def _unconfined():
    """Overrides conftest's stub, which would otherwise answer for the subject.

    Every other module wants `memory.limit` pinned to unconfined so the suite does
    not inherit the runner's cgroup state. This module IS `memory.limit`, so the
    stub would make its tests assert against the stub and pass no matter what the
    real function did — green, and measuring nothing.
    """
    yield


def test_a_v2_limit_is_read_as_that_many_bytes():
    assert memory._parsed("2147483648\n") == 2147483648


def test_v2_says_max_in_words_when_nothing_caps_the_process():
    assert memory._parsed("max\n") is memory.Unconfined.UNCONFINED


def test_a_v1_limit_is_read_as_that_many_bytes():
    """v1 spells a real cap the same way v2 does, so it needs no separate arm."""
    assert memory._parsed("2147483648") == 2147483648


def test_v1s_uncapped_sentinel_is_not_read_as_an_enormous_limit():
    """[LAW:no-silent-failure] The number v1 writes instead of a word.

    Read literally it is about 8 exbibytes, which is not a lie that fails — every
    ceiling fits under it, so the refusal simply never fires and a v1 host looks
    generously provisioned forever.
    """
    assert memory._parsed("9223372036854771712") is memory.Unconfined.UNCONFINED


def test_a_word_this_was_not_written_against_is_uncapped_rather_than_invented():
    """The unrecognised case can only ever invent a ceiling nobody set."""
    assert memory._parsed("unlimited") is memory.Unconfined.UNCONFINED
    assert memory._parsed("") is memory.Unconfined.UNCONFINED


def test_neither_layout_present_is_an_answer_and_not_a_raise(tmp_path):
    """macOS, bare metal, and `docker run` without `--memory` are all this case."""
    absent = (tmp_path / "no-v2", tmp_path / "no-v1")
    assert memory.limit(absent) is memory.Unconfined.UNCONFINED


def test_a_single_readable_layout_decides(tmp_path):
    """A host presents one layout; the absent one contributes nothing."""
    v1 = tmp_path / "v1"
    v1.write_text("4294967296")
    assert memory.limit((tmp_path / "absent-v2", v1)) == 4294967296


def test_the_tightest_limit_wins_whichever_order_it_is_found_in(tmp_path):
    """[FRAMING:representation] The kernel holds a process to the smallest cap.

    Asserted in both orders, because taking the FIRST readable answer passes one
    of them by accident -- and the accident is the direction that loses a limit.
    """
    tight, loose = tmp_path / "tight", tmp_path / "loose"
    tight.write_text("2147483648")
    loose.write_text("4294967296")
    assert memory.limit((tight, loose)) == 2147483648
    assert memory.limit((loose, tight)) == 2147483648


def test_unconfined_is_not_a_number_any_arithmetic_can_reach():
    """[LAW:types-are-the-program] The point of the enum, pinned.

    Spelled as `None` or `0`, "no limit" would survive `limit or 0` and arrive in
    the comparison as the smallest possible ceiling — an unconfined laptop
    refusing every concurrency there is. A caller cannot do arithmetic on this by
    accident; it has to branch on it.
    """
    assert not isinstance(memory.Unconfined.UNCONFINED, int)
    with pytest.raises(TypeError):
        memory.Unconfined.UNCONFINED > 1  # noqa: B015


def test_the_workflow_reads_the_same_two_files_this_module_does():
    """[LAW:one-source-of-truth] Where the limit lives, spelled once.

    `publish-image.yaml` reads both layouts to explain an exit 137, and this
    module reads them to refuse the concurrency that causes one. Two spellings of
    the same kernel paths drift in the direction where one of them consults a file
    nothing writes any more — and that failure is invisible, because an absent
    file is a legitimate answer here.

    Read off the workflow file for the reason `tests/test_workflow.py` does: it is
    what act_runner executes.
    """
    workflow = WORKFLOW.read_text()
    for path in memory.LIMIT_FILES:
        assert str(path) in workflow, (
            f"{path} is read by elvenspeak.memory but named nowhere in "
            f"{WORKFLOW.name}; the two have drifted about where the limit lives"
        )


def test_the_workflow_names_no_cgroup_memory_file_this_module_misses():
    """The other direction, which is the one that catches a layout being added."""
    named = set(re.findall(r"/sys/fs/cgroup/\S*memory[\w./]*", WORKFLOW.read_text()))
    assert named == {str(path) for path in memory.LIMIT_FILES}


def test_a_limit_that_exists_and_cannot_be_read_is_reported_rather_than_assumed(
    tmp_path, caplog
):
    """[LAW:no-silent-failure] The one case where this module could cause the OOM.

    Absent and unreadable end at the same answer -- unconfined -- and unconfined is
    the answer that declines to refuse anything. So a deployment that really is
    capped, whose limit file this process cannot read, is served as uncapped by the
    very module that exists to stop that. It cannot be refused (there is no number
    to refuse against), so it is said out loud.

    A directory rather than a `chmod 000` file, for the reason
    `tests/test_voices.py` gives at the equivalent spot: the gitea runner executes
    this suite as root (`user: root (uid 0)`, printed by the publish workflow) and
    root reads a mode-000 file happily, so that version would fail on the one
    runner that matters. `read_text()` on a directory raises `IsADirectoryError` —
    an `OSError`, and not a `FileNotFoundError`, so it lands on exactly the arm
    under test — and it refuses whoever asks.
    """
    unreadable = tmp_path / "memory.max"
    unreadable.mkdir()

    with caplog.at_level(logging.WARNING, logger="elvenspeak.memory"):
        assert memory.limit((unreadable,)) is memory.Unconfined.UNCONFINED

    assert any(r.levelno == logging.WARNING for r in caplog.records), (
        "a limit file that exists and cannot be read was treated as no limit "
        "without saying so -- the silent path this module exists to close"
    )
    assert str(unreadable) in caplog.text


def test_a_layout_that_is_simply_absent_does_not_warn(tmp_path, caplog):
    """The ordinary case, which must stay quiet or the warning means nothing.

    Exactly one cgroup layout exists on a given host and neither exists on macOS,
    so warning on absence would fire on every developer machine and on the
    not-present half of every real one -- and a warning that fires always is a
    warning nobody reads when it finally matters.
    """
    with caplog.at_level(logging.WARNING, logger="elvenspeak.memory"):
        assert memory.limit((tmp_path / "absent",)) is memory.Unconfined.UNCONFINED
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# ------------------------------- whose cgroup the mount root actually answers for

# The mount root is this process's own limit only under a private cgroup
# namespace. These pin the resolution against real `/proc/self/cgroup` text,
# because the failure it prevents is the silent one: on a host namespace the root
# read SUCCEEDS and reports the host's uncapped value, so nothing warns and a
# confined deployment is served as unconfined.


def test_a_private_namespace_resolves_to_the_root_and_changes_nothing():
    """Docker's default on a v2 host writes `0::/`, and the root IS ours there."""
    assert memory._own("0::/\n") == ()


def test_a_host_namespace_resolves_to_this_process_own_subtree():
    """`--cgroupns=host`: the root is the host's, and ours is named in the line."""
    assert memory._own("0::/docker/abc123\n") == (
        Path("/sys/fs/cgroup/docker/abc123/memory.max"),
        Path("/sys/fs/cgroup/docker/memory.max"),
    )


def test_every_ancestor_is_resolved_and_not_only_the_leaf():
    """The kernel enforces `memory.max` down the whole chain, not just at the leaf.

    A leaf that says `max` beneath a slice carrying the real ceiling is still
    confined by that slice, so reading only the leaf would report unconfined for a
    process the kernel is actively capping.
    """
    assert memory._own("0::/a/b/c\n") == (
        Path("/sys/fs/cgroup/a/b/c/memory.max"),
        Path("/sys/fs/cgroup/a/b/memory.max"),
        Path("/sys/fs/cgroup/a/memory.max"),
    )


def test_a_v1_line_resolves_through_the_memory_controller_only():
    """v1 writes one line per controller and only the memory one is ours."""
    text = "7:memory:/docker/abc\n3:cpu,cpuacct:/docker/abc\n"
    assert memory._own(text) == (
        Path("/sys/fs/cgroup/memory/docker/abc/memory.limit_in_bytes"),
        Path("/sys/fs/cgroup/memory/docker/memory.limit_in_bytes"),
    )


def test_a_nomad_task_scope_resolves():
    """The shape this service actually deploys into."""
    assert memory._own("0::/nomad.slice/elvenspeak.scope\n") == (
        Path("/sys/fs/cgroup/nomad.slice/elvenspeak.scope/memory.max"),
        Path("/sys/fs/cgroup/nomad.slice/memory.max"),
    )


def test_a_leaf_that_declares_no_limit_does_not_hide_a_tighter_one_beside_it(
    tmp_path,
):
    """A `max` leaf must not stop the search, nor answer for the files after it.

    Not the ancestor walk end to end -- this hands `limit` an explicit pair and so
    never reaches `_own`; `test_the_roots_are_consulted_alongside_the_resolved_cgroup`
    below is what covers the resolution. What this pins is the arm beside it: that
    `tightest` still being UNCONFINED when the first real number arrives takes that
    number, rather than the `max` before it winning by arriving first.
    """
    leaf, ancestor = tmp_path / "leaf", tmp_path / "ancestor"
    leaf.write_text("max")
    ancestor.write_text("2147483648")

    assert memory.limit((leaf, ancestor)) == 2147483648


def test_the_processes_own_cgroup_outranks_an_uncapped_root(tmp_path, monkeypatch):
    """The root would answer, and answer wrongly.

    A host-namespace root says `max` and a read of it succeeds, so a search that
    took the first readable answer would return unconfined with nothing amiss to
    report -- the silent misdetection this resolution exists to prevent.
    """
    own = tmp_path / "own"
    own.write_text("2147483648")
    root = tmp_path / "root"
    root.write_text("max")
    monkeypatch.setattr(memory, "_own", lambda _: (own,))
    monkeypatch.setattr(memory, "LIMIT_FILES", (root,))
    proc = tmp_path / "proc"
    proc.write_text("0::/whatever\n")
    monkeypatch.setattr(memory, "PROC_SELF_CGROUP", proc)

    assert memory.limit() == 2147483648


def test_no_proc_self_cgroup_falls_back_to_the_roots(tmp_path, monkeypatch):
    """macOS, and anywhere else without procfs: there is no subtree to resolve.

    Called with no argument, so the function reads the module attribute -- the
    path `limit()` itself takes. This assertion alone cannot prove the attribute
    was consulted, since ANY unreadable source yields the roots; the test below is
    what pins that, and the two are only meaningful together.
    """
    monkeypatch.setattr(memory, "PROC_SELF_CGROUP", tmp_path / "absent")
    assert memory._candidates() == memory.LIMIT_FILES


def test_the_default_source_is_the_module_attribute_and_is_really_read(
    tmp_path, monkeypatch
):
    """The load-bearing half: a patched attribute that CHANGES the answer.

    Pointing the attribute at an absent file proves nothing on its own -- every
    unreadable source falls back to the roots, so that assertion holds whatever
    the function actually read. Pointing it at a file whose contents can only come
    from being parsed is what makes the patch bite.
    """
    proc = tmp_path / "cgroup"
    proc.write_text("0::/nomad.slice/elvenspeak.scope\n")
    monkeypatch.setattr(memory, "PROC_SELF_CGROUP", proc)

    assert memory._candidates()[0] == Path(
        "/sys/fs/cgroup/nomad.slice/elvenspeak.scope/memory.max"
    )


def test_a_malformed_line_is_skipped_rather_than_guessed_at():
    """A line this was not written against can only invent a path."""
    assert memory._own("garbage\n0::/real\n") == (
        Path("/sys/fs/cgroup/real/memory.max"),
    )


def test_an_unreadable_candidate_does_not_end_the_search(tmp_path, caplog):
    """The `continue` is the fallback the docstrings call important.

    A single-element tuple cannot tell `continue` from an early return -- both
    answer unconfined -- so this pairs an unreadable candidate with a readable,
    meaningful one. If a refactor short-circuited on `OSError`, the real limit
    below it would be silently lost, which is the failure this module is about.
    """
    unreadable = tmp_path / "unreadable"
    unreadable.mkdir()
    readable = tmp_path / "readable"
    readable.write_text("2147483648")

    with caplog.at_level(logging.WARNING, logger="elvenspeak.memory"):
        assert memory.limit((unreadable, readable)) == 2147483648

    assert str(unreadable) in caplog.text, (
        "the unreadable candidate must still be reported -- it may have held a "
        "tighter limit than the one that was found"
    )


def test_an_unreadable_proc_self_cgroup_is_reported(tmp_path, caplog, monkeypatch):
    """[LAW:no-silent-failure] The same split, at the step that picks the files.

    If this is unreadable the subtree cannot be resolved and only the mount roots
    are consulted -- and in a host namespace those are someone else's and usually
    uncapped. That is the module's own silent misdetection, reached one layer above
    where it was closed, so it does not pass quietly.
    """
    unreadable = tmp_path / "cgroup"
    unreadable.mkdir()
    monkeypatch.setattr(memory, "PROC_SELF_CGROUP", unreadable)

    with caplog.at_level(logging.WARNING, logger="elvenspeak.memory"):
        assert memory._candidates() == memory.LIMIT_FILES

    assert str(unreadable) in caplog.text


def test_an_absent_proc_self_cgroup_does_not_warn(tmp_path, caplog, monkeypatch):
    """macOS and anything else without procfs: expected, so it must stay quiet."""
    monkeypatch.setattr(memory, "PROC_SELF_CGROUP", tmp_path / "absent")
    with caplog.at_level(logging.WARNING, logger="elvenspeak.memory"):
        assert memory._candidates() == memory.LIMIT_FILES
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# ------------------------------------- the roots, on the shape this really deploys


def test_the_roots_are_consulted_alongside_the_resolved_cgroup(tmp_path, monkeypatch):
    """[LAW:no-silent-failure] The production path, and the one term nothing pinned.

    On `--cgroupns=private` -- Docker's default on a v2 host, which is what this
    service deploys as -- `/proc/self/cgroup` reads `0::/`, so `_own` correctly
    returns nothing and the mount roots are the ONLY candidates that hold the
    limit. Dropping `+ LIMIT_FILES` from `_candidates` therefore leaves every other
    test in this file green while answering "unconfined" for every containerized
    deployment there is.

    That is one edit away from the exact silent misdetection this module exists to
    close, and the wording of `_candidates`' own docstring invites it -- "the roots
    are all there ever was to ask" reads as though they were only the fallback. So
    the roots are asserted to be there on the resolved path too.
    """
    proc = tmp_path / "cgroup"
    proc.write_text("0::/nomad.slice/elvenspeak.scope\n")
    monkeypatch.setattr(memory, "PROC_SELF_CGROUP", proc)

    candidates = memory._candidates()

    assert candidates[0] == Path(
        "/sys/fs/cgroup/nomad.slice/elvenspeak.scope/memory.max"
    )
    for root in memory.LIMIT_FILES:
        assert root in candidates, (
            f"{root} is missing from the candidates; under a private cgroup "
            f"namespace the roots are the only files that hold the limit, so "
            f"dropping them answers 'unconfined' for every container"
        )


def test_a_private_namespace_still_asks_the_roots(tmp_path, monkeypatch):
    """The same point at the exact input production presents: `0::/`."""
    proc = tmp_path / "cgroup"
    proc.write_text("0::/\n")
    monkeypatch.setattr(memory, "PROC_SELF_CGROUP", proc)

    assert memory._candidates() == memory.LIMIT_FILES


def test_v1s_uncapped_sentinel_is_recognised_at_every_page_size():
    """The sentinel is `LONG_MAX` rounded to a page, so its value varies by kernel.

    Matching the 4 KiB spelling exactly reads a 64 KiB host's "unlimited" as a real
    8-exbibyte ceiling, and that refuses a deployment which was fine -- loudly and
    wrongly, on a machine nobody tested. aarch64 and ppc64le kernels really do use
    64 KiB pages.
    """
    for page in (4096, 16384, 65536):
        sentinel = ((1 << 63) - 1) // page * page
        assert memory._parsed(str(sentinel)) is memory.Unconfined.UNCONFINED, (
            f"the v1 unlimited sentinel for a {page}-byte page ({sentinel}) was "
            f"read as a real limit"
        )


def test_a_dead_cgroups_deleted_suffix_is_not_part_of_its_name():
    """The kernel appends this when a task still names a removed cgroup."""
    assert memory._own("0::/nomad.slice/elvenspeak.scope (deleted)\n") == (
        Path("/sys/fs/cgroup/nomad.slice/elvenspeak.scope/memory.max"),
        Path("/sys/fs/cgroup/nomad.slice/memory.max"),
    )


def test_a_resolved_cgroup_whose_files_are_all_missing_is_reported(
    tmp_path, monkeypatch, caplog
):
    """[LAW:no-silent-failure] Confined, and unfindable, is not "nothing caps me".

    `/proc/self/cgroup` says this process is in a non-root cgroup and not one of
    that cgroup's limit files exists -- which happens when /sys/fs/cgroup is not
    mounted in this mount namespace, or is mounted where these paths do not name.
    Two pieces of evidence, and answering "unconfined" quietly discards both.
    """
    proc = tmp_path / "cgroup"
    proc.write_text("0::/nomad.slice/elvenspeak.scope\n")
    monkeypatch.setattr(memory, "PROC_SELF_CGROUP", proc)
    monkeypatch.setattr(memory, "LIMIT_FILES", (tmp_path / "absent-root",))

    with caplog.at_level(logging.WARNING, logger="elvenspeak.memory"):
        assert memory.limit() is memory.Unconfined.UNCONFINED

    assert any(r.levelno == logging.WARNING for r in caplog.records), (
        "a process that positively reported being in a non-root cgroup, whose "
        "limit files were all missing, was called unconfined without comment"
    )


def test_a_machine_with_no_cgroups_at_all_stays_quiet(tmp_path, monkeypatch, caplog):
    """macOS: no /proc/self/cgroup, no roots, and nothing worth saying about it."""
    monkeypatch.setattr(memory, "PROC_SELF_CGROUP", tmp_path / "absent")
    monkeypatch.setattr(memory, "LIMIT_FILES", (tmp_path / "absent-root",))

    with caplog.at_level(logging.WARNING, logger="elvenspeak.memory"):
        assert memory.limit() is memory.Unconfined.UNCONFINED

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
