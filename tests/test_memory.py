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

import re
from pathlib import Path

import pytest

from elvenspeak import memory

WORKFLOW = Path(__file__).parent.parent / ".gitea" / "workflows" / "publish-image.yaml"


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


def test_the_first_readable_layout_decides(tmp_path):
    """A host presents one layout; the search stops at the one that is there."""
    v1 = tmp_path / "v1"
    v1.write_text("4294967296")
    assert memory.limit((tmp_path / "absent-v2", v1)) == 4294967296


def test_a_present_layout_is_preferred_over_a_later_one(tmp_path):
    v2, v1 = tmp_path / "v2", tmp_path / "v1"
    v2.write_text("2147483648")
    v1.write_text("4294967296")
    assert memory.limit((v2, v1)) == 2147483648


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
