"""A test that fails on purpose, to prove the merge gate blocks a red commit.

This file exists to be deleted. It is the demonstration ticket
piper-routing-7e2.11 asks for: a pull request carrying a failing test, watched
being refused a merge. Nothing here is a real assertion about elvenspeak.
"""


def test_the_merge_gate_refuses_this_commit():
    assert False, "deliberate failure — piper-routing-7e2.11's demonstration"
