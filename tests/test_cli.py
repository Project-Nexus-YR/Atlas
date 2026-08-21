"""Tests for the command line entry point.

The contract worth pinning is the pipe contract: stdout carries the table and
nothing else, so two runs can be diffed against each other without narration
contaminating the comparison. Everything after that is the argument surface - a
flag that quietly failed to reach the config would change a published number
while the header still printed the value the caller asked for.

Every run here is deliberately tiny. The default suite takes seconds, and none
of these properties needs a big world to show itself.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from atlas import logging as atlas_logging
from atlas.cli import _parser, main

_SMALL = ["--tasks", "12", "--subjects", "4"]


@pytest.fixture(autouse=True)
def _rebind_stderr_logging(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Let each test's ``capsys`` see the narration that test produced.

    ``configure_logging`` attaches its handler once per process and binds
    ``sys.stderr`` as it stands at that moment. Left alone, every test after the
    first would be writing into the first test's closed capture buffer.
    """
    monkeypatch.setattr(atlas_logging, "_CONFIGURED", False)
    logging.getLogger("atlas").handlers.clear()
    yield
    logging.getLogger("atlas").handlers.clear()


def _body(out: str) -> str:
    """The table below the header, which names the seed and would differ trivially."""
    return out.split("\n", 1)[1]


def test_main_returns_zero_and_prints_the_table_to_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(_SMALL)

    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("Atlas 0.2.0")
    assert "overall" in out


def test_stdout_carries_the_table_alone_and_the_narration_goes_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main([*_SMALL, "--log-level", "INFO"])

    captured = capsys.readouterr()
    assert "resolved" not in captured.out
    assert "resolved" in captured.err


def test_the_same_seed_reproduces_the_table_exactly(capsys: pytest.CaptureFixture[str]) -> None:
    main([*_SMALL, "--seed", "7"])
    first = capsys.readouterr().out
    main([*_SMALL, "--seed", "7"])
    second = capsys.readouterr().out

    assert first == second


def test_a_different_seed_changes_the_numbers_and_not_only_the_header(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main([*_SMALL, "--seed", "7"])
    first = capsys.readouterr().out
    main([*_SMALL, "--seed", "8"])
    second = capsys.readouterr().out

    assert _body(first) != _body(second)


def test_the_seed_top_k_and_task_count_reach_the_configuration_the_header_reports(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["--seed", "5", "--top-k", "3", "--tasks", "8", "--subjects", "4"])

    header = capsys.readouterr().out.splitlines()[0]
    assert "seed 5" in header
    assert "8 tasks" in header
    assert "top_k 3" in header


def test_the_subject_count_reaches_the_generated_world(capsys: pytest.CaptureFixture[str]) -> None:
    """The one flag the header does not echo, so it is pinned by its effect instead."""
    main(["--tasks", "8", "--subjects", "4"])
    narrow = capsys.readouterr().out
    main(["--tasks", "8", "--subjects", "8"])
    wide = capsys.readouterr().out

    assert _body(narrow) != _body(wide)


def test_the_parser_defaults_are_the_published_run() -> None:
    args = _parser().parse_args([])

    assert (args.seed, args.top_k, args.tasks, args.subjects) == (42, 5, 200, 24)
