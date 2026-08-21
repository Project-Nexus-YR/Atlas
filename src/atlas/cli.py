"""Command line entry point.

One command, one table. ``stdout`` carries the table and nothing else, so a run
can be piped or diffed without narration contaminating it; progress goes to
stderr through :mod:`atlas.logging`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from atlas.config import ExperimentConfig, WorldConfig, settings
from atlas.experiment.compare import format_table, run_comparison
from atlas.logging import configure_logging

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atlas",
        description="Compare the Knowledge Evolution Engine against a Vector-RAG baseline.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the world and the encoder.")
    parser.add_argument("--top-k", type=int, default=5, help="Concepts each arm may return.")
    parser.add_argument("--tasks", type=int, default=200, help="Tasks requested from the suite.")
    parser.add_argument("--subjects", type=int, default=24, help="Areas of the simulated codebase.")
    parser.add_argument("--log-level", default=settings.log_level, help="Stderr log level.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the comparison and print the table. Returns a process exit code."""
    args = _parser().parse_args(argv)
    configure_logging(args.log_level)

    config = ExperimentConfig(
        seed=args.seed,
        top_k=args.top_k,
        world=WorldConfig(subjects=args.subjects, tasks=args.tasks),
    )
    print(format_table(run_comparison(config)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
