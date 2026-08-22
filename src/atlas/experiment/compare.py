"""Run every arm over one world and render the comparison.

This is the module behind the single seeded command. It builds the world once
and hands the identical task suite and the identical experience stream to every
arm, so the only difference between the columns is the memory system.

The table is written to be read adversarially. It reports every family
separately rather than only the headline, because some families are solved by
lexical similarity alone and an overall number that hid that would be flattering
rather than informative. It reports ``novel`` - how many retrieved concepts the
graph reached by expansion rather than by lexical seeding - because a KEE win
with ``novel`` at zero is a win for the Jaccard seeder in
:mod:`atlas.memory.store` and not evidence for spreading activation. And it
reports ``revisions``, the number of times the engine judged an experience to
contradict something it already believed; if that is zero, the drift mechanism
never fired and any drift-family advantage came from somewhere else.

There is one arm under test and the rest are controls. ``vs best`` is measured
against the strongest control on each row, never against a chosen one, so adding
a control can only make the graph arm's reported margin worse. That is the point
of adding them: a baseline is only worth running if it is allowed to win.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from atlas.bench.tasks import build_tasks
from atlas.bench.world import build_world
from atlas.config import ExperimentConfig
from atlas.embedding import HashingEmbedder
from atlas.experiment.runner import run_arm
from atlas.memory.bm25 import BM25Memory
from atlas.memory.kee import KnowledgeEvolutionEngine
from atlas.memory.vector_rag import VectorRAGMemory
from atlas.types import ArmResult, MemorySystem, TaskFamily

__all__ = ["ArmReport", "Comparison", "format_table", "run_comparison"]

_FAMILY_ORDER = (
    TaskFamily.DIRECT,
    TaskFamily.DISTRACTOR,
    TaskFamily.COMPOSITIONAL,
    TaskFamily.DRIFT,
)

_LABEL_WIDTH = 16
_COUNT_WIDTH = 5
_ARM_WIDTH = 13
_DELTA_WIDTH = 10


@dataclass(frozen=True, slots=True)
class ArmReport:
    """One arm's graded outcomes and the end-state statistics it reported.

    ``label`` is the column heading. It is carried separately from
    ``result.arm`` so the table can read well without renaming the arms, whose
    names appear in log lines that are matched on.
    """

    label: str
    result: ArmResult
    stats: dict[str, float]


@dataclass(frozen=True, slots=True)
class Comparison:
    """Every arm over one world, with the end-state statistics each reported.

    The split between ``subject`` and ``baselines`` is load-bearing rather than
    presentational: it is what defines the delta column, and it records which
    arm the experiment is actually about.
    """

    config: ExperimentConfig
    subject: ArmReport
    baselines: tuple[ArmReport, ...]

    @property
    def arms(self) -> tuple[ArmReport, ...]:
        """Every arm in column order: the subject first, then the controls."""
        return (self.subject, *self.baselines)


def run_comparison(config: ExperimentConfig) -> Comparison:
    """Build one world and drive every arm through it."""
    world = build_world(config.world, config.seed)
    tasks = build_tasks(world, config.world, config.seed)

    def report(label: str, memory: MemorySystem) -> ArmReport:
        result = run_arm(memory, world, tasks, config)
        return ArmReport(label=label, result=result, stats=memory.statistics())

    return Comparison(
        config=config,
        subject=report("KEE", KnowledgeEvolutionEngine(config)),
        baselines=(
            report(
                "Vector-RAG",
                VectorRAGMemory(HashingEmbedder(config.embedding_dim, config.seed), config),
            ),
            report("BM25", BM25Memory(config)),
        ),
    )


def _rate(pair: tuple[int, int]) -> float:
    resolved, total = pair
    return 100.0 * resolved / total if total else 0.0


def _novel(result: ArmResult) -> int:
    return sum(outcome.novel_items for outcome in result.outcomes)


def _rule(arms: int) -> str:
    return "-" * (_LABEL_WIDTH + _COUNT_WIDTH + _ARM_WIDTH * arms + _DELTA_WIDTH)


def _row(label: str, count: int, rates: Sequence[float]) -> str:
    """One table row: the per-arm rates, then the subject's margin over the best control."""
    cells = "".join(f"{rate:>{_ARM_WIDTH - 1}.1f}%" for rate in rates)
    delta = rates[0] - max(rates[1:], default=0.0)
    return f"{label:<{_LABEL_WIDTH}}{count:>{_COUNT_WIDTH}}{cells}{delta:>+{_DELTA_WIDTH}.1f}"


def format_table(comparison: Comparison) -> str:
    """Render the comparison as the plain-text table the CLI prints to stdout."""
    arms = comparison.arms
    families = [arm.result.by_family() for arm in arms]
    config = comparison.config

    heading = "".join(f"{arm.label:>{_ARM_WIDTH}}" for arm in arms)
    lines = [
        f"Atlas 0.2.0  ·  {' vs '.join(arm.label for arm in arms)}  ·  seed {config.seed}"
        f"  ·  {comparison.subject.result.total} tasks  ·  top_k {config.top_k}",
        "",
        f"{'family':<{_LABEL_WIDTH}}{'n':>{_COUNT_WIDTH}}{heading}{'vs best':>{_DELTA_WIDTH}}",
        _rule(len(arms)),
    ]

    for family in _FAMILY_ORDER:
        pairs = [table.get(family, (0, 0)) for table in families]
        if pairs[0][1] == 0:
            continue
        lines.append(_row(family.value, pairs[0][1], [_rate(pair) for pair in pairs]))

    lines += [
        _rule(len(arms)),
        _row(
            "overall",
            comparison.subject.result.total,
            [100.0 * arm.result.resolve_rate for arm in arms],
        ),
        "",
    ]
    for arm in arms:
        statistics = "  ·  ".join(f"{key} {value:.0f}" for key, value in arm.stats.items())
        lines.append(f"{arm.label:<12}{statistics}  ·  novel retrievals {_novel(arm.result)}")
    lines += [
        "",
        "Synthetic retrieval benchmark. A score here is evidence about retrieval",
        "and nothing else - no code is written and no patch is executed.",
    ]
    return "\n".join(lines)
