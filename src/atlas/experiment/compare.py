"""Run both arms over one world and render the comparison.

This is the module behind the single seeded command. It builds the world once
and hands the identical task suite and the identical experience stream to both
arms, so the only difference between the two columns is the memory system.

The table is written to be read adversarially. It reports every family
separately rather than only the headline, because two of the four families are
solved by lexical similarity alone and an overall number that hid that would be
flattering rather than informative. It reports ``novel`` - how many retrieved
concepts the graph reached by expansion rather than by lexical seeding - because
a KEE win with ``novel`` at zero is a win for the Jaccard seeder in
:mod:`atlas.memory.store` and not evidence for spreading activation. And it
reports ``revisions``, the number of times the engine judged an experience to
contradict something it already believed; if that is zero, the drift mechanism
never fired and any drift-family advantage came from somewhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

from atlas.bench.tasks import build_tasks
from atlas.bench.world import build_world
from atlas.config import ExperimentConfig
from atlas.embedding import HashingEmbedder
from atlas.experiment.runner import run_arm
from atlas.memory.kee import KnowledgeEvolutionEngine
from atlas.memory.vector_rag import VectorRAGMemory
from atlas.types import ArmResult, TaskFamily

__all__ = ["Comparison", "format_table", "run_comparison"]

_FAMILY_ORDER = (
    TaskFamily.DIRECT,
    TaskFamily.DISTRACTOR,
    TaskFamily.COMPOSITIONAL,
    TaskFamily.DRIFT,
)


@dataclass(frozen=True, slots=True)
class Comparison:
    """Both arms over one world, with the end-state statistics each reported."""

    config: ExperimentConfig
    kee: ArmResult
    vector: ArmResult
    kee_stats: dict[str, float]
    vector_stats: dict[str, float]


def run_comparison(config: ExperimentConfig) -> Comparison:
    """Build one world and drive both arms through it."""
    world = build_world(config.world, config.seed)
    tasks = build_tasks(world, config.world, config.seed)

    kee = KnowledgeEvolutionEngine(config)
    vector = VectorRAGMemory(HashingEmbedder(config.embedding_dim, config.seed), config)

    kee_result = run_arm(kee, world, tasks, config)
    vector_result = run_arm(vector, world, tasks, config)

    return Comparison(
        config=config,
        kee=kee_result,
        vector=vector_result,
        kee_stats=kee.statistics(),
        vector_stats=vector.statistics(),
    )


def _rate(pair: tuple[int, int]) -> float:
    resolved, total = pair
    return 100.0 * resolved / total if total else 0.0


def _novel(result: ArmResult) -> int:
    return sum(outcome.novel_items for outcome in result.outcomes)


def format_table(comparison: Comparison) -> str:
    """Render the comparison as the plain-text table the CLI prints to stdout."""
    kee_families = comparison.kee.by_family()
    vector_families = comparison.vector.by_family()
    config = comparison.config

    lines = [
        f"Atlas 0.2.0  ·  KEE vs Vector-RAG  ·  seed {config.seed}"
        f"  ·  {comparison.kee.total} tasks  ·  top_k {config.top_k}",
        "",
        f"{'family':<16}{'n':>5}{'KEE':>10}{'Vector-RAG':>13}{'delta':>9}",
        f"{'-' * 16}{'-' * 5}{'-' * 10}{'-' * 13}{'-' * 9}",
    ]

    for family in _FAMILY_ORDER:
        kee_pair = kee_families.get(family, (0, 0))
        vector_pair = vector_families.get(family, (0, 0))
        if kee_pair[1] == 0:
            continue
        kee_rate, vector_rate = _rate(kee_pair), _rate(vector_pair)
        lines.append(
            f"{family.value:<16}{kee_pair[1]:>5}{kee_rate:>9.1f}%"
            f"{vector_rate:>12.1f}%{kee_rate - vector_rate:>+9.1f}"
        )

    kee_rate = 100.0 * comparison.kee.resolve_rate
    vector_rate = 100.0 * comparison.vector.resolve_rate
    lines += [
        f"{'-' * 16}{'-' * 5}{'-' * 10}{'-' * 13}{'-' * 9}",
        f"{'overall':<16}{comparison.kee.total:>5}{kee_rate:>9.1f}%"
        f"{vector_rate:>12.1f}%{kee_rate - vector_rate:>+9.1f}",
        "",
        f"KEE         concepts {comparison.kee_stats['concepts']:.0f}"
        f"  ·  revisions {comparison.kee_stats['revisions']:.0f}"
        f"  ·  merges {comparison.kee_stats['merges']:.0f}"
        f"  ·  novel retrievals {_novel(comparison.kee)}",
        f"Vector-RAG  vectors {comparison.vector_stats['vectors']:.0f}"
        f"  ·  dim {comparison.vector_stats['dim']:.0f}"
        f"  ·  novel retrievals {_novel(comparison.vector)}",
        "",
        "Synthetic retrieval benchmark. A score here is evidence about retrieval",
        "and nothing else - no code is written and no patch is executed.",
    ]
    return "\n".join(lines)
