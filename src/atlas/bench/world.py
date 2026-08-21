"""The synthetic world the benchmark runs on.

This is a SYNTHETIC corpus, not a sample of real engineering knowledge. It is
generated so that four retrieval difficulties can be isolated from one another -
direct lookup, association, knowledge drift, and distractor pressure - and so a
gap between two memory arms can be attributed to one of them. It therefore
licenses claims of the form "arm A retrieves better than arm B under drift" and
nothing wider. It says nothing about real repository text, nothing about
end-to-end software-engineering ability, and nothing about corpora whose lexical
statistics differ from this generator's - which are uniform by construction.

Facts read ``{subject} {predicate} {value}``. The value vocabularies are
token-disjoint across predicates, so two facts about one subject overlap on
exactly the subject word. That single property is what lets a task demand a hop
that similarity cannot make.

Everything is a pure function of ``(config, seed)``: no global ``random``, no
``hash()``, no clock, no filesystem. Two runs of one seed produce identical
worlds, so a published number can be re-derived.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from atlas.config import WorldConfig
from atlas.types import Experience, Fact

__all__ = ["World", "build_world"]

_SUBJECTS: tuple[str, ...] = (
    "parser",
    "retry",
    "cache",
    "auth",
    "migration",
    "scheduler",
    "router",
    "serializer",
    "indexer",
    "throttle",
    "session",
    "webhook",
    "queue",
    "logger",
    "validator",
    "uploader",
    "tokenizer",
    "resolver",
    "pool",
    "watcher",
    "encoder",
    "sandbox",
    "tracer",
    "planner",
)

# Value vocabularies share no token with each other, with any predicate name, or
# with any subject name. Two facts about one subject therefore overlap on the
# subject word alone, which is the invariant COMPOSITIONAL tasks are built on.
_PREDICATES: dict[str, tuple[str, ...]] = {
    "attempt limit": (
        "zero attempts",
        "one attempt",
        "two attempts",
        "three attempts",
        "five attempts",
        "eight attempts",
        "twelve attempts",
        "twenty attempts",
    ),
    "timeout window": (
        "half second",
        "ten seconds",
        "thirty seconds",
        "ninety seconds",
        "four minutes",
        "fifteen minutes",
        "hourly ceiling",
        "daily ceiling",
    ),
    "storage backend": (
        "redis cluster",
        "memcached shard",
        "local lru",
        "sqlite file",
        "postgres table",
        "process dict",
        "disk mmap",
        "bypass layer",
    ),
    "identity scheme": (
        "jwt bearer",
        "hmac signature",
        "mutual tls",
        "apikey header",
        "hardware code",
        "browser cookie",
        "basic credentials",
        "signed url",
    ),
    "error handler": (
        "exponential backoff",
        "parking lane",
        "circuit breaker",
        "fail fast",
        "silent drop",
        "alert operator",
        "fallback default",
        "escalate upstream",
    ),
    "batch size": (
        "tiny chunk",
        "small chunk",
        "medium chunk",
        "large chunk",
        "huge chunk",
        "dynamic chunk",
        "partition wide",
        "stream unbounded",
    ),
    "log sink": (
        "stdout json",
        "rotating archive",
        "syslog daemon",
        "kafka topic",
        "otel collector",
        "cloud bucket",
        "ring buffer",
        "null device",
    ),
    "lock strategy": (
        "optimistic version",
        "pessimistic row",
        "advisory key",
        "lease renewal",
        "open access",
        "single writer",
        "fair ordering",
        "spin wait",
    ),
}

# Distractors are concentrated rather than spread: a fact either has this many
# near-copies or none at all. Spreading them evenly would make every fact
# equally crowded and collapse DIRECT and DISTRACTOR into the same task.
_COPIES_PER_CROWDED_FACT = 4


@dataclass(frozen=True, slots=True)
class World:
    """A generated corpus plus the lookups the task builder needs.

    ``stable_by_subject`` lists the facts drift never touched, per subject; tasks
    that are not about drift are drawn from there so each family fails for one
    reason. ``crowded_ids`` are the facts that spawned distractors.
    """

    facts: dict[str, Fact]
    experiences: tuple[Experience, ...]
    stable_by_subject: dict[str, tuple[str, ...]]
    crowded_ids: frozenset[str]
    drift_pairs: tuple[tuple[str, str], ...]
    distractor_ids: frozenset[str]


def build_world(config: WorldConfig, seed: int) -> World:
    """Generate the corpus and the experience stream that delivers it."""
    rng = random.Random(seed)
    predicates = tuple(_PREDICATES)
    subjects = tuple(_subject_name(i) for i in range(config.subjects))
    used: dict[str, set[tuple[str, str]]] = {subject: set() for subject in subjects}

    catalogue: list[Fact] = []
    predicate_of: dict[str, str] = {}

    def add(subject: str, predicate: str, supersedes: str | None = None) -> Fact:
        value = _fresh_value(rng, used[subject], predicate)
        text = f"{subject} {predicate} {value}"
        fact = Fact(
            id=f"fact-{len(catalogue):04d}",
            subject=subject,
            text=text,
            tokens=tuple(text.split()),
            supersedes=supersedes,
        )
        catalogue.append(fact)
        predicate_of[fact.id] = predicate
        return fact

    base_by_subject = {
        subject: tuple(
            add(subject, predicate)
            for predicate in _predicate_picks(rng, predicates, config.facts_per_subject)
        )
        for subject in subjects
    }
    base = [fact for facts in base_by_subject.values() for fact in facts]

    # Drift: a replacement with the same subject and predicate but one changed
    # value. Nearly identical wording is the point - it is what stops a
    # similarity arm from telling the current fact from the stale one.
    drift_pairs = tuple(
        (old.id, add(old.subject, predicate_of[old.id], supersedes=old.id).id)
        for old in rng.sample(base, round(config.drift_rate * len(base)))
    )

    budget = round(config.distractor_ratio * len(base))
    crowd_order = list(base)
    rng.shuffle(crowd_order)
    crowd_size = (budget + _COPIES_PER_CROWDED_FACT - 1) // _COPIES_PER_CROWDED_FACT
    crowded = crowd_order[:crowd_size]
    distractor_ids: list[str] = []
    for index in range(budget):
        # Round-robin, so the budget lands evenly and no crowded fact is left
        # with a single shadow while another carries the whole allocation.
        source = crowded[index % crowd_size]
        distractor_ids.append(add(source.subject, predicate_of[source.id]).id)

    ordered = _in_arrival_order(rng, catalogue, drift_pairs)
    facts: dict[str, Fact] = {}
    experiences: list[Experience] = []
    for timestep, draft in enumerate(ordered, start=1):
        fact = replace(draft, timestep=timestep)
        facts[fact.id] = fact
        experiences.append(
            Experience(
                id=f"exp-{timestep:04d}",
                fact_id=fact.id,
                text=fact.text,
                tokens=fact.tokens,
                subject=fact.subject,
                timestep=timestep,
            )
        )

    superseded = {old for old, _ in drift_pairs}
    return World(
        facts=facts,
        experiences=tuple(experiences),
        stable_by_subject={
            subject: tuple(fact.id for fact in group if fact.id not in superseded)
            for subject, group in base_by_subject.items()
        },
        crowded_ids=frozenset(fact.id for fact in crowded),
        drift_pairs=drift_pairs,
        distractor_ids=frozenset(distractor_ids),
    )


def _in_arrival_order(
    rng: random.Random,
    catalogue: list[Fact],
    drift_pairs: tuple[tuple[str, str], ...],
) -> list[Fact]:
    """Interleave every fact into one stream, each replacement after its predecessor.

    Replacements are scattered through the stream rather than appended in a
    second era, because an era boundary would hand any arm a free rule - "later
    half wins" - that has nothing to do with retrieval quality.
    """
    keys = {fact.id: rng.random() for fact in catalogue}
    for old_id, new_id in drift_pairs:
        earlier = keys[old_id]
        keys[new_id] = earlier + (1.0 - earlier) * rng.random()
    # Ties break on id, and a replacement is always minted after its predecessor,
    # so the strictly-later invariant survives even a float collision.
    return sorted(catalogue, key=lambda fact: (keys[fact.id], fact.id))


def _predicate_picks(rng: random.Random, predicates: tuple[str, ...], count: int) -> list[str]:
    """Distinct predicates per subject while supply lasts."""
    if count <= len(predicates):
        return rng.sample(predicates, count)
    return [rng.choice(predicates) for _ in range(count)]


def _fresh_value(rng: random.Random, seen: set[tuple[str, str]], predicate: str) -> str:
    """Pick a value unused with this predicate in this subject.

    No two facts anywhere may carry identical text: a distractor that reads
    exactly like the fact it shadows would make its task unanswerable by any
    means, which is a broken instance rather than a hard one.
    """
    value = rng.choice([v for v in _PREDICATES[predicate] if (predicate, v) not in seen])
    seen.add((predicate, value))
    return value


def _subject_name(index: int) -> str:
    """Suffix the vocabulary when more subjects are asked for than words exist."""
    word = _SUBJECTS[index % len(_SUBJECTS)]
    lap = index // len(_SUBJECTS)
    return word if lap == 0 else f"{word}{lap + 1}"
