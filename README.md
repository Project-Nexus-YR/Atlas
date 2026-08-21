# Atlas

Measures whether an evolving knowledge graph retrieves better context than vector
similarity search, on a synthetic corpus where facts go stale over time.

The answer, on the benchmark in this repo, is **mostly no**. That result is the
point of the repository: the engine is instrumented so the ways it loses are
legible, and the harness is built so neither arm can win by construction.

## Result

```
$ atlas --seed 42

Atlas 0.2.0  ·  KEE vs Vector-RAG  ·  seed 42  ·  186 tasks  ·  top_k 5

family              n       KEE   Vector-RAG    delta
-----------------------------------------------------
direct             50     56.0%       100.0%    -44.0
distractor         50     78.0%       100.0%    -22.0
compositional      50      0.0%         0.0%     +0.0
drift              36     22.2%         5.6%    +16.7
-----------------------------------------------------
overall           186     40.3%        54.8%    -14.5

KEE         concepts 115  ·  revisions 68  ·  merges 0  ·  novel retrievals 149
Vector-RAG  vectors 468  ·  dim 256  ·  novel retrievals 0
```

The graph arm loses overall. It wins one family — `drift`, by 16.7 points, which
is the family it was built for — and is beaten badly on the two families that
plain lexical matching already solves.

Reproduce with `atlas --seed 42`, or `python -m atlas.cli --seed 42`. The run is
deterministic: same seed, same table, byte for byte. `stdout` carries only the
table, so it can be diffed; all progress logging goes to `stderr`.

## What is being compared

Both arms are handed the identical stream of `Experience` objects in the
identical order, and are asked the identical questions through the same
`MemorySystem` protocol. The only difference is what happens in between.

**Vector-RAG** (baseline) embeds each experience once with a hashing encoder and
retrieves by cosine similarity — SMART `lnc.ltc`, with query-side smoothed IDF
over a frozen document matrix. Its `consolidate` is a real, documented no-op.
That contrast *is* the experiment: a vector index never revisits what it stored.

**KEE** (the arm under test) stores concepts in a graph and does two things the
baseline does not:

- **Belief revision.** An experience that closely resembles a known concept
  without matching it is read as *news about that concept*, and the old concept
  is marked contradicted. Confidence is a Laplace-smoothed beta mean over
  evidence weights, so contradiction genuinely drives it down.
- **Forgetting.** Concepts whose utility falls below a threshold are removed
  outright, where utility is
  `0.4·confidence + 0.2·recency + 0.3·frequency − 0.1·contradiction`.

Retrieval is spreading activation: the query energises lexically-matching seed
nodes and that energy flows along weighted edges, attenuating per hop.

### Neither arm can see the answer

A memory arm never receives a `Fact`, only an `Experience`. `Fact.supersedes` —
which says that one fact invalidates another — is therefore unreadable by both
arms, and `Experience` carries no such field. A test pins that.

So when the KEE decides a claim is stale, it is deciding from two experiences
that say almost, but not quite, the same thing about the same subject. That
heuristic is genuinely fallible: a distractor built from the same subject
vocabulary lands in the same overlap band as a real revision, and the engine
discredits a good concept on the strength of noise. Those losses are real and
they are in the table above.

### Neither arm can rig the grade

A task is resolved iff `required_fact_ids ⊆ retrieved` **and**
`forbidden_fact_ids ∩ retrieved == ∅`. The two clauses pull against each other,
so no arm can win by returning everything within its budget.

The grader reads only the task and the `RetrievalResult`. It has no parameter
through which an agent-authored claim of success could arrive — the arm cannot
grade itself, because it is never asked.

## The families, and why two of them are 100%

| family | what it asks | why it is here |
|---|---|---|
| `direct` | one fact, query is its own wording | floor: any retriever should pass |
| `distractor` | one fact, with same-subject decoys present | tests precision under noise |
| `compositional` | two facts about one subject, token-disjoint | needs an edge, not similarity |
| `drift` | a fact whose replacement arrived later | tests whether stale beliefs die |

`direct` and `distractor` are solved outright by the baseline. That is reported
rather than hidden, because an overall number that concealed it would be
flattering instead of informative: the headline is really carried by two
families, and their weight in it is set by mixing ratios this repo chose.

## The compositional result

Both arms score **0.0%**, and for the KEE the reason is now measured rather than
guessed at. Two independent causes:

1. **Availability.** Forgetting removes one of the two required concepts in
   30 of 50 tasks before the question is asked. Winning `drift` costs
   `compositional` — that trade is the mechanism working as designed, not a bug.
2. **Ranking.** In the 20 tasks where both concepts survive, the connecting edge
   is present in **20 of 20**, energy does flow across it, and the partner
   concept still lands at median rank 23 against a `top_k` of 5. Best observed
   rank was 5 — one slot short.

Disabling forgetting entirely fixes cause 1 completely (50/50 survive, 50/50
edged) and makes cause 2 *worse*: median partner rank falls from 23 to 54, and
the family stays at 0.0%.

Closing that gap would mean raising `top_k`, which is a harness parameter and
would be tuning the benchmark rather than the method, or degree-normalising
activation, which changes what the algorithm is. Both were declined. The number
above is what the shipped algorithm scores.

One honest note on the mechanism: `activation.py` claims that two weak seeds can
jointly push a shared neighbour above what either lights alone. This family never
engages that. Its query is exactly one fact's wording, so there is one strong
seed and one path — convergence never happens.

### A bug this exposed

Edge formation drew link candidates only from the lexical seed index, while the
edge weight function granted a bonus to concepts sharing a subject. A pair
sharing a subject but few tokens can never appear in a Jaccard-ranked shortlist,
so that bonus was unreachable and no subject edge was ever written. Every one of
the 50 compositional pairs sat at Jaccard 0.1111 against a 0.15 threshold.

The graph was, in effect, a redundant copy of its own seed index: spreading
activation could only reach nodes lexical scoring had already ranked highly. That
is the one thing this arm exists to do better than a vector index.

Fixing it moved `direct` 50.0 → 56.0, `distractor` 66.0 → 78.0, and overall
36.0 → 40.3. It did not move `compositional`, for the reasons above. Two
regression tests pin it.

## Limitations

Stated because they bear on how far the number generalises.

- **Activation is summed without degree normalisation.** A node with many edges
  accumulates more energy simply for being well-connected. This is a known
  property of the classical algorithm and was kept deliberately; normalising it
  would change what is being measured.
- **Utility is clamped to `[0,1]`,** which makes the four weights not
  scale-invariant: doubling all of them is not a no-op. Weight ablation is not a
  deliverable here.
- **The corpus is synthetic** — five-token templates, a 152-token vocabulary, no
  polysemy, no negation, no noise. Drift is a single replacement changing a
  single token.
- **`compositional` encodes the graph's own schema into the task definition** by
  giving the two facts token-disjoint vocabularies. It is a test built to need an
  edge.
- **`top_k` is a harness parameter** and is the sole source of difficulty in
  three of the four families.
- **There is no agent.** Removing the agent's ability to grade itself also
  removed the agent. Success here is set membership over fact ids: a proxy for
  grounded action, not action. No code is written and no patch is executed.

The bar this was built against is [`princeton-nlp/SWE-agent`](https://github.com/SWE-agent/SWE-agent),
which publishes resolve rates nowhere near 100%. A retrieval benchmark reporting
a perfect score would be reporting its own harness.

### Declared but unused

Noted rather than removed, since they predate this work:

- `Settings.neo4j_uri`, `neo4j_user`, `neo4j_password`, `wandb_project`,
  `wandb_enabled` in `src/atlas/config.py` are read by nothing. Only
  `settings.log_level` is consumed.
- The `graph`, `tracking`, and `swebench` optional-dependency extras are
  consumed by no source file.
- `docker/docker-compose.yml` starts PostgreSQL, Qdrant, and Neo4j. The code in
  this repo is in-memory and connects to none of them.

## Layout

```
src/atlas/
├── cli.py                  # the one command; stdout is the table, stderr is narration
├── config.py               # every knob that changes a published number
├── types.py                # Fact, Experience, Concept, Task, MemorySystem protocol
├── embedding.py            # hashing encoder, SMART lnc.ltc
├── logging.py
├── bench/
│   ├── world.py            # generates facts, drift, distractors, the experience stream
│   ├── tasks.py            # builds the four task families
│   └── grader.py           # required ⊆ retrieved ∧ forbidden ∩ retrieved = ∅
├── memory/
│   ├── store.py            # concept graph: structure only
│   ├── scoring.py          # utility, confidence, edge weight, pruning
│   ├── consolidation.py    # offline pass: link, then forget
│   ├── activation.py       # spreading activation
│   ├── kee.py              # the arm under test
│   └── vector_rag.py       # the baseline
└── experiment/
    ├── runner.py           # the evaluation protocol: what each arm sees, and when
    └── compare.py          # drives both arms, renders the table
```

### The protocol

One pass along the timeline. For each task: deliver every experience whose
timestep has arrived, consolidate **once per distinct timestep** (not once per
task, so an arm is not punished for how many questions share a moment), then ask.

Ordering is load-bearing. Ingesting the whole corpus up front and only then
asking would erase the `drift` family entirely, because a superseded fact and its
replacement would arrive together and nothing would need revising.

## Development

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

Four gates, all of which must be green:

```bash
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m ruff format --check .
.venv/Scripts/python.exe -m mypy .
.venv/Scripts/python.exe -m pytest
```

`[tool.mypy] strict = true` is set in `pyproject.toml`, so plain `mypy` is the
strict gate.

Requires Python 3.13.

## License

MIT — see [LICENSE](LICENSE).
