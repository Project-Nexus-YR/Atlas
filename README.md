# Atlas

Measures whether an evolving knowledge graph retrieves better context than vector
similarity search, on a synthetic corpus where facts go stale over time.

The graph arm loses the headline by 14.5 points. The more useful finding is what
that headline is made of: a BM25 control saturates two of the four families at
100%, and a third is 0.0% for every arm, so three quarters of the benchmark
distinguishes nothing. On the one family that does discriminate, the graph arm is
the only arm that scores at all.

That is the point of the repository: the engine is instrumented so the ways it
loses are legible, and the harness is built so no arm can win by construction.

## Result

```
$ atlas --seed 42

Atlas 0.2.0  ·  KEE vs Vector-RAG vs BM25  ·  seed 42  ·  186 tasks  ·  top_k 5

family              n          KEE   Vector-RAG         BM25   vs best
----------------------------------------------------------------------
direct             50        56.0%       100.0%       100.0%     -44.0
distractor         50        78.0%       100.0%       100.0%     -22.0
compositional      50         0.0%         0.0%         0.0%      +0.0
drift              36        22.2%         5.6%         0.0%     +16.7
----------------------------------------------------------------------
overall           186        40.3%        54.8%        53.8%     -14.5

KEE         concepts 115  ·  revisions 68  ·  merges 0  ·  novel retrievals 149
Vector-RAG  concepts 468  ·  vectors 468  ·  dim 256  ·  novel retrievals 0
BM25        concepts 468  ·  documents 468  ·  vocabulary 152  ·  novel retrievals 0
```

`vs best` is the graph arm's margin over the **strongest** control on that row,
never over a nominated one, so adding a control can only make the reported margin
worse. Read down the columns rather than across the bottom row:

- `direct` and `distractor` are **saturated**. BM25 — term overlap and nothing
  else, no embeddings, no graph — scores 100% on both. Whatever those families
  measure, it is not retrieval quality above the lexical floor, and the vector
  arm's perfect score there is not evidence that embeddings bought anything.
- `compositional` is **0.0% for all three arms**. A family no arm scores on
  separates no arms.
- `drift` is the only family where the three arms come apart, and BM25's **0.0%**
  is the informative cell: drift is not reachable by term overlap at all, so this
  family asks for something the other two never did. Consolidation is the only
  mechanism present that supplies any of it.

That last row is 8 tasks against 2 against 0, out of 36 — a clear ordering on a
small sample, not a confidence interval. And the headline is carried by two
families whose ceiling a lexical baseline reaches, weighted by mixing ratios this
repo chose. The graph arm's deficit on them is real — it loses to plain term
matching on tasks plain term matching solves — but it is a deficit on the half of
the benchmark that turned out not to be asking anything.

Reproduce with `atlas --seed 42`, or `python -m atlas.cli --seed 42`. The run is
deterministic: same seed, same table, byte for byte. `stdout` carries only the
table, so it can be diffed; all progress logging goes to `stderr`.

## What is being compared

All three arms are handed the identical stream of `Experience` objects in the
identical order, and are asked the identical questions through the same
`MemorySystem` protocol. The only difference is what happens in between.

**BM25** (lexical control) is Okapi BM25 over an inverted index: term frequency
saturating at `k1`, length normalised by `b`, RSJ idf. No embeddings, no graph,
and a `consolidate` that is a no-op. It is here to price the corpus — a family it
scores 100% on is a family that needed nothing but term overlap. It returns only
documents sharing at least one query term, so it can return fewer than `top_k`
where the other two always fill the budget; under a grader that punishes
forbidden facts as well as rewarding required ones, that is a trade rather than a
handicap.

**Vector-RAG** (dense control) embeds each experience once with a hashing encoder
and retrieves by cosine similarity — SMART `lnc.ltc`, with query-side smoothed IDF
over a frozen document matrix. Its `consolidate` is a real, documented no-op.
That contrast *is* the experiment: a vector index never revisits what it stored.

**KEE** (the arm under test) stores concepts in a graph and does two things no
control does:

- **Belief revision.** An experience that closely resembles a known concept
  without matching it is read as *news about that concept*, and the old concept
  is marked contradicted. Confidence is a Laplace-smoothed beta mean over
  evidence weights, so contradiction genuinely drives it down.
- **Forgetting.** Concepts whose utility falls below a threshold are removed
  outright, where utility is
  `0.4·confidence + 0.2·recency + 0.3·frequency − 0.1·contradiction`.

Retrieval is spreading activation: the query energises lexically-matching seed
nodes and that energy flows along weighted edges, attenuating per hop.

### No arm can see the answer

A memory arm never receives a `Fact`, only an `Experience`. `Fact.supersedes` —
which says that one fact invalidates another — is therefore unreadable by every
arm, and `Experience` carries no such field. A test pins that.

So when the KEE decides a claim is stale, it is deciding from two experiences
that say almost, but not quite, the same thing about the same subject. That
heuristic is genuinely fallible: a distractor built from the same subject
vocabulary lands in the same overlap band as a real revision, and the engine
discredits a good concept on the strength of noise. Those losses are real and
they are in the table above.

### No arm can rig the grade

A task is resolved iff `required_fact_ids ⊆ retrieved` **and**
`forbidden_fact_ids ∩ retrieved == ∅`. The two clauses pull against each other,
so no arm can win by returning everything within its budget.

The grader reads only the task and the `RetrievalResult`. It has no parameter
through which an agent-authored claim of success could arrive — the arm cannot
grade itself, because it is never asked.

## The families, and what each one measures

| family | what it asks | why it is here |
|---|---|---|
| `direct` | one fact, query is its own wording | floor: any retriever should pass |
| `distractor` | one fact, with same-subject decoys present | tests precision under noise |
| `compositional` | two facts about one subject, token-disjoint | needs an edge, not similarity |
| `drift` | a fact whose replacement arrived later | tests whether stale beliefs die |

The lexical control is what turned the right-hand column of that table from an
intention into a measurement. `direct` and `distractor` were *designed* as a floor
and a precision test; what the control shows is that both are floors. Term overlap
alone clears them, so `distractor` does not in fact ask for precision the lexical
baseline lacks.

That is reported rather than hidden, because an overall number concealing it would
be flattering instead of informative. It also bounds what this benchmark can
currently establish. With one saturated pair, one family at zero for everyone, and
one family of 36 tasks doing all of the discriminating, the defensible claim is
about `drift` — not about the overall row.

## The compositional result

All three arms score **0.0%**, and for the KEE the reason is now measured rather
than guessed at. Two independent causes:

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
  accumulates more energy simply for being well-connected. That matches Collins &
  Loftus (1975), where spread is not discounted by degree — but not ACT-R, where
  associative strength carries a `−ln(fan)` term and source activation is split
  `W/n` across the elements of the query. So this is one of two classical choices,
  not the classical behaviour, and it is the choice that leaves hub nodes
  advantaged. Changing it is a live option rather than a closed one.
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
- **Three of the four families do not discriminate.** Two are saturated by the
  lexical control and one is zero for every arm, so the overall row is a weighted
  average over tasks that mostly separate nothing.
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
│   ├── vector_rag.py       # the dense control
│   └── bm25.py             # the lexical control
└── experiment/
    ├── runner.py           # the evaluation protocol: what each arm sees, and when
    └── compare.py          # drives every arm, renders the table
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
