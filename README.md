# Atlas

A research platform for studying adaptive long-term memory in autonomous software engineering agents. Atlas implements a biologically-inspired Knowledge Evolution Engine (KEE) that enables agents to retain, consolidate, and evolve knowledge over time rather than operating statelessly.

## Overview

Atlas is designed for ML researchers investigating how autonomous coding agents can develop lifelong learning capabilities. The platform benchmarks agents on SWE-bench, comparing a baseline Vector-RAG approach against the Knowledge Evolution Engine, which uses spreading activation retrieval, episodic-to-semantic memory distillation, and utility-based forgetting -- inspired by cognitive science models of human memory consolidation.

## Features

- **Knowledge Evolution Engine** -- Living ecosystem of concepts with confidence, uncertainty, activation energy, and utility scoring
- **Spreading activation retrieval** -- Energy propagation through a semantic graph instead of vector similarity search
- **Episodic memory buffer** -- Short-to-medium term experience store before knowledge distillation
- **Knowledge consolidation pipeline** -- Reflection -> Pattern Discovery -> Concept Creation -> Graph Update -> Memory Compression
- **Cognitive timescales** -- Three-loop system (immediate, periodic, rare) modeled after biological memory consolidation
- **Utility-based forgetting** -- Concepts scored by `0.4*Confidence + 0.2*Recency + 0.3*Frequency - 0.1*Contradiction` with automatic pruning
- **SWE-bench integration** -- Evaluation harness for benchmarking against real-world GitHub issues
- **Neo4j knowledge graph** -- Backed by a graph database for persistent concept storage and relationship queries
- **Experiment tracking** -- Weights & Biases integration for logging metrics and comparing runs
- **Docker Compose infrastructure** -- One-command setup for PostgreSQL, Qdrant, and Neo4j

## Architecture

```
ExperimentRunner + SWEBenchHarness
         |
    Agent (abstract)
    SWEAgent (concrete)
         |
KnowledgeEvolutionEngine (KEE)
    +-- CognitiveTimescales
         |
    +-- EpisodicMemory (buffer)
    +-- GraphEcosystem (Neo4j)
         |
    AdaptiveMemoryManager
    (consolidation pipeline)
```

The agent follows a Plan-Act-Reflect-Learn loop. During planning, KEE's spreading activation retrieves relevant architectural context. After acting, the experience is reflected upon and assimilated back into the ecosystem. Periodically, the consolidation pipeline distills raw experiences into abstract concepts.

## Technologies Used

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Graph Database | Neo4j 5.20 (via neo4j Python driver) |
| Vector Database | Qdrant (declared, not yet consumed) |
| Relational Database | PostgreSQL 15 (via SQLAlchemy + psycopg2) |
| Experiment Tracking | Weights & Biases |
| LLM Integration | OpenRouter API (default: `zhipu/glm-4`) |
| Data Validation | Pydantic v2, pydantic-settings |
| Build System | Hatchling |
| Linting | Ruff (line-length 88) |
| Type Checking | mypy (strict mode) |
| Containerization | Docker Compose v3.8 |

## Folder Structure

```
ML_Project/
+-- pyproject.toml                  # Project manifest and tool configuration
+-- docker/
|   +-- docker-compose.yml          # PostgreSQL, Qdrant, Neo4j services
+-- atlas/
    +-- configs/
    |   +-- settings.py             # Centralized configuration (pydantic-settings)
    +-- memory/
    |   +-- base.py                 # Abstract Memory interface
    |   +-- episodic.py             # EpisodicMemory buffer (Experience dataclass)
    |   +-- graph.py                # GraphEcosystem (Neo4j), Concept, Evidence
    |   +-- evolution.py            # KnowledgeEvolutionEngine, CognitiveTimescales
    |   +-- adaptive.py             # AdaptiveMemoryManager (consolidation pipeline)
    +-- agents/
    |   +-- base.py                 # Abstract Agent interface
    |   +-- swe_agent.py            # SWEAgent (SWE-bench concrete agent)
    +-- experiments/
    |   +-- logging.py              # ExperimentLogger (W&B / stdout)
    |   +-- runner.py               # ExperimentRunner (orchestrator)
    +-- evaluation/
        +-- harness.py              # SWEBenchHarness (evaluation driver)
```

## Installation

### Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- An OpenRouter API key (or OpenAI key)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd atlas

# Install in editable mode
pip install -e ".[dev]"

# Start infrastructure services
cd docker
docker-compose up -d
cd ..

# Create a .env file in the project root
cat > .env << EOF
OPENROUTER_API_KEY=your_key_here
WANDB_API_KEY=your_key_here
EOF
```

## Running

### Lint and Type Check

```bash
ruff check atlas/
mypy atlas/
```

### Run Tests

```bash
pytest
```

### Run Experiments

The project is currently a library without a CLI entry point. To run experiments, instantiate the agent and harness in a Python script:

```python
from atlas.memory.graph import GraphEcosystem
from atlas.memory.evolution import KnowledgeEvolutionEngine
from atlas.agents.swe_agent import SWEAgent
from atlas.experiments.runner import ExperimentRunner
from atlas.evaluation.harness import SWEBenchHarness

ecosystem = GraphEcosystem("bolt://localhost:7687", "neo4j", "password")
kee = KnowledgeEvolutionEngine(ecosystem)
agent = SWEAgent(kee)
harness = SWEBenchHarness(use_kee=True)
runner = ExperimentRunner(agent, harness)
runner.run(max_steps=10)
```

### Docker Compose Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `postgres` | `postgres:15-alpine` | 5432 | Relational storage |
| `qdrant` | `qdrant/qdrant:latest` | 6333, 6334 | Vector similarity search |
| `neo4j` | `neo4j:5.20.0` | 7474, 7687 | Knowledge graph backend |

## Interesting Technical Details

- **Utility score formula**: `UtilityScore = 0.4*Confidence + 0.2*Recency + 0.3*Frequency - 0.1*Contradiction`, where recency decays linearly over 7 days and frequency is logistically normalized at 100 occurrences.
- **Cognitive timescales**: Three biological loops -- immediate (real-time), periodic (merge/split/promote concepts), and rare (prune dead nodes) -- modeled after sleep-based memory consolidation in mammals.
- **Spreading activation**: Rather than vector similarity search, the design injects energy at nodes related to the current plan and lets it propagate through weighted semantic edges, a technique from cognitive science and semantic networks.
- **Consolidation pipeline**: Raw experiences -> LLM reflections -> pattern clustering -> abstract concept creation -> graph update -> memory compression (removing distilled raw experiences to prevent token bloat).

## Challenges

- **Early-stage prototype**: Most core algorithms (spreading activation, reflection, pattern discovery) are stubbed with placeholder implementations
- **No entry point**: No `main.py` or CLI exists yet to run the system end-to-end
- **No tests**: Despite pytest being declared as a dependency, no test files exist
- **Memory ABC not subclassed**: The `Memory` abstract base class is defined but the concrete implementations (`EpisodicMemory`, `GraphEcosystem`) do not inherit from it
- **Unused dependencies**: FastAPI, Qdrant, and SQLAlchemy are declared but not consumed in any source file

## Future Improvements

- Implement the full spreading activation algorithm in `KnowledgeEvolutionEngine.retrieve_context()`
- Build the LLM-powered reflection and pattern discovery stages
- Add CLI entry point for running experiments
- Create integration tests for the consolidation pipeline
- Wire up Qdrant for vector-based baseline comparison
- Implement FastAPI endpoints for the REST API
- Add SWE-bench dataset loading and evaluation metrics

## License

MIT License -- see [LICENSE](LICENSE) for details.
