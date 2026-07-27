from dataclasses import dataclass
from typing import List

from atlas.memory.episodic import EpisodicMemory, Experience
from atlas.memory.graph import Concept, GraphMemory


@dataclass
class Reflection:
    """An insight extracted from a single or small group of raw experiences."""
    content: str
    source_experiences: List[Experience]


@dataclass
class Pattern:
    """A recurring theme or rule identified across multiple reflections."""
    theme: str
    source_reflections: List[Reflection]


class AdaptiveMemoryManager:
    """
    Orchestrates the lifelong learning pipeline:
    Experience -> Reflection -> Pattern Discovery -> Concept Creation -> Compression -> Graph Update.
    """

    def __init__(self, episodic_store: EpisodicMemory, graph_store: GraphMemory) -> None:
        self.episodic_store = episodic_store
        self.graph_store = graph_store

    def run_consolidation_pipeline(self) -> None:
        """
        Executes the full memory distillation pipeline.
        This would typically be called asynchronously or during agent "sleep" cycles.
        """
        experiences = self.episodic_store.retrieve_all()
        if not experiences:
            return

        print(f"[AdaptiveMemory] Starting consolidation pipeline for {len(experiences)} experiences.")

        # 1. Reflection
        reflections = self._reflect(experiences)

        # 2. Pattern Discovery
        patterns = self._discover_patterns(reflections)

        # 3. Concept Creation
        concepts = self._create_concepts(patterns)

        # 4. Knowledge Graph Update
        self._update_knowledge_graph(concepts)

        # 5. Memory Compression
        # If successfully distilled into concepts, we can safely delete the raw experiences
        # to prevent token/context bloat.
        self._compress_memory(experiences)
        
        print("[AdaptiveMemory] Consolidation complete.")

    def _reflect(self, experiences: List[Experience]) -> List[Reflection]:
        """Analyzes recent raw experiences and extracts actionable insights."""
        # Stub: In a real system, an LLM call would analyze the experiences.
        return [Reflection(content="Stubbed reflection based on recent events.", source_experiences=experiences)]

    def _discover_patterns(self, reflections: List[Reflection]) -> List[Pattern]:
        """Groups related reflections to identify recurring issues or successful strategies."""
        # Stub: Can be clustering or LLM-driven grouping.
        return [Pattern(theme="Stubbed pattern across reflections.", source_reflections=reflections)]

    def _create_concepts(self, patterns: List[Pattern]) -> List[Concept]:
        """Distills patterns into abstract, generalized knowledge rules."""
        concepts = []
        for p in patterns:
            concepts.append(Concept(
                name="DiscoveredConcept",
                description=p.theme,
                concept_type="GeneralizedRule"
            ))
        return concepts

    def _update_knowledge_graph(self, concepts: List[Concept]) -> None:
        """Pushes the new concepts to the Neo4j backend."""
        for concept in concepts:
            self.graph_store.store_concept(concept)

    def _compress_memory(self, experiences: List[Experience]) -> None:
        """Archives or deletes raw experiences from EpisodicMemory that have been successfully distilled."""
        self.episodic_store.remove_experiences(experiences)
        print(f"[AdaptiveMemory] Compressed/removed {len(experiences)} raw experiences.")
