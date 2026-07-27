import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase


@dataclass
class Evidence:
    """Represents a trace of an experience supporting or contradicting a concept."""
    experience_id: str
    weight: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class Concept:
    """
    First-class knowledge object in the Knowledge Evolution Engine.
    Not a static memory, but a living node in the ecosystem.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    abstraction_level: int = 0  # 0 = concrete, higher = more abstract
    confidence: float = 0.5     # [0.0, 1.0]
    uncertainty: float = 0.5    # [0.0, 1.0]
    activation: float = 0.0     # Energy during retrieval
    utility_score: float = 0.5  # Future usefulness
    
    supporting_evidence: List[Evidence] = field(default_factory=list)
    contradictory_evidence: List[Evidence] = field(default_factory=list)
    
    frequency: int = 1
    recency: float = field(default_factory=time.time)
    creation_timestamp: float = field(default_factory=time.time)
    last_update_timestamp: float = field(default_factory=time.time)

    def update_utility(self) -> None:
        """
        UtilityScore = alpha * Confidence + beta * Recency + gamma * Frequency - delta * Contradiction
        """
        alpha, beta, gamma, delta = 0.4, 0.2, 0.3, 0.1
        
        # Normalize recency relative to current time (simplified)
        age = time.time() - self.recency
        recency_factor = max(0.0, 1.0 - (age / (86400 * 7))) # Decay over 7 days

        # Contradiction ratio
        total_evidence = len(self.supporting_evidence) + len(self.contradictory_evidence)
        contradiction_ratio = len(self.contradictory_evidence) / total_evidence if total_evidence > 0 else 0.0
        
        # Normalize frequency logistically
        freq_factor = min(1.0, self.frequency / 100.0)

        self.utility_score = (
            alpha * self.confidence + 
            beta * recency_factor + 
            gamma * freq_factor - 
            delta * contradiction_ratio
        )
        self.last_update_timestamp = time.time()


class GraphEcosystem:
    """
    The living knowledge graph (replaces GraphMemory).
    Maintains concepts and handles spreading activation and structural queries.
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def store_concept(self, concept: Concept) -> None:
        """Upserts a Concept node in Neo4j."""
        query = """
        MERGE (c:Concept {id: $id})
        SET c.name = $name,
            c.description = $description,
            c.abstraction_level = $abstraction_level,
            c.confidence = $confidence,
            c.uncertainty = $uncertainty,
            c.activation = $activation,
            c.utility_score = $utility_score,
            c.frequency = $frequency,
            c.recency = $recency,
            c.creation_timestamp = $creation_timestamp,
            c.last_update_timestamp = $last_update_timestamp
        """
        with self.driver.session() as session:
            session.run(query, **concept.__dict__)

    def store_relationship(self, concept_a_id: str, concept_b_id: str, weight: float, relation_type: str = "RELATES_TO") -> None:
        """Creates a weighted edge for spreading activation."""
        query = f"""
        MATCH (a:Concept {{id: $a_id}})
        MATCH (b:Concept {{id: $b_id}})
        MERGE (a)-[r:{relation_type}]->(b)
        SET r.weight = $weight
        """
        with self.driver.session() as session:
            session.run(query, a_id=concept_a_id, b_id=concept_b_id, weight=weight)
