from abc import ABC, abstractmethod
from typing import Any, Dict

from atlas.memory.evolution import KnowledgeEvolutionEngine


class Agent(ABC):
    """
    Abstract base class for autonomous agents.
    Every agent must conform to this interface and interact with the KEE.
    """

    def __init__(self, kee: KnowledgeEvolutionEngine) -> None:
        self.kee = kee

    @abstractmethod
    def plan(self, objective: str) -> Any:
        """Formulate a plan based on the objective using KEE for spreading activation retrieval."""
        pass

    @abstractmethod
    def act(self, plan: Any) -> Any:
        """Execute the next step in the plan."""
        pass

    @abstractmethod
    def reflect(self, outcome: Any) -> Any:
        """Analyze the outcome of an action, generating an experience for the KEE."""
        pass

    @abstractmethod
    def learn(self, reflection: Any) -> None:
        """
        Assimilate the reflection/experience into the Knowledge Evolution Engine.
        This triggers the frequent cognitive loop.
        """
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """Return metrics evaluating the agent's current performance/state."""
        pass
