from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Memory(ABC):
    """
    Abstract base class defining the standard interface for all memory
    implementations (e.g., Vector, Graph, Episodic, Semantic).
    """

    @abstractmethod
    def store(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Stores a memory."""
        pass

    @abstractmethod
    def retrieve(self, query: Any, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Retrieves memories relevant to the query."""
        pass

    @abstractmethod
    def update(self, key: str, value: Any) -> None:
        """Updates an existing memory."""
        pass

    @abstractmethod
    def summarize(self) -> str:
        """Generates a compressed summary of the current memory state."""
        pass

    @abstractmethod
    def forget(self, key: str) -> bool:
        """Removes a memory. Returns True if successful."""
        pass

    @abstractmethod
    def importance(self, key: str) -> float:
        """Calculates or retrieves the importance score of a given memory."""
        pass

    @abstractmethod
    def statistics(self) -> Dict[str, Any]:
        """Returns statistics about the memory (e.g., size, token count, capacity)."""
        pass
