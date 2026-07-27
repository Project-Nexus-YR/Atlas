import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Experience:
    """
    Represents a single atomic action and its outcome in the environment.
    """
    action: str
    observation: str
    success: bool
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EpisodicMemory:
    """
    A short-to-medium term buffer for raw experiences.
    Experiences are stored here before being distilled into long-term concepts.
    """

    def __init__(self) -> None:
        self.buffer: List[Experience] = []

    def store(self, experience: Experience) -> None:
        """Stores a new experience in the buffer."""
        self.buffer.append(experience)

    def retrieve_all(self) -> List[Experience]:
        """Retrieves all pending experiences in the buffer."""
        return self.buffer

    def clear(self) -> None:
        """Clears the buffer after compression/distillation."""
        self.buffer.clear()

    def remove_experiences(self, experiences: List[Experience]) -> None:
        """Removes specific experiences that have been successfully compressed."""
        self.buffer = [exp for exp in self.buffer if exp not in experiences]

    def size(self) -> int:
        """Returns the number of experiences currently in the buffer."""
        return len(self.buffer)
