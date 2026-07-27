import os
from typing import Any, Dict, Optional

from atlas.configs.settings import settings

# Graceful degradation if wandb is not installed during early dev
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ExperimentLogger:
    """
    Standardized logging interface for Atlas experiments.
    Currently defaults to Weights & Biases (wandb) if available,
    otherwise falls back to stdout.
    """

    def __init__(self, project: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.project = project
        self.config = config or {}
        self.run = None

    def start(self) -> None:
        """Initializes the experiment logging run."""
        if WANDB_AVAILABLE and settings.wandb_api_key:
            self.run = wandb.init(
                project=self.project,
                config=self.config,
                reinit=True
            )
        else:
            print(f"[Logger] Started experiment run for project '{self.project}'")
            print(f"[Logger] Config: {self.config}")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Logs runtime metrics (e.g., token usage, success rate, memory size)."""
        if self.run:
            self.run.log(metrics, step=step)
        else:
            print(f"[Logger] Metrics: {metrics}")

    def finish(self) -> None:
        """Closes the experiment run."""
        if self.run:
            self.run.finish()
        else:
            print("[Logger] Finished experiment run.")
