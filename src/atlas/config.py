"""Run configuration.

Every knob that changes a published number lives here, so a result can be
reproduced from the config block the runner prints alongside it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["ExperimentConfig", "KEEConfig", "Settings", "WorldConfig", "settings"]


class WorldConfig(BaseModel):
    """Shape of the generated world and task suite."""

    model_config = {"frozen": True}

    subjects: int = Field(default=24, ge=2, description="Distinct areas of the codebase.")
    facts_per_subject: int = Field(default=6, ge=2)
    drift_rate: float = Field(default=0.25, ge=0.0, le=1.0, description="Facts later superseded.")
    distractor_ratio: float = Field(default=2.0, ge=0.0, description="Distractors per real fact.")
    tasks: int = Field(default=200, ge=1)


class KEEConfig(BaseModel):
    """Knowledge Evolution Engine hyper-parameters."""

    model_config = {"frozen": True}

    decay: float = Field(default=0.65, gt=0.0, lt=1.0, description="Energy kept per hop.")
    max_hops: int = Field(default=3, ge=1, le=8)
    activation_floor: float = Field(default=0.02, gt=0.0, description="Stop spreading below this.")
    seed_width: int = Field(default=6, ge=1, description="Nodes energised from the query.")
    edge_threshold: float = Field(default=0.15, ge=0.0, description="Minimum edge weight kept.")
    merge_threshold: float = Field(
        default=0.85,
        gt=0.0,
        le=1.0,
        description="Overlap at which an experience restates a concept.",
    )
    revision_threshold: float = Field(
        default=0.45, gt=0.0, le=1.0, description="Overlap at which it instead revises one."
    )
    utility_threshold: float = Field(default=0.25, ge=0.0, description="Prune below this utility.")
    alpha_confidence: float = 0.4
    beta_recency: float = 0.2
    gamma_frequency: float = 0.3
    delta_contradiction: float = 0.1
    recency_halflife: int = Field(default=40, ge=1, description="Timesteps to half recency.")


class ExperimentConfig(BaseModel):
    """One comparison run."""

    model_config = {"frozen": True}

    seed: int = 42
    top_k: int = Field(default=5, ge=1)
    embedding_dim: int = Field(default=256, ge=16)
    world: WorldConfig = Field(default_factory=WorldConfig)
    kee: KEEConfig = Field(default_factory=KEEConfig)


class Settings(BaseSettings):
    """Process-level settings that never change a number, only where it goes."""

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="ATLAS_", env_file_encoding="utf-8", extra="ignore"
    )

    log_level: str = "INFO"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    wandb_project: str = "atlas-experiments"
    wandb_enabled: bool = False


settings = Settings()
