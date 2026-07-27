from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Global configuration for the Atlas platform.
    Loads from environment variables or a .env file.
    """
    # Project Info
    project_name: str = "Atlas"
    
    # LLM Settings
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "zhipu/glm-4"  # Default to GLM model on OpenRouter
    temperature: float = 0.0

    # Experiment Settings
    wandb_project: str = "atlas-experiments"
    wandb_api_key: str = ""
    seed: int = 42

    # Database URLs
    postgres_url: str = "postgresql://user:password@localhost:5432/atlas"
    qdrant_url: str = "http://localhost:6333"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
