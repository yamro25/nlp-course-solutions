import os
import yaml
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class LLMConfig(BaseModel):
    provider: str = "openai"
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0

class EmbeddingConfig(BaseModel):
    provider: str = "openai"
    model_name: str = "text-embedding-3-large"

class VectorDBConfig(BaseModel):
    provider: str = "chroma"
    persist_directory: str = "./data/chroma_db"

class RetrievalConfig(BaseModel):
    top_k: int = 4
    dense_weight: float = 0.5
    sparse_weight: float = 0.5

class AppConfig(BaseSettings):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    @classmethod
    def load_from_yaml(cls, yaml_path: str = "config/settings.yaml") -> "AppConfig":
        path = Path(yaml_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        return cls()

app_config = AppConfig.load_from_yaml()