import os
import yaml
from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
YAML_PATH = BASE_DIR / "models" / "llm_config.yaml"

def load_yaml_config():
    if YAML_PATH.exists():
        with open(YAML_PATH, "r") as f:
            try:
                return yaml.safe_load(f) or {}
            except Exception:
                return {}
    return {}

yaml_cfg = load_yaml_config()
rag_cfg = yaml_cfg.get("rag", {})

class Settings(BaseSettings):
    # App Settings
    app_name: str = "Football RAG SaaS API"
    debug: bool = False
    
    # Paths
    llm_config_path: str = str(YAML_PATH)
    
    # DB DSN (defaults from yaml config if available)
    postgres_dsn: str = Field(
        default=rag_cfg.get("postgres_dsn", "postgresql://postgres:123@localhost:5432/football_rag"),
        alias="postgres_dsn"
    )
    
    # KG Provider (defaults from yaml config if available)
    kg_provider: str = Field(
        default=rag_cfg.get("kg_provider", "postgres"),
        alias="kg_provider"
    )
    
    # JWT Settings
    jwt_secret_key: str = "super-secret-football-rag-saas-key-change-in-production-2026"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440  # 24 hours
    
    # CORS
    cors_origins: Union[str, List[str]] = ["*"]
    
    # Default Supervisor
    football_admin_username: str = "admin"
    football_admin_email: str = "admin@football.com"
    football_admin_password: str = "AdminPass123!"

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

settings = Settings()
