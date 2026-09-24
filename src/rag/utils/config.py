"""Configuration loader with environment variable override support."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Settings:
    """Minimal settings from environment."""

    def __init__(self):
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.COHERE_API_KEY = os.getenv("COHERE_API_KEY")
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
        self.RAG_CONFIG_PATH = os.getenv("RAG_CONFIG_PATH", "configs/config.yaml")
        self.RAG_TENANT_ID = os.getenv("RAG_TENANT_ID", "default")
        self.RAG_ENV = os.getenv("RAG_ENV", "development")


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        example = path.parent / "config.example.yaml"
        if example.exists():
            path = example
        else:
            # Last fallback – empty config so the app can still start
            print(f"[WARN] Config not found: {path}. Using empty config.")
            return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class AppConfig:
    """Singleton-style config accessor."""

    _instance: Optional["AppConfig"] = None

    def __init__(self, config_path: Optional[str] = None):
        self.settings = Settings()
        path = config_path or self.settings.RAG_CONFIG_PATH
        self.raw = load_yaml_config(path)
        if self.settings.RAG_ENV:
            self.raw.setdefault("project", {})["environment"] = self.settings.RAG_ENV

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "AppConfig":
        """Preferred way to get the config instance."""
        if cls._instance is None or config_path is not None:
            cls._instance = cls(config_path)
        return cls._instance

    # Keep old name working
    @classmethod
    def get(cls, config_path: Optional[str] = None) -> "AppConfig":
        return cls.load(config_path)

    def lookup(self, *keys: str, default: Any = None) -> Any:
        """Get a nested config value: config.lookup('embeddings', 'provider')"""
        node = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    @property
    def project_name(self) -> str:
        return self.lookup("project", "name", default="multimodal-rag")