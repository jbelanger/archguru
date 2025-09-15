"""
Configuration management for ArchGuru
Handles OpenRouter API keys and model settings
"""
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration settings for the ArchGuru platform"""

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    # Default competing models for simple-first approach
    DEFAULT_MODELS = [
        "openai/gpt-4o-mini",
        "anthropic/claude-3-haiku"
    ]

    @classmethod
    def get_models(cls) -> List[str]:
        """Get competing models from ARCHGURU_MODELS environment variable or defaults"""
        models_env = os.getenv("ARCHGURU_MODELS", "")
        if models_env:
            return [m.strip() for m in models_env.split(",") if m.strip()]
        return cls.DEFAULT_MODELS

    @classmethod
    def get_arbiter_model(cls) -> str:
        """Get the final arbiter model from environment variables or default"""
        return os.getenv("ARCHGURU_ARBITER_MODEL", "openai/gpt-4o")

    # Decision types supported
    DECISION_TYPES = [
        "project-structure",
        "database",
        "deployment",
        "api-design"
    ]

    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present"""
        if not cls.OPENROUTER_API_KEY:
            print("❌ Error: OPENROUTER_API_KEY not found in environment")
            print("   Please create a .env file with your OpenRouter API key")
            return False
        return True