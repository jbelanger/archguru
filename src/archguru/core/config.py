"""
Configuration management for ArchGuru
Handles OpenRouter API keys and model settings
"""
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration settings for the ArchGuru platform"""

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    # Model teams for competition (configurable via environment variables)
    DEFAULT_MODEL_TEAMS = {
        "team_a": [
            "openai/gpt-oss-120b",
            "openai/gpt-5-nano"
        ],
        "team_b": [
            "qwen/qwen3-next-80b-a3b-thinking",
            "qwen/qwen3-next-80b-a3b-instruct"
        ],
        "team_c": [
            "openrouter/sonoma-dusk-alpha",
            "openrouter/sonoma-sky-alpha"
        ]
    }

    @classmethod
    def get_model_teams(cls) -> Dict[str, List[str]]:
        """Get model teams from environment variables or defaults"""
        model_teams = {}

        # Load custom model teams from environment
        team_a_models = os.getenv("ARCHGURU_TEAM_A_MODELS", "").split(",")
        team_b_models = os.getenv("ARCHGURU_TEAM_B_MODELS", "").split(",")
        team_c_models = os.getenv("ARCHGURU_TEAM_C_MODELS", "").split(",")

        # Use environment variables if provided, otherwise use defaults
        model_teams["team_a"] = [m.strip() for m in team_a_models if m.strip()] or cls.DEFAULT_MODEL_TEAMS["team_a"]
        model_teams["team_b"] = [m.strip() for m in team_b_models if m.strip()] or cls.DEFAULT_MODEL_TEAMS["team_b"]
        model_teams["team_c"] = [m.strip() for m in team_c_models if m.strip()] or cls.DEFAULT_MODEL_TEAMS["team_c"]

        return model_teams

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