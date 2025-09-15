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

    # Model teams for competition
    MODEL_TEAMS = {
        "openai": [
            "openai/gpt-4o",
            "openai/gpt-4o-mini"
        ],
        "claude": [
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3-haiku"
        ],
        "llama": [
            "meta-llama/llama-3.2-3b-instruct",
            "meta-llama/llama-3.1-8b-instruct"
        ]
    }

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