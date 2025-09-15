from enum import Enum
from typing import Final

# Model competition
DEFAULT_K_FACTOR: Final[float] = 32.0
DEFAULT_ELO_RATING: Final[float] = 1200.0
MAX_TOOL_RESULTS: Final[int] = 10
DEFAULT_TOOL_RESULTS: Final[int] = 5

# Response parsing
MAX_RECOMMENDATION_LENGTH: Final[int] = 240
MAX_REASONING_LENGTH: Final[int] = 900
MAX_PREVIEW_LINES: Final[int] = 12

# Timeouts
API_TIMEOUT: Final[int] = 8
MODEL_MAX_TOKENS: Final[int] = 2000
MODEL_TEMPERATURE: Final[float] = 0.7
ARBITER_MAX_TOKENS: Final[int] = 1500
ARBITER_TEMPERATURE: Final[float] = 0.3

class DecisionType(str, Enum):
    PROJECT_STRUCTURE = "project-structure"
    DATABASE = "database"
    DEPLOYMENT = "deployment"
    API_DESIGN = "api-design"
    
class TeamType(str, Enum):
    COMPETITOR = "competitor"
    BASIC = "basic"

class WinnerSource(str, Enum):
    ARBITER = "arbiter"
    FALLBACK = "fallback"
    SINGLE_VALID = "single_valid"
    NO_VALID = "no_valid_responses"