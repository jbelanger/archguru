"""
Data models for architecture decisions and model responses
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class DecisionRequest:
    """Represents an architecture decision request"""
    decision_type: str
    language: Optional[str] = None
    framework: Optional[str] = None
    requirements: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ModelResponse:
    """Response from a single model with research tracking"""
    model_name: str
    team: str  # openai, claude, llama
    recommendation: str
    reasoning: str
    trade_offs: List[str]
    confidence_score: float
    response_time: float
    success: bool = True  # v0.4: Explicit success flag instead of checking "Error:" prefix
    research_steps: List[Dict[str, Any]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.research_steps is None:
            self.research_steps = []


@dataclass
class DecisionResult:
    """Final result with all model responses and competition outcome"""
    request: DecisionRequest
    model_responses: List[ModelResponse]
    winning_model: Optional[str] = None
    consensus_recommendation: Optional[str] = None
    debate_summary: Optional[str] = None
    total_time: Optional[float] = None
    winner_source: Optional[str] = None