"""
LangGraph state management for model competition pipeline
"""
from typing import List, Optional, TypedDict
from ..models.decision import DecisionRequest, ModelResponse, DecisionResult


class CompetitionState(TypedDict):
    """State for the model competition pipeline"""
    request: DecisionRequest
    model_responses: List[ModelResponse]
    winning_model: Optional[str]
    consensus_recommendation: Optional[str]
    debate_summary: Optional[str]
    arbiter_evaluation: Optional[str]
    winner_source: Optional[str]
    error_message: Optional[str]
    current_step: str