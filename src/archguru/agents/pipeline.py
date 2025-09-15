"""
LangGraph pipeline for model competition
Chapter 1 MVP: Basic linear pipeline
"""
import asyncio
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from ..core.state import CompetitionState
from ..models.decision import DecisionRequest, DecisionResult
from ..api.openrouter import OpenRouterClient


class ModelCompetitionPipeline:
    """LangGraph pipeline for running model competition"""

    def __init__(self):
        self.client = OpenRouterClient()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph competition pipeline"""
        workflow = StateGraph(CompetitionState)

        # Add nodes
        workflow.add_node("run_competition", self._run_competition)
        workflow.add_node("select_winner", self._select_winner)
        workflow.add_node("generate_consensus", self._generate_consensus)

        # Set entry point
        workflow.set_entry_point("run_competition")

        # Add edges
        workflow.add_edge("run_competition", "select_winner")
        workflow.add_edge("select_winner", "generate_consensus")
        workflow.add_edge("generate_consensus", END)

        return workflow.compile()

    async def _run_competition(self, state: CompetitionState) -> Dict[str, Any]:
        """Run the model competition"""
        request = state["request"]

        try:
            responses = await self.client.run_model_competition(
                decision_type=request.decision_type,
                language=request.language,
                framework=request.framework,
                requirements=request.requirements
            )

            return {
                "model_responses": responses,
                "current_step": "competition_complete"
            }

        except Exception as e:
            return {
                "error_message": f"Competition failed: {str(e)}",
                "current_step": "error"
            }

    async def _select_winner(self, state: CompetitionState) -> Dict[str, Any]:
        """Simple winner selection for MVP - will enhance later"""
        responses = state["model_responses"]

        if not responses:
            return {"winning_model": None}

        # For MVP: select based on fastest response time with no errors
        valid_responses = [r for r in responses if not r.recommendation.startswith("Error:")]

        if valid_responses:
            winner = min(valid_responses, key=lambda r: r.response_time)
            return {"winning_model": winner.model_name}

        return {"winning_model": None}

    async def _generate_consensus(self, state: CompetitionState) -> Dict[str, Any]:
        """Generate consensus recommendation - simplified for MVP"""
        responses = state["model_responses"]
        winning_model = state.get("winning_model")

        if winning_model:
            winner_response = next((r for r in responses if r.model_name == winning_model), None)
            if winner_response:
                return {
                    "consensus_recommendation": f"Based on model competition, recommended approach: {winner_response.recommendation}",
                    "current_step": "complete"
                }

        return {
            "consensus_recommendation": "No clear consensus reached",
            "current_step": "complete"
        }

    async def run(self, request: DecisionRequest) -> DecisionResult:
        """Run the complete pipeline"""
        initial_state: CompetitionState = {
            "request": request,
            "model_responses": [],
            "winning_model": None,
            "consensus_recommendation": None,
            "error_message": None,
            "current_step": "starting"
        }

        print("🔄 Starting LangGraph pipeline...")
        result = await self.graph.ainvoke(initial_state)

        return DecisionResult(
            request=request,
            model_responses=result.get("model_responses", []),
            winning_model=result.get("winning_model"),
            consensus_recommendation=result.get("consensus_recommendation"),
            debate_summary="MVP: Basic competition complete"
        )