"""
LangGraph pipeline for model team competition and debate
Phase 2: Multi-model competition with cross-model debate
"""
from typing import Dict, Any, Union
from langgraph.graph import StateGraph, END
from ..core.state import CompetitionState
from ..models.decision import DecisionRequest, DecisionResult
from ..api.openrouter import OpenRouterClient
from .debate import ModelDebateEngine


class ModelCompetitionPipeline:
    """LangGraph pipeline for multi-model team competition"""

    def __init__(self):
        self.client = OpenRouterClient()
        self.debate_engine = ModelDebateEngine()
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Build the LangGraph competition pipeline"""
        workflow = StateGraph(CompetitionState)

        # Phase 2: Multi-model competition
        workflow.add_node("run_model_competition", self._run_model_competition)
        workflow.add_node("run_debate", self._run_debate)
        workflow.add_node("generate_final_result", self._generate_final_result)

        # Set entry point
        workflow.set_entry_point("run_model_competition")

        # Add edges
        workflow.add_edge("run_model_competition", "run_debate")
        workflow.add_edge("run_debate", "generate_final_result")
        workflow.add_edge("generate_final_result", END)

        return workflow.compile()


    async def _run_model_competition(self, state: CompetitionState) -> Dict[str, Any]:
        """Run multi-model team competition"""
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
                "error_message": f"Model competition failed: {str(e)}",
                "current_step": "error"
            }

    async def _run_debate(self, state: CompetitionState) -> Dict[str, Any]:
        """Run cross-model debate and evaluation"""
        responses = state["model_responses"]

        if not responses:
            return {
                "error_message": "No model responses to debate",
                "current_step": "error"
            }

        try:
            debate_result = await self.debate_engine.run_cross_model_debate(responses)

            return {
                "winning_model": debate_result["winning_model"],
                "consensus_recommendation": debate_result["consensus_recommendation"],
                "debate_summary": debate_result["debate_summary"],
                "arbiter_evaluation": debate_result.get("arbiter_evaluation"),
                "winner_source": debate_result.get("winner_source", "unknown"),  # v0.4: Track selection method
                "current_step": "debate_complete"
            }

        except Exception as e:
            return {
                "error_message": f"Debate failed: {str(e)}",
                "current_step": "error"
            }

    async def _generate_final_result(self, state: CompetitionState) -> CompetitionState:
        """Generate final competition result"""
        responses = state["model_responses"]
        winning_model = state.get("winning_model")
        consensus = state.get("consensus_recommendation")
        winner_source = state.get("winner_source")
        arbiter_evaluation = state.get("arbiter_evaluation")

        if responses and winning_model and consensus:
            successful_models = [r for r in responses if getattr(r, 'success', True)]
            # Update only specific fields instead of replacing the entire state
            state["consensus_recommendation"] = consensus
            state["debate_summary"] = f"Competition complete: {len(successful_models)}/{len(responses)} models succeeded. Winner: {winning_model}"
            state["current_step"] = "complete"
            return state  # Return the entire state to preserve all fields
        else:
            # Update only specific fields instead of replacing the entire state
            state["consensus_recommendation"] = "Competition failed to produce results"
            state["debate_summary"] = "Model competition encountered errors"
            state["current_step"] = "complete"
            return state  # Return the entire state to preserve all fields

    async def run(self, request: DecisionRequest) -> DecisionResult:
        """Run the complete competition pipeline"""
        initial_state: CompetitionState = {
            "request": request,
            "model_responses": [],
            "winning_model": None,
            "consensus_recommendation": None,
            "debate_summary": None,
            "arbiter_evaluation": None,
            "winner_source": None,
            "error_message": None,
            "current_step": "starting"
        }

        print("🔄 Starting LangGraph model competition pipeline...")
        result = await self.graph.ainvoke(initial_state)

        return DecisionResult(
            request=request,
            model_responses=result.get("model_responses", []),
            winning_model=result.get("winning_model"),
            consensus_recommendation=result.get("consensus_recommendation"),
            debate_summary=result.get("debate_summary", "Competition complete"),
            arbiter_evaluation=result.get("arbiter_evaluation"),  # v0.6: Pass detailed evaluation
            winner_source=result.get("winner_source")
        )