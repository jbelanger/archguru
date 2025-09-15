from typing import Dict, Any
from langgraph.graph import StateGraph, END
from ..core.state import CompetitionState
from ..core.constants import WinnerSource
from ..models.decision import DecisionRequest, DecisionResult
from ..api.openrouter import OpenRouterClient
from .debate import ModelDebateEngine

class ModelCompetitionPipeline:
    """Simplified LangGraph pipeline for multi-model competition"""

    def __init__(self):
        self.client = OpenRouterClient()
        self.debate_engine = ModelDebateEngine()
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Build the LangGraph competition pipeline"""
        workflow = StateGraph(CompetitionState)

        # Add nodes
        workflow.add_node("compete", self._run_competition)
        workflow.add_node("debate", self._run_debate)
        workflow.add_node("finalize", self._finalize_result)

        # Set flow
        workflow.set_entry_point("compete")
        workflow.add_edge("compete", "debate")
        workflow.add_edge("debate", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    async def _run_competition(self, state: CompetitionState) -> CompetitionState:
        """Run multi-model competition"""
        request = state["request"]
        
        try:
            responses = await self.client.run_model_competition(
                decision_type=request.decision_type,
                language=request.language,
                framework=request.framework,
                requirements=request.requirements
            )
            
            state["model_responses"] = responses
            state["current_step"] = "competition_complete"
        except Exception as e:
            state["error_message"] = f"Competition failed: {str(e)}"
            state["current_step"] = "error"
        
        return state

    async def _run_debate(self, state: CompetitionState) -> CompetitionState:
        """Run cross-model debate"""
        if not state.get("model_responses"):
            state["error_message"] = "No responses to debate"
            state["current_step"] = "error"
            return state
        
        try:
            debate_result = await self.debate_engine.run_cross_model_debate(
                state["model_responses"]
            )
            
            state.update({
                "winning_model": debate_result["winning_model"],
                "consensus_recommendation": debate_result["consensus_recommendation"],
                "debate_summary": debate_result["debate_summary"],
                "arbiter_evaluation": debate_result.get("arbiter_evaluation"),
                "winner_source": debate_result.get("winner_source", WinnerSource.ARBITER),
                "current_step": "debate_complete"
            })
        except Exception as e:
            state["error_message"] = f"Debate failed: {str(e)}"
            state["current_step"] = "error"
        
        return state

    async def _finalize_result(self, state: CompetitionState) -> CompetitionState:
        """Finalize competition result"""
        responses = state.get("model_responses", [])
        successful = sum(1 for r in responses if getattr(r, 'success', True))
        
        if state.get("consensus_recommendation"):
            state["debate_summary"] = (
                f"Competition complete: {successful}/{len(responses)} models succeeded. "
                f"Winner: {state.get('winning_model', 'None')}"
            )
        else:
            state["consensus_recommendation"] = "Competition failed to produce results"
            state["debate_summary"] = "Model competition encountered errors"
        
        state["current_step"] = "complete"
        return state

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

        print("🔄 Starting model competition pipeline...")
        result = await self.graph.ainvoke(initial_state)

        return DecisionResult(
            request=request,
            model_responses=result.get("model_responses", []),
            winning_model=result.get("winning_model"),
            consensus_recommendation=result.get("consensus_recommendation"),
            debate_summary=result.get("debate_summary", "Competition complete"),
            arbiter_evaluation=result.get("arbiter_evaluation"),
            winner_source=result.get("winner_source")
        )