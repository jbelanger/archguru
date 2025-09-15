"""
LangGraph pipeline for single model autonomous research
v0.1: Single model with research tools
"""
import asyncio
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from ..core.state import CompetitionState
from ..models.decision import DecisionRequest, DecisionResult
from ..api.openrouter import OpenRouterClient


class ModelResearchPipeline:
    """LangGraph pipeline for single model autonomous research"""

    def __init__(self):
        self.client = OpenRouterClient()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph research pipeline"""
        workflow = StateGraph(CompetitionState)

        # Add nodes for v0.1 single model research
        workflow.add_node("run_research", self._run_research)
        workflow.add_node("generate_result", self._generate_result)

        # Set entry point
        workflow.set_entry_point("run_research")

        # Add edges
        workflow.add_edge("run_research", "generate_result")
        workflow.add_edge("generate_result", END)

        return workflow.compile()

    async def _run_research(self, state: CompetitionState) -> Dict[str, Any]:
        """Run single model autonomous research"""
        request = state["request"]

        try:
            response = await self.client.run_single_model_research(
                decision_type=request.decision_type,
                language=request.language,
                framework=request.framework,
                requirements=request.requirements
            )

            return {
                "model_responses": [response],
                "current_step": "research_complete"
            }

        except Exception as e:
            return {
                "error_message": f"Research failed: {str(e)}",
                "current_step": "error"
            }

    async def _generate_result(self, state: CompetitionState) -> Dict[str, Any]:
        """Generate final result from research"""
        responses = state["model_responses"]

        if responses and len(responses) > 0:
            response = responses[0]
            if not response.recommendation.startswith("Error:"):
                research_summary = f"Research completed using {len(response.research_steps)} tool calls"
                return {
                    "consensus_recommendation": response.recommendation,
                    "debate_summary": research_summary,
                    "current_step": "complete"
                }

        return {
            "consensus_recommendation": "Research failed to produce recommendations",
            "debate_summary": "No research completed",
            "current_step": "complete"
        }

    async def run(self, request: DecisionRequest) -> DecisionResult:
        """Run the complete research pipeline"""
        initial_state: CompetitionState = {
            "request": request,
            "model_responses": [],
            "winning_model": None,
            "consensus_recommendation": None,
            "error_message": None,
            "current_step": "starting"
        }

        print("🔄 Starting LangGraph research pipeline...")
        result = await self.graph.ainvoke(initial_state)

        return DecisionResult(
            request=request,
            model_responses=result.get("model_responses", []),
            winning_model=result.get("model_responses", [None])[0].model_name if result.get("model_responses") else None,
            consensus_recommendation=result.get("consensus_recommendation"),
            debate_summary=result.get("debate_summary", "Single model research complete")
        )