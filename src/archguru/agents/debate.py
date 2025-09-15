"""
Cross-model debate and evaluation logic for ArchGuru
Handles model-vs-model competition and consensus building
"""
import asyncio
from typing import List, Dict, Any, Optional
from ..models.decision import ModelResponse
from ..api.openrouter import OpenRouterClient
from ..core.config import Config


class ModelDebateEngine:
    """Manages cross-model debates and evaluation"""

    def __init__(self):
        self.client = OpenRouterClient()

    async def run_cross_model_debate(self, responses: List[ModelResponse]) -> Dict[str, Any]:
        """
        Run cross-model debate where models critique each other's recommendations
        """
        if len(responses) < 2:
            return {
                "debate_summary": "Not enough responses for debate",
                "winning_model": responses[0].model_name if responses else None,
                "consensus_recommendation": responses[0].recommendation if responses else None
            }

        print("🥊 Starting cross-model debate...")

        # Prepare debate context
        debate_context = self._prepare_debate_context(responses)

        # Run arbiter evaluation
        arbiter_result = await self._run_arbiter_evaluation(debate_context, responses)

        return arbiter_result

    def _prepare_debate_context(self, responses: List[ModelResponse]) -> str:
        """Prepare context for the debate with all model responses"""
        context = "ARCHITECTURAL DECISION RESPONSES FROM COMPETING AI MODELS:\n\n"

        for i, response in enumerate(responses, 1):
            context += f"=== MODEL {i}: {response.model_name} ===\n"
            context += f"Recommendation: {response.recommendation}\n"
            context += f"Reasoning: {response.reasoning}\n"
            context += f"Research Steps: {len(response.research_steps)} tool calls\n"
            context += f"Response Time: {response.response_time:.2f}s\n"

            if response.trade_offs:
                context += f"Trade-offs: {', '.join(response.trade_offs)}\n"

            context += "\n" + "="*50 + "\n\n"

        return context

    async def _run_arbiter_evaluation(self, debate_context: str, responses: List[ModelResponse]) -> Dict[str, Any]:
        """Run final arbiter model to evaluate all responses and pick winner"""
        arbiter_model = Config.get_arbiter_model()

        arbiter_prompt = f"""You are the final arbiter in an AI architecture competition. Below are responses from multiple AI models to the same architectural decision request.

{debate_context}

Your task is to:
1. Analyze each model's recommendation for technical accuracy, completeness, and practicality
2. Evaluate the quality of research each model performed
3. Consider which recommendation would work best in production
4. Select the winning model and explain why
5. Synthesize the best ideas into a final consensus recommendation

Provide your evaluation in this format:

WINNER: [model_name]

EVALUATION:
[Your detailed analysis of why this model won, referencing specific strengths and addressing weaknesses of other responses]

CONSENSUS RECOMMENDATION:
[Your synthesized recommendation that combines the best aspects of all responses]

DEBATE SUMMARY:
[Brief summary of the key differences between models and what made the winner stand out]"""

        print(f"  🏅 Arbiter evaluation using {arbiter_model}...")

        try:
            # Get arbiter's evaluation (without research tools for final judgment)
            arbiter_response = await self._get_arbiter_response(arbiter_model, arbiter_prompt)

            # Parse arbiter response
            winner, evaluation, consensus, summary = self._parse_arbiter_response(arbiter_response)

            return {
                "winning_model": winner,
                "debate_summary": summary,
                "consensus_recommendation": consensus,
                "arbiter_evaluation": evaluation,
                "arbiter_model": arbiter_model,
                "winner_source": "arbiter"  # v0.4: Track selection method
            }

        except Exception as e:
            print(f"❌ Arbiter evaluation failed: {str(e)}")
            # Fallback to simple scoring
            return self._fallback_evaluation(responses)

    async def _get_arbiter_response(self, model_name: str, prompt: str) -> str:
        """Get response from arbiter model without research tools"""
        try:
            response = self.client.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.3  # Lower temperature for more consistent evaluation
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Arbiter model {model_name} failed: {str(e)}")

    def _parse_arbiter_response(self, response: str) -> tuple:
        """Parse the arbiter's structured response"""
        lines = response.split('\n')
        winner = None
        evaluation = ""
        consensus = ""
        summary = ""

        current_section = None

        for line in lines:
            line = line.strip()

            if line.startswith("WINNER:"):
                winner = line.replace("WINNER:", "").strip()
            elif line.startswith("EVALUATION:"):
                current_section = "evaluation"
                continue
            elif line.startswith("CONSENSUS RECOMMENDATION:"):
                current_section = "consensus"
                continue
            elif line.startswith("DEBATE SUMMARY:"):
                current_section = "summary"
                continue
            elif line and current_section:
                if current_section == "evaluation":
                    evaluation += line + "\n"
                elif current_section == "consensus":
                    consensus += line + "\n"
                elif current_section == "summary":
                    summary += line + "\n"

        return (
            winner or "No winner selected",
            evaluation.strip() or "No evaluation provided",
            consensus.strip() or "No consensus reached",
            summary.strip() or "No summary available"
        )

    def _fallback_evaluation(self, responses: List[ModelResponse]) -> Dict[str, Any]:
        """Fallback evaluation based on simple metrics when arbiter fails"""
        if not responses:
            return {
                "winning_model": None,
                "debate_summary": "No responses to evaluate",
                "consensus_recommendation": "No recommendations available"
            }

        # Simple scoring: research steps + response quality + confidence
        best_response = None
        best_score = -1

        for response in responses:
            if not getattr(response, 'success', True):
                continue

            score = (
                len(response.research_steps) * 2 +  # Research effort
                len(response.reasoning) / 50 +       # Reasoning depth
                response.confidence_score * 10       # Model confidence
            )

            if score > best_score:
                best_score = score
                best_response = response

        if best_response:
            return {
                "winning_model": best_response.model_name,
                "debate_summary": f"Winner selected by fallback scoring (score: {best_score:.1f})",
                "consensus_recommendation": best_response.recommendation,
                "arbiter_evaluation": "Fallback evaluation used due to arbiter failure",
                "winner_source": "fallback"  # v0.4: Track selection method
            }
        else:
            return {
                "winning_model": responses[0].model_name,
                "debate_summary": "All models failed, selected first response",
                "consensus_recommendation": responses[0].recommendation,
                "winner_source": "fallback"  # v0.4: Track selection method
            }