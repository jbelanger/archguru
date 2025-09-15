"""
Cross-model debate and evaluation logic for ArchGuru
Handles model-vs-model competition and consensus building
"""
import asyncio
import re
from typing import List, Dict, Any, Optional
from ..models.decision import ModelResponse
from ..api.openrouter import OpenRouterClient
from ..core.config import Config
from ..core.constants import ARBITER_MAX_TOKENS, ARBITER_TEMPERATURE, WinnerSource
from ..core.response_parser import ResponseParser


class ModelDebateEngine:
    """Manages cross-model debates and evaluation"""

    def __init__(self):
        self.client = OpenRouterClient()
        self.parser = ResponseParser()

    async def run_cross_model_debate(self, responses: List[ModelResponse]) -> Dict[str, Any]:
        """
        Run cross-model debate where models critique each other's recommendations
        """
        if len(responses) < 2:
            return {
                "debate_summary": "Not enough responses for debate",
                "winning_model": responses[0].model_name if responses else None,
                "consensus_recommendation": responses[0].recommendation if responses else None,
                "winner_source": WinnerSource.SINGLE_VALID.value
            }

        print("🥊 Starting cross-model debate...")

        # Prepare debate context
        debate_context = self._prepare_debate_context(responses)

        # Run arbiter evaluation
        arbiter_result = await self._run_arbiter_evaluation(debate_context, responses)

        return arbiter_result

    def _prepare_debate_context(self, responses: List[ModelResponse]) -> str:
        """Prepare context for the debate with all model responses"""
        def clip(s, n):
            return (s or "")[:n].rstrip()

        context = "ARCHITECTURAL DECISION RESPONSES FROM COMPETING AI MODELS:\n\n"

        valid_responses = []
        for i, response in enumerate(responses, 1):
            # Skip failed responses for debate context
            if not getattr(response, 'success', True) or not response.recommendation or response.recommendation.startswith("Error:"):
                context += f"=== MODEL {i}: {response.model_name} ===\n"
                context += f"STATUS: FAILED - {clip(response.recommendation, 100)}\n"
                context += "\n" + "="*50 + "\n\n"
                continue

            valid_responses.append(response)
            context += f"=== MODEL {i}: {response.model_name} ===\n"
            context += f"Recommendation: {clip(response.recommendation, 240)}\n"
            context += f"Reasoning: {clip(response.reasoning, 900)}\n"

            # Enhanced research behavior tracking
            research_steps = len(response.research_steps or [])
            skipped_research = getattr(response, 'skipped_research', False)
            if skipped_research:
                context += f"Research Steps: {research_steps} tool calls (SKIPPED EXPECTED RESEARCH)\n"
            else:
                context += f"Research Steps: {research_steps} tool calls\n"

            context += f"Response Time: {response.response_time:.2f}s\n"

            if response.trade_offs:
                context += f"Trade-offs: {', '.join(response.trade_offs[:5])}\n"

            context += "\n" + "="*50 + "\n\n"

        # If no valid responses, return early
        if not valid_responses:
            context += "\nNOTE: No models provided valid responses for evaluation."

        return context

    async def _run_arbiter_evaluation(self, debate_context: str, responses: List[ModelResponse]) -> Dict[str, Any]:
        """Run final arbiter model to evaluate all responses and pick winner"""
        # Check if we have valid responses to evaluate
        valid_responses = [r for r in responses if getattr(r, 'success', True) and r.recommendation and not r.recommendation.startswith("Error:")]

        if not valid_responses:
            print(f"  ⚠️  No valid responses to evaluate (total responses: {len(responses)})")
            return {
                "winning_model": None,
                "debate_summary": "No valid responses to evaluate",
                "consensus_recommendation": "Final Recommendation: All models failed to provide valid responses.",
                "arbiter_evaluation": "No models provided valid responses for evaluation",
                "winner_source": WinnerSource.NO_VALID.value
            }

        if len(valid_responses) == 1:
            winner = valid_responses[0]
            print(f"  ℹ️  Only one valid response, auto-selecting {winner.model_name}")
            return {
                "winning_model": winner.model_name,
                "debate_summary": f"Only one valid response from {winner.model_name}",
                "consensus_recommendation": self._ensure_strong_reco(winner.recommendation),
                "arbiter_evaluation": f"Single valid response from {winner.model_name}",
                "winner_source": WinnerSource.SINGLE_VALID.value
            }

        arbiter_model = Config.get_arbiter_model()

        arbiter_prompt = f"""You are an expert technical evaluator analyzing architectural recommendations. Below are responses from multiple AI systems providing guidance on the same architectural decision.

{debate_context}

Your task is to evaluate each response using this structured rubric, then select the most comprehensive and practical recommendation:

EVALUATION RUBRIC (Rate each model 1-5 for each criterion):
1. EVIDENCE QUALITY: How well-researched and credible are the sources/examples cited?
2. RISK AWARENESS: Does the model identify key risks, limitations, and failure modes?
3. CLARITY & STRUCTURE: Is the recommendation clear, well-organized, and actionable?
4. PRODUCTION READINESS: How practical and implementable is the solution in real environments?

For each response, provide a brief evaluation using this rubric, then select the most suitable recommendation.

Provide your evaluation in this format:

EVALUATION SCORING:
Response 1 ({responses[0].model_name if responses else 'N/A'}):
- Evidence Quality: [1-5]/5 - [brief reason]
- Risk Awareness: [1-5]/5 - [brief reason]
- Clarity & Structure: [1-5]/5 - [brief reason]
- Production Readiness: [1-5]/5 - [brief reason]

Response 2 ({responses[1].model_name if len(responses) > 1 else 'N/A'}):
- Evidence Quality: [1-5]/5 - [brief reason]
- Risk Awareness: [1-5]/5 - [brief reason]
- Clarity & Structure: [1-5]/5 - [brief reason]
- Production Readiness: [1-5]/5 - [brief reason]

{f"Response 3 ({responses[2].model_name}):" if len(responses) > 2 else ""}
{f"- Evidence Quality: [1-5]/5 - [brief reason]" if len(responses) > 2 else ""}
{f"- Risk Awareness: [1-5]/5 - [brief reason]" if len(responses) > 2 else ""}
{f"- Clarity & Structure: [1-5]/5 - [brief reason]" if len(responses) > 2 else ""}
{f"- Production Readiness: [1-5]/5 - [brief reason]" if len(responses) > 2 else ""}

SELECTED: Response 1|Response 2|Response 3  (choose the most comprehensive)

SELECTION REASONING:
[One paragraph explaining why this response was selected based on the evaluation scores]

CONSENSUS RECOMMENDATION:
Start with: "Final Recommendation: <one concise sentence>"
Then provide 3–6 reasoning bullets, trade-offs, and 3–7 implementation steps.
Include a short "Evidence:" bullet list if you referenced any repos.

SUMMARY:
[Brief summary of the key differences between responses and what made the selected one stand out]"""

        print(f"  🏅 Arbiter evaluation using {arbiter_model}...")

        try:
            # Get arbiter's evaluation (without research tools for final judgment)
            arbiter_response = await self._get_arbiter_response(arbiter_model, arbiter_prompt)
        except Exception as e1:
            print(f"❌ Primary arbiter failed: {e1}. Retrying with backup model...")
            backup = Config.get_models()[0]  # Use first competitor model as backup
            try:
                arbiter_response = await self._get_arbiter_response(backup, arbiter_prompt)
                arbiter_model = backup
                print(f"  ✅ Backup arbiter {backup} succeeded")
            except Exception as e2:
                print(f"❌ Backup arbiter failed: {e2}")
                return self._fallback_evaluation(responses)

        try:
            # Parse arbiter response
            winner, evaluation, consensus, summary = self._parse_arbiter_response(arbiter_response, responses)

            if winner and winner != "No winner selected":
                print(f"  ✅ Arbiter selected: {winner}")
            else:
                print(f"  ⚠️  Arbiter failed to select winner: {winner}")

            return {
                "winning_model": winner,
                "debate_summary": summary,
                "consensus_recommendation": consensus,
                "arbiter_evaluation": evaluation,
                "arbiter_model": arbiter_model,
                "winner_source": WinnerSource.ARBITER.value
            }

        except Exception as e:
            print(f"❌ Arbiter response parsing failed: {str(e)}")
            print(f"  🔍 Arbiter response preview: {repr(arbiter_response[:200]) if 'arbiter_response' in locals() else 'No response available'}")
            # Fallback to simple scoring
            return self._fallback_evaluation(responses)

    async def _get_arbiter_response(self, model_name: str, prompt: str) -> str:
        """Get response from arbiter model without research tools"""
        try:
            response = self.client.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=ARBITER_MAX_TOKENS,
                temperature=ARBITER_TEMPERATURE
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise Exception(f"Arbiter model {model_name} failed: {str(e)}")

    def _parse_arbiter_response(self, response: str, responses: List[ModelResponse]) -> tuple:
        """Parse the arbiter's structured response with rubric scoring"""
        lines = response.split('\n')
        winner = None
        rubric_scores = ""
        winner_reasoning = ""
        consensus = ""
        summary = ""

        current_section = None

        for line in lines:
            line = line.strip()

            # Parse selected response with flexible matching
            if line.startswith("SELECTED:") or line.startswith("WINNER:") or line.startswith("Winner:") or line.startswith("CHOSEN:"):
                winner_raw = re.sub(r'^(SELECTED|WINNER|Winner|CHOSEN)\s*[:\-]\s*', '', line, flags=re.IGNORECASE).strip()
                winner = self._resolve_winner_token(winner_raw, responses)
            elif line.startswith("EVALUATION SCORING:") or line.startswith("RUBRIC SCORING:"):
                current_section = "rubric"
                continue
            elif line.startswith("SELECTION REASONING:") or line.startswith("WINNER REASONING:"):
                current_section = "winner_reasoning"
                continue
            elif line.startswith("CONSENSUS RECOMMENDATION:"):
                current_section = "consensus"
                continue
            elif line.startswith("DEBATE SUMMARY:") or line.startswith("SUMMARY:"):
                current_section = "summary"
                continue
            elif line and current_section:
                if current_section == "rubric":
                    rubric_scores += line + "\n"
                elif current_section == "winner_reasoning":
                    winner_reasoning += line + "\n"
                elif current_section == "consensus":
                    consensus += line + "\n"
                elif current_section == "summary":
                    summary += line + "\n"

        # Combine rubric scores and winner reasoning for the evaluation field
        evaluation = ""
        if rubric_scores.strip():
            evaluation += "RUBRIC SCORING:\n" + rubric_scores.strip() + "\n\n"
        if winner_reasoning.strip():
            evaluation += "WINNER REASONING:\n" + winner_reasoning.strip()

        return (
            winner or "No winner selected",
            evaluation.strip() or "No evaluation provided",
            self._ensure_strong_reco(consensus.strip()) or "Final Recommendation: No consensus reached.",
            summary.strip() or "No summary available"
        )

    def _resolve_winner_token(self, token: str, responses: List[ModelResponse]) -> Optional[str]:
        """Resolve winner token to actual model name"""
        if not token:
            return None

        t = token.strip()

        # Reject placeholder lists (arbiter echoing the prompt)
        if '|' in t:
            print(f"  ⚠️  Arbiter returned placeholder list: {t}")
            return None

        # Accept "Model 1", "Response 1", or just "1" format - map to actual model name
        m = re.search(r'\bResponse\s*(\d+)\b', t, re.IGNORECASE)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(responses):
                return responses[idx].model_name
            return None

        # Accept direct model name (partial or full match)
        if len(t) > 3:  # Avoid matching very short strings
            t_low = t.lower()
            for response in responses:
                if response.model_name.lower() in t_low or t_low in response.model_name.lower():
                    return response.model_name

        return None

    def _ensure_strong_reco(self, text: str) -> str:
        """Post-parse guard to ensure recommendation starts with 'Final Recommendation:'"""
        if not text:
            return "Final Recommendation: No consensus reached."
        t = text.strip()

        # If the model already complied, keep exactly that line
        for line in t.splitlines():
            s = line.strip()
            if s.lower().startswith("final recommendation:"):
                return s

        # Skip headings and pick the first substantive line
        skip = {"reasoning:", "trade-offs:", "implementation steps:", "evidence:"}
        for line in t.splitlines():
            s = line.strip()
            if not s or s.lower() in skip or s.endswith(":"):
                continue
            first_sentence = s.split(".")[0].strip()
            if first_sentence:
                print(f"     🔧 Guard: normalized consensus header")
                return f"Final Recommendation: {first_sentence[:160]}."
        return "Final Recommendation: No consensus reached."

    def _fallback_evaluation(self, responses: List[ModelResponse]) -> Dict[str, Any]:
        """Fallback evaluation based on simple metrics when arbiter fails"""
        if not responses:
            return {
                "winning_model": None,
                "debate_summary": "No responses to evaluate",
                "consensus_recommendation": "No recommendations available",
                "winner_source": WinnerSource.NO_VALID.value
            }

        # Simple scoring: research steps + response quality + confidence
        best_response = None
        best_score = -1.0

        for response in responses:
            if not getattr(response, 'success', True):
                continue

            score = (
                len(response.research_steps or []) * 2 +  # Research effort
                len(response.reasoning or "") / 50 +       # Reasoning depth
                (response.confidence_score or 0) * 10     # Model confidence
            )

            if score > best_score:
                best_score = score
                best_response = response

        if best_response:
            return {
                "winning_model": best_response.model_name,
                "debate_summary": f"Winner selected by fallback scoring (score: {best_score:.1f})",
                "consensus_recommendation": self._ensure_strong_reco(best_response.recommendation),
                "arbiter_evaluation": "Fallback evaluation used due to arbiter failure",
                "winner_source": WinnerSource.FALLBACK.value
            }
        else:
            return {
                "winning_model": responses[0].model_name,
                "debate_summary": "All models failed, selected first response",
                "consensus_recommendation": self._ensure_strong_reco(responses[0].recommendation),
                "winner_source": WinnerSource.FALLBACK.value
            }