import asyncio
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
        """Run cross-model debate where models critique each other's recommendations"""
        
        # Filter valid responses
        valid_responses = self._get_valid_responses(responses)
        
        # Handle edge cases
        if not valid_responses:
            return self._no_valid_responses_result()
        
        if len(valid_responses) == 1:
            return self._single_valid_response_result(valid_responses[0])
        
        # Run arbiter evaluation
        print("🥊 Starting cross-model debate...")
        debate_context = self._prepare_debate_context(responses)
        return await self._run_arbiter_evaluation(debate_context, valid_responses)

    def _get_valid_responses(self, responses: List[ModelResponse]) -> List[ModelResponse]:
        """Filter to only valid responses"""
        return [
            r for r in responses 
            if getattr(r, 'success', True) 
            and r.recommendation 
            and not r.recommendation.startswith("Error:")
        ]

    def _no_valid_responses_result(self) -> Dict[str, Any]:
        """Result when no valid responses exist"""
        print("  ⚠️  No valid responses to evaluate")
        return {
            "winning_model": None,
            "debate_summary": "No valid responses to evaluate",
            "consensus_recommendation": "Final Recommendation: All models failed to provide valid responses.",
            "arbiter_evaluation": "No models provided valid responses for evaluation",
            "winner_source": WinnerSource.NO_VALID
        }

    def _single_valid_response_result(self, response: ModelResponse) -> Dict[str, Any]:
        """Result when only one valid response exists"""
        print(f"  ℹ️  Only one valid response, auto-selecting {response.model_name}")
        return {
            "winning_model": response.model_name,
            "debate_summary": f"Only one valid response from {response.model_name}",
            "consensus_recommendation": self.parser.ensure_final_recommendation(response.recommendation),
            "arbiter_evaluation": f"Single valid response from {response.model_name}",
            "winner_source": WinnerSource.SINGLE_VALID
        }

    def _prepare_debate_context(self, responses: List[ModelResponse]) -> str:
        """Prepare context for the debate with all model responses"""
        context_lines = ["ARCHITECTURAL DECISION RESPONSES FROM COMPETING AI MODELS:\n"]
        
        for i, response in enumerate(responses, 1):
            context_lines.append(f"=== MODEL {i}: {response.model_name} ===")
            
            if not getattr(response, 'success', True):
                context_lines.append(f"STATUS: FAILED - {response.recommendation[:100]}")
            else:
                context_lines.append(f"Recommendation: {response.recommendation[:240]}")
                context_lines.append(f"Reasoning: {response.reasoning[:900]}")
                context_lines.append(f"Research Steps: {len(response.research_steps or [])} tool calls")
                context_lines.append(f"Response Time: {response.response_time:.2f}s")
                
                if response.trade_offs:
                    context_lines.append(f"Trade-offs: {', '.join(response.trade_offs[:5])}")
            
            context_lines.append("=" * 50 + "\n")
        
        return "\n".join(context_lines)

    async def _run_arbiter_evaluation(
        self, 
        debate_context: str, 
        valid_responses: List[ModelResponse]
    ) -> Dict[str, Any]:
        """Run final arbiter model to evaluate all responses and pick winner"""
        
        arbiter_model = Config.get_arbiter_model()
        print(f"  🏅 Arbiter evaluation using {arbiter_model}...")
        
        arbiter_prompt = self._build_arbiter_prompt(debate_context, valid_responses)
        
        try:
            arbiter_response = await self._get_arbiter_response(arbiter_model, arbiter_prompt)
            return self._parse_arbiter_result(arbiter_response, valid_responses, arbiter_model)
        except Exception as e:
            print(f"❌ Arbiter failed: {e}. Using fallback...")
            return self._fallback_evaluation(valid_responses)

    def _build_arbiter_prompt(self, context: str, responses: List[ModelResponse]) -> str:
        """Build structured arbiter evaluation prompt"""
        
        # Build response list for prompt
        response_names = [f"Response {i+1} ({r.model_name})" for i, r in enumerate(responses[:3])]
        
        return f"""You are an expert technical evaluator analyzing architectural recommendations.

{context}

Evaluate each response using this rubric (1-5 scale):
1. EVIDENCE QUALITY: How well-researched and credible are the sources?
2. RISK AWARENESS: Does the model identify key risks and limitations?
3. CLARITY & STRUCTURE: Is the recommendation clear and actionable?
4. PRODUCTION READINESS: How practical is the solution?

Provide your evaluation in this EXACT format:

SELECTED: {' | '.join([f'Response {i+1}' for i in range(len(responses))])}

SELECTION REASONING:
[One paragraph explaining why this response was selected]

CONSENSUS RECOMMENDATION:
Final Recommendation: [one concise sentence]
[3-6 reasoning bullets, trade-offs, and implementation steps]

SUMMARY:
[Brief summary of key differences between responses]"""

    async def _get_arbiter_response(self, model_name: str, prompt: str) -> str:
        """Get response from arbiter model"""
        response = self.client.client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=ARBITER_MAX_TOKENS,
            temperature=ARBITER_TEMPERATURE
        )
        return response.choices[0].message.content or ""

    def _parse_arbiter_result(
        self, 
        response: str, 
        valid_responses: List[ModelResponse],
        arbiter_model: str
    ) -> Dict[str, Any]:
        """Parse arbiter response into structured result"""
        
        sections = self._extract_arbiter_sections(response)
        winner = self._resolve_winner(sections.get("selected", ""), valid_responses)
        
        return {
            "winning_model": winner or "No winner selected",
            "debate_summary": sections.get("summary", "Competition complete"),
            "consensus_recommendation": self.parser.ensure_final_recommendation(
                sections.get("consensus", "")
            ),
            "arbiter_evaluation": sections.get("reasoning", "No evaluation provided"),
            "arbiter_model": arbiter_model,
            "winner_source": WinnerSource.ARBITER
        }

    def _extract_arbiter_sections(self, response: str) -> Dict[str, str]:
        """Extract sections from arbiter response"""
        sections = {}
        current_section = None
        current_content = []
        
        for line in response.split('\n'):
            line = line.strip()
            
            if line.startswith("SELECTED:"):
                sections["selected"] = line.replace("SELECTED:", "").strip()
            elif line.startswith("SELECTION REASONING:"):
                current_section = "reasoning"
                current_content = []
            elif line.startswith("CONSENSUS RECOMMENDATION:"):
                if current_section and current_content:
                    sections[current_section] = "\n".join(current_content)
                current_section = "consensus"
                current_content = []
            elif line.startswith("SUMMARY:"):
                if current_section and current_content:
                    sections[current_section] = "\n".join(current_content)
                current_section = "summary"
                current_content = []
            elif line and current_section:
                current_content.append(line)
        
        # Save last section
        if current_section and current_content:
            sections[current_section] = "\n".join(current_content)
        
        return sections

    def _resolve_winner(self, selected: str, responses: List[ModelResponse]) -> Optional[str]:
        """Resolve winner selection to model name"""
        import re
        
        if not selected:
            return None
        
        # Try to match "Response N" pattern
        match = re.match(r'Response\s*(\d+)', selected, re.IGNORECASE)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(responses):
                return responses[idx].model_name
        
        # Try partial model name match
        selected_lower = selected.lower()
        for response in responses:
            if response.model_name.lower() in selected_lower:
                return response.model_name
        
        return None

    def _fallback_evaluation(self, responses: List[ModelResponse]) -> Dict[str, Any]:
        """Simple scoring fallback when arbiter fails"""
        if not responses:
            return self._no_valid_responses_result()
        
        # Score based on research depth and response quality
        best_response = max(
            responses,
            key=lambda r: (
                len(r.research_steps or []) * 2 +
                len(r.reasoning or "") / 50 +
                (r.confidence_score or 0) * 10
            )
        )
        
        return {
            "winning_model": best_response.model_name,
            "debate_summary": "Winner selected by fallback scoring",
            "consensus_recommendation": self.parser.ensure_final_recommendation(
                best_response.recommendation
            ),
            "arbiter_evaluation": "Fallback evaluation used",
            "winner_source": WinnerSource.FALLBACK
        }