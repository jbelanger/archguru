"""
OpenRouter API integration for model competition
Handles requests to multiple models via OpenRouter
"""
import asyncio
from typing import List, Dict, Any
from openai import OpenAI
from ..core.config import Config
from ..models.decision import ModelResponse
import time


class OpenRouterClient:
    """Client for interacting with OpenRouter API"""

    def __init__(self):
        if not Config.validate():
            raise ValueError("Invalid configuration")

        self.client = OpenAI(
            base_url=Config.OPENROUTER_BASE_URL,
            api_key=Config.OPENROUTER_API_KEY
        )

    def generate_prompt(self, decision_type: str, language: str = None,
                       framework: str = None, requirements: str = None) -> str:
        """Generate a prompt for architectural decision"""
        prompt = f"""You are an expert software architect. Provide a detailed recommendation for this architectural decision:

Decision Type: {decision_type}
Language/Stack: {language or 'Not specified'}
Framework: {framework or 'Not specified'}
Requirements: {requirements or 'None specified'}

Please provide:
1. Your specific recommendation
2. Detailed reasoning for your choice
3. Trade-offs and alternatives considered
4. Implementation considerations

Be concise but thorough. Focus on practical, production-ready advice."""

        return prompt

    async def get_model_response(self, model_name: str, team: str, prompt: str) -> ModelResponse:
        """Get response from a single model"""
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.7
            )

            response_time = time.time() - start_time
            content = response.choices[0].message.content

            # Simple parsing for MVP - will enhance in later chapters
            parts = content.split('\n\n')
            recommendation = parts[0] if parts else content[:200]
            reasoning = parts[1] if len(parts) > 1 else "See full response"

            return ModelResponse(
                model_name=model_name,
                team=team,
                recommendation=recommendation,
                reasoning=reasoning,
                trade_offs=["Will be parsed in later chapters"],
                confidence_score=0.8,  # Placeholder for MVP
                response_time=response_time
            )

        except Exception as e:
            print(f"❌ Error with {model_name}: {str(e)}")
            return ModelResponse(
                model_name=model_name,
                team=team,
                recommendation=f"Error: {str(e)}",
                reasoning="Model failed to respond",
                trade_offs=[],
                confidence_score=0.0,
                response_time=time.time() - start_time
            )

    async def run_model_competition(self, decision_type: str, language: str = None,
                                   framework: str = None, requirements: str = None) -> List[ModelResponse]:
        """Run competition between all model teams"""
        prompt = self.generate_prompt(decision_type, language, framework, requirements)
        responses = []

        print("🤖 Starting model competition...")

        # For MVP, just use one model from each team to keep it simple
        test_models = [
            ("openai/gpt-4o-mini", "openai"),
            ("x-ai/grok-code-fast-1", "claude"),
            ("meta-llama/llama-3.2-3b-instruct:free", "llama")
        ]

        for model_name, team in test_models:
            print(f"  📊 Querying {team} team ({model_name})...")
            response = await self.get_model_response(model_name, team, prompt)
            responses.append(response)

        return responses