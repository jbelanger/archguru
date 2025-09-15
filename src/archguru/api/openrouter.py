"""
OpenRouter API integration with function calling support
Handles LLM requests with research tool access
"""
import asyncio
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from ..core.config import Config
from ..models.decision import ModelResponse
from .github import GitHubClient
from .reddit import RedditClient
from .stackoverflow import StackOverflowClient
import time

# v0.5: Strict output format constant for team generation
STRICT_OUTPUT_FORMAT = """OUTPUT FORMAT (STRICT):
Final Recommendation: <one concise sentence>

Reasoning:
- <3–6 bullets, ≤12 words each>

Trade-offs:
- <2–5 bullets, name the axis>

Implementation Steps:
- <3–7 bullets, concrete>

Evidence:
- <0–5 GitHub links or repo names only>"""


class OpenRouterClient:
    """Client for interacting with OpenRouter API with research tools"""

    def __init__(self):
        if not Config.validate():
            raise ValueError("Invalid configuration")

        self.client = OpenAI(
            base_url=Config.OPENROUTER_BASE_URL,
            api_key=Config.OPENROUTER_API_KEY
        )

        # Initialize research tools
        self.github = GitHubClient()
        self.reddit = RedditClient()
        self.stackoverflow = StackOverflowClient()

    def get_research_tools(self) -> List[Dict[str, Any]]:
        """Define available research tools for the LLM"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_github_repos",
                    "description": "Search GitHub repositories for examples and patterns",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "language": {"type": "string", "description": "Programming language filter"},
                            "limit": {"type": "integer", "description": "Number of results (max 10)", "default": 5}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_reddit_discussions",
                    "description": "Search Reddit for community discussions and opinions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "subreddits": {"type": "array", "items": {"type": "string"}, "description": "Specific subreddits to search"},
                            "limit": {"type": "integer", "description": "Number of results (max 10)", "default": 5}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_stackoverflow",
                    "description": "Search StackOverflow for technical questions and solutions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "tags": {"type": "array", "items": {"type": "string"}, "description": "Technology tags to filter by"},
                            "limit": {"type": "integer", "description": "Number of results (max 10)", "default": 5}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    def execute_tool_call(self, tool_call) -> str:
        """Execute a tool call and return results"""
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        try:
            if function_name == "search_github_repos":
                results = self.github.search_repositories(**arguments)
                return json.dumps(results, indent=2)

            elif function_name == "search_reddit_discussions":
                results = self.reddit.search_discussions(**arguments)
                return json.dumps(results, indent=2)

            elif function_name == "search_stackoverflow":
                results = self.stackoverflow.search_questions(**arguments)
                return json.dumps(results, indent=2)

            else:
                return f"Unknown function: {function_name}"

        except Exception as e:
            return f"Error executing {function_name}: {str(e)}"

    async def get_model_response(self, model_name: str, prompt: str) -> ModelResponse:
        """Get response from a single model with research tools"""
        start_time = time.time()
        research_steps = []

        try:
            messages = [{"role": "user", "content": prompt}]
            tools = self.get_research_tools()

            # Try initial request with tools
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                    max_tokens=2000,
                    temperature=0.7
                )
            except Exception as tool_error:
                # If tools fail (404/400), fallback to simple completion
                if "tool" in str(tool_error).lower() or "404" in str(tool_error) or "400" in str(tool_error):
                    print(f"  ⚠️  {model_name} doesn't support tools, using basic completion")
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=2000,
                        temperature=0.7
                    )
                    # Return basic response without research
                    response_time = time.time() - start_time
                    content = response.choices[0].message.content
                    
                    # v0.5: Simple parsing (keep original logic)
                    parts = content.split('\n\n', 2)
                    recommendation = parts[0] if parts else content[:200]
                    reasoning_text = parts[1] if len(parts) > 1 else "Basic response without research"
                    trade_offs = ["No research performed"]
                    confidence_score = 0.7

                    return ModelResponse(
                        model_name=model_name,
                        team="basic",
                        recommendation=recommendation,
                        reasoning=reasoning_text,
                        trade_offs=trade_offs,
                        confidence_score=confidence_score,
                        response_time=response_time,
                        research_steps=[]
                    )
                else:
                    raise tool_error

            # Handle tool calls
            while response.choices[0].message.tool_calls:
                assistant_message = response.choices[0].message
                messages.append(assistant_message)

                for tool_call in assistant_message.tool_calls:
                    print(f"  🔍 {model_name} researching: {tool_call.function.name}")
                    research_steps.append({
                        "function": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    })

                    tool_result = self.execute_tool_call(tool_call)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })

                # Get next response
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                    max_tokens=2000,
                    temperature=0.7
                )

            response_time = time.time() - start_time
            content = response.choices[0].message.content

            # v0.5: Simple parsing (keep original logic)
            parts = content.split('\n\n', 2)
            recommendation = parts[0] if parts else content[:200]
            reasoning_text = parts[1] if len(parts) > 1 else "See full response"
            trade_offs = ["Analysis based on research"]
            confidence_score = 0.8

            return ModelResponse(
                model_name=model_name,
                team="competitor",
                recommendation=recommendation,
                reasoning=reasoning_text,
                trade_offs=trade_offs,
                confidence_score=confidence_score,
                response_time=response_time,
                research_steps=research_steps
            )

        except Exception as e:
            print(f"❌ Error with {model_name}: {str(e)}")
            return ModelResponse(
                model_name=model_name,
                team="competitor",
                recommendation=f"Error: {str(e)}",
                reasoning="Model failed to respond",
                trade_offs=[],
                confidence_score=0.0,
                response_time=time.time() - start_time,
                success=False,  # v0.4: Explicit failure flag
                research_steps=research_steps
            )


    async def run_model_competition(self, decision_type: str, language: Optional[str] = None,
                                   framework: Optional[str] = None, requirements: Optional[str] = None) -> List[ModelResponse]:
        """Run multi-model team competition for Phase 2"""
        prompt = f"""You are an expert software architect competing with other AI models to provide the best architectural guidance. I need your help with an architectural decision.

Decision Type: {decision_type}
Language/Stack: {language or 'Not specified'}
Framework: {framework or 'Not specified'}
Requirements: {requirements or 'None specified'}

Before making your recommendation, please research this topic using the available tools:
- Search GitHub for examples and patterns
- Look at community discussions on Reddit
- Check StackOverflow for technical considerations

{STRICT_OUTPUT_FORMAT}

Focus on practical, production-ready advice. Be confident and specific in your recommendations. Your response will be compared against other AI models."""

        models = Config.get_models()
        responses = []

        print("🏆 Starting model competition...")

        # Build list of (model, task) pairs for cleaner async handling
        jobs = []
        for model_name in models:
            print(f"  🤖 {model_name}")
            task = self.get_model_response(model_name, prompt)
            jobs.append((model_name, task))

        # Execute all models concurrently
        model_responses = await asyncio.gather(*(job[1] for job in jobs), return_exceptions=True)

        # Process results with correct model pairing
        for (model_name, _), response in zip(jobs, model_responses):
            if isinstance(response, Exception):
                # Create error response for failed models
                error_response = ModelResponse(
                    model_name=model_name,
                    team="competitor",  # Simple team name since teams are eliminated
                    recommendation=f"Error: {str(response)}",
                    reasoning="Model failed to respond",
                    trade_offs=[],
                    confidence_score=0.0,
                    response_time=0.0,
                    success=False,  # v0.4: Explicit failure flag
                    research_steps=[]
                )
                responses.append(error_response)
            else:
                # Set simple team name for successful responses
                if isinstance(response, ModelResponse):
                    response.team = "competitor"
                    responses.append(response)

        return responses