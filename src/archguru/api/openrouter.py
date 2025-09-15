"""
OpenRouter API integration with function calling support
Handles LLM requests with research tool access
"""
import asyncio
import json
from typing import List, Dict, Any
from openai import OpenAI
from ..core.config import Config
from ..models.decision import ModelResponse
from .github import GitHubClient
from .reddit import RedditClient
from .stackoverflow import StackOverflowClient
import time


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

            # Initial request with tools
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                max_tokens=2000,
                temperature=0.7
            )

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

            # Parse the final response
            parts = content.split('\n\n', 2)
            recommendation = parts[0] if parts else content[:200]
            reasoning = parts[1] if len(parts) > 1 else "See full response"

            return ModelResponse(
                model_name=model_name,
                team="research",
                recommendation=recommendation,
                reasoning=reasoning,
                trade_offs=["Analysis based on research"],
                confidence_score=0.8,
                response_time=response_time,
                research_steps=research_steps
            )

        except Exception as e:
            print(f"❌ Error with {model_name}: {str(e)}")
            return ModelResponse(
                model_name=model_name,
                team="research",
                recommendation=f"Error: {str(e)}",
                reasoning="Model failed to respond",
                trade_offs=[],
                confidence_score=0.0,
                response_time=time.time() - start_time,
                research_steps=research_steps
            )

    async def run_single_model_research(self, decision_type: str, language: str = None,
                                       framework: str = None, requirements: str = None) -> ModelResponse:
        """Run single model with autonomous research for v0.1"""
        prompt = f"""You are an expert software architect. I need your help with an architectural decision.

Decision Type: {decision_type}
Language/Stack: {language or 'Not specified'}
Framework: {framework or 'Not specified'}
Requirements: {requirements or 'None specified'}

Before making your recommendation, please research this topic using the available tools:
- Search GitHub for examples and patterns
- Look at community discussions on Reddit
- Check StackOverflow for technical considerations

After your research, provide:
1. Your specific recommendation
2. Detailed reasoning based on your research findings
3. Trade-offs and alternatives you discovered
4. Implementation considerations

Focus on practical, production-ready advice based on real-world evidence."""

        print("🤖 Starting single model research...")
        print(f"  📊 Using GPT-4o with autonomous research...")

        # Use GPT-4o for v0.1
        response = await self.get_model_response("openai/gpt-4o", prompt)
        return response