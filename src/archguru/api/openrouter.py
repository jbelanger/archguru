"""
OpenRouter API integration with function calling support
Handles LLM requests with research tool access
"""
import asyncio
import json
from typing import List, Optional
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from ..core.config import Config
from ..core.constants import (
    MODEL_MAX_TOKENS, MODEL_TEMPERATURE,
    DEFAULT_TOOL_RESULTS, MAX_TOOL_RESULTS
)
from ..core.response_parser import ResponseParser
from ..models.decision import ModelResponse
from .github import GitHubClient
from .reddit import RedditClient
from .stackoverflow import StackOverflowClient
import time

# v0.5: Strict output format constant for team generation
STRICT_OUTPUT_FORMAT = """OUTPUT FORMAT (STRICT):
Final Recommendation: <one concise sentence>

Reasoning:
- <3-6 bullets, ≤12 words each>

Trade-offs:
- <2-5 bullets, name the axis>

Implementation Steps:
- <3-7 bullets, concrete>

Evidence:
- <0-5 GitHub links or repo names only>"""


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
        self.parser = ResponseParser()

    def get_research_tools(self) -> List[ChatCompletionToolParam]:
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
                            "limit": {"type": "integer", "description": f"Number of results (max {MAX_TOOL_RESULTS})", "default": DEFAULT_TOOL_RESULTS}
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
                            "limit": {"type": "integer", "description": f"Number of results (max {MAX_TOOL_RESULTS})", "default": DEFAULT_TOOL_RESULTS}
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
                            "limit": {"type": "integer", "description": f"Number of results (max {MAX_TOOL_RESULTS})", "default": DEFAULT_TOOL_RESULTS}
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

        # Clamp limit to MAX_TOOL_RESULTS
        if 'limit' in arguments:
            arguments['limit'] = min(arguments['limit'], MAX_TOOL_RESULTS)

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
            messages: List[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]
            tools = self.get_research_tools()

            # Try initial request with tools using constants
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                    max_tokens=MODEL_MAX_TOKENS,
                    temperature=MODEL_TEMPERATURE
                )
            except Exception as tool_error:
                # If tools fail (404/400), fallback to simple completion
                if "tool" in str(tool_error).lower() or "404" in str(tool_error) or "400" in str(tool_error):
                    print(f"  ⚠️  {model_name} doesn't support tools, using basic completion")
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=MODEL_MAX_TOKENS,
                        temperature=MODEL_TEMPERATURE
                    )
                    # Return basic response without research
                    response_time = time.time() - start_time
                    content = response.choices[0].message.content
                    
                    # Use parser for consistency
                    parsed = self.parser.parse_model_response(content or "")
                    
                    return ModelResponse(
                        model_name=model_name,
                        team="basic",
                        recommendation=parsed.recommendation,
                        reasoning=parsed.reasoning,
                        trade_offs=parsed.trade_offs if parsed.trade_offs else ["No research performed"],
                        confidence_score=0.7,
                        response_time=response_time,
                        research_steps=[]
                    )
                else:
                    raise tool_error

            # Handle tool calls
            while response.choices[0].message.tool_calls:
                assistant_message = response.choices[0].message
                # Create proper assistant message dict
                assistant_msg: ChatCompletionMessageParam = {
                    "role": "assistant",
                    "content": assistant_message.content or ""
                }
                # Add tool_calls if present, converting to proper format
                if assistant_message.tool_calls:
                    tool_calls_param = []
                    for tc in assistant_message.tool_calls:
                        function_obj = getattr(tc, 'function', None)
                        if function_obj:
                            tool_calls_param.append({
                                "id": getattr(tc, 'id', ''),
                                "type": "function",
                                "function": {
                                    "name": getattr(function_obj, 'name', 'unknown'),
                                    "arguments": getattr(function_obj, 'arguments', '{}')
                                }
                            })
                    assistant_msg["tool_calls"] = tool_calls_param
                messages.append(assistant_msg)

                for tool_call in (assistant_message.tool_calls or []):
                    # Use getattr for safe attribute access
                    function_obj = getattr(tool_call, 'function', None)
                    if function_obj:
                        function_name = getattr(function_obj, 'name', 'unknown')
                        function_args = getattr(function_obj, 'arguments', '{}')

                        print(f"  🔍 {model_name} researching: {function_name}")
                        research_steps.append({
                            "function": function_name,
                            "arguments": function_args
                        })

                        try:
                            tool_result = self.execute_tool_call(tool_call)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_result
                            })
                        except Exception as tool_error:
                            print(f"  ❌ {model_name} tool execution failed: {tool_error}")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Error: {str(tool_error)}"
                            })

                # Get next response
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                    max_tokens=MODEL_MAX_TOKENS,
                    temperature=MODEL_TEMPERATURE
                )

            response_time = time.time() - start_time
            content = response.choices[0].message.content

            # Track if model skipped research tools (for performance metrics)
            skipped_research = len(research_steps) == 0 and "research" in prompt.lower()
            if skipped_research:
                # Check response object for token usage info
                response_info = ""
                if hasattr(response, 'usage'):
                    usage = getattr(response, 'usage', None)
                    if usage:
                        prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                        completion_tokens = getattr(usage, 'completion_tokens', 0)
                        response_info = f" (tokens: {prompt_tokens}→{completion_tokens})"

                print(f"  ⚠️  {model_name} skipped research tools (time: {response_time:.2f}s){response_info}")
                if content:
                    print(f"      Response preview: {content[:150].strip() if content else ''}...")
                # Continue to allow the response but mark it for lower scoring

            # Use parser for consistency
            if not content or content.strip() == "":
                print(f"  ⚠️  {model_name} returned empty content (research steps: {len(research_steps)}, time: {response_time:.2f}s)")
                return ModelResponse(
                    model_name=model_name,
                    team="competitor",
                    recommendation="Error: Empty response",
                    reasoning=f"Model returned empty content after {len(research_steps)} research steps in {response_time:.2f}s",
                    trade_offs=[],
                    confidence_score=0.0,
                    response_time=response_time,
                    success=False,
                    research_steps=research_steps
                )

            # Parse response using ResponseParser
            parsed = self.parser.parse_model_response(content)

            # Adjust confidence score based on research behavior
            base_confidence = 0.8
            if skipped_research:
                base_confidence = 0.6  # Lower confidence for models that skip research

            return ModelResponse(
                model_name=model_name,
                team="competitor",
                recommendation=parsed.recommendation,
                reasoning=parsed.reasoning,
                trade_offs=parsed.trade_offs if parsed.trade_offs else (["Analysis based on research"] if not skipped_research else ["Analysis without research"]),
                confidence_score=base_confidence,
                response_time=response_time,
                research_steps=research_steps,
                skipped_research=skipped_research
            )

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error with {model_name}: {error_msg}")
            print(f"  🔍 Exception type: {type(e).__name__}")

            # Extract detailed error information
            detailed_error = error_msg

            # Check for HTTP response attributes more carefully
            try:
                if hasattr(e, 'response') and getattr(e, 'response', None):
                    response_obj = getattr(e, 'response')
                    status_code = getattr(response_obj, 'status_code', 'unknown')
                    response_text = getattr(response_obj, 'text', 'unknown')[:200]
                    print(f"  🔍 HTTP status: {status_code}")
                    print(f"  🔍 Response text: {response_text}")
                    detailed_error = f"{error_msg} (HTTP {status_code}: {response_text})"

                # Check for OpenAI-specific error details
                if hasattr(e, 'body') and getattr(e, 'body', None):
                    body = getattr(e, 'body')
                    if isinstance(body, dict):
                        error_info = body.get('error', {})
                        if isinstance(error_info, dict):
                            error_type = error_info.get('type', '')
                            error_code = error_info.get('code', '')
                            error_message = error_info.get('message', '')
                            if error_message:
                                print(f"  🔍 API error: {error_type} ({error_code}): {error_message}")
                                detailed_error = f"{error_type}: {error_message}"

                # Check for rate limiting or model availability issues
                if "rate" in error_msg.lower():
                    detailed_error = f"Rate limited: {error_msg}"
                elif "model" in error_msg.lower() and ("not found" in error_msg.lower() or "unavailable" in error_msg.lower()):
                    detailed_error = f"Model unavailable: {error_msg}"
                elif "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
                    detailed_error = f"Authentication error: {error_msg}"

            except Exception:
                pass  # Don't fail on logging errors
            return ModelResponse(
                model_name=model_name,
                team="competitor",
                recommendation=f"Error: {detailed_error}",
                reasoning=f"Model failed to respond: {detailed_error}",
                trade_offs=[],
                confidence_score=0.0,
                response_time=time.time() - start_time,
                success=False,
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
                    team="competitor",
                    recommendation=f"Error: {str(response)}",
                    reasoning="Model failed to respond",
                    trade_offs=[],
                    confidence_score=0.0,
                    response_time=0.0,
                    success=False,
                    research_steps=[]
                )
                responses.append(error_response)
            else:
                # Set simple team name for successful responses
                if isinstance(response, ModelResponse):
                    response.team = "competitor"
                    responses.append(response)

        return responses