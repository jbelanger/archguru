#!/usr/bin/env python3
"""
ArchGuru CLI - Universal AI Architecture Decision Platform
Phase 2: Multi-model team competition with cross-model debate
"""
import argparse
import asyncio
import sys
import time

from ..models.decision import DecisionRequest
from ..agents.pipeline import ModelCompetitionPipeline
from ..core.config import Config
from ..storage.repo import persist_pipeline_result, ArchGuruRepo


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="archguru",
        description="Universal AI Architecture Decision Platform - Get architectural guidance from competing AI models"
    )

    parser.add_argument(
        "--type",
        choices=["project-structure", "database", "deployment", "api-design"],
        help="Type of architectural decision to make"
    )

    parser.add_argument(
        "--language",
        help="Programming language or technology stack"
    )

    parser.add_argument(
        "--framework",
        help="Framework or specific technology preference"
    )

    parser.add_argument(
        "--requirements",
        help="Additional requirements or constraints"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output including model details and arbiter evaluation"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show usage statistics from previous runs"
    )

    return parser


async def run_decision(args) -> int:
    """Run the architectural decision process"""
    print("🏗️  ArchGuru Phase 2 - AI Architecture Model Competition Platform")
    print("=" * 70)

    print(f"Decision Type: {args.type}")
    if args.language:
        print(f"Language/Stack: {args.language}")
    if args.framework:
        print(f"Framework: {args.framework}")
    if args.requirements:
        print(f"Requirements: {args.requirements}")

    if not Config.validate():
        print("\n❌ Configuration error - please check your .env file")
        return 1

    request = DecisionRequest(
        decision_type=args.type,
        language=args.language,
        framework=args.framework,
        requirements=args.requirements
    )

    try:
        start_time = time.time()
        pipeline = ModelCompetitionPipeline()
        result = await pipeline.run(request)
        total_time = time.time() - start_time

        # v0.3 Persistence hook - persist result after pipeline completion
        try:
            # Convert ModelResponse objects to dict format for persistence
            model_responses_data = []
            for response in result.model_responses:
                model_data = {
                    'model': response.model_name,
                    'team': response.team,
                    'recommendation': response.recommendation,
                    'reasoning': response.reasoning,
                    'trade_offs': response.trade_offs,
                    'confidence_score': response.confidence_score,
                    'response_time_sec': response.response_time,
                    'success': not response.recommendation.startswith("Error:"),
                    'error': response.recommendation if response.recommendation.startswith("Error:") else None,
                    'tool_calls': [
                        {
                            'function': step.get('function', ''),
                            'arguments': step.get('arguments', {}),
                            'result_excerpt': str(step.get('result', ''))[:500]
                        }
                        for step in (response.research_steps or [])
                    ]
                }
                model_responses_data.append(model_data)

            run_id = persist_pipeline_result(
                decision_type=args.type,
                language=args.language,
                framework=args.framework,
                requirements=args.requirements,
                model_responses=model_responses_data,
                arbiter_model=Config.get_arbiter_model(),
                consensus_recommendation=result.consensus_recommendation,
                debate_summary=result.debate_summary,
                total_time_sec=total_time
            )
            print(f"\n💾 Run saved: {run_id}")
        except Exception as e:
            print(f"\n⚠️  Warning: Failed to save run data: {str(e)}")

        await _display_competition_results(result, args.verbose)
        return 0

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if args.verbose:
            import traceback
            print(f"\nFull traceback:\n{traceback.format_exc()}")
        return 1


async def _display_competition_results(result, verbose: bool = False):
    """Display results from model competition"""
    def _preview(text: str, lines: int = 12) -> str:
        return "\n".join((text or "").strip().splitlines()[:lines])

    responses = result.model_responses or []
    successful_responses = [r for r in responses if not r.recommendation.startswith("Error:")]

    print(f"\n🏆 Competition Results:")
    print(f"Models competed: {len(responses)}")
    print(f"Successful responses: {len(successful_responses)}")

    if result.winning_model:
        print(f"🥇 Winner: {result.winning_model}")

    print(f"\n📋 Final Recommendation:")
    print(result.consensus_recommendation or "No consensus reached")

    if verbose:
        # v0.2 requirement: side-by-side recommendations + research approach
        print(f"\n🧪 Side-by-side recommendations:")
        for r in responses:
            if r.recommendation.startswith("Error:"):
                continue
            funcs = [step.get("function", "") for step in (r.research_steps or [])]
            print(f"\n--- {r.model_name} ({r.team}) — {r.response_time:.2f}s ---")
            print(_preview(r.recommendation, 12))
            if funcs:
                print(f"🔎 Research approach: {', '.join(funcs)}")

        print(f"\n🎯 Individual Model Performance:")
        for i, response in enumerate(responses, 1):
            status = "✅" if not response.recommendation.startswith("Error:") else "❌"
            print(f"  {i}. {status} {response.model_name} ({response.team})")
            print(f"     Research: {len(response.research_steps)} steps")
            print(f"     Time: {response.response_time:.2f}s")
            if response.recommendation.startswith("Error:"):
                print(f"     Error: {response.recommendation}")
            print()

        if result.debate_summary:
            print(f"🥊 Debate Summary:")
            print(result.debate_summary)




async def show_stats() -> int:
    """Show usage statistics from previous runs"""
    try:
        repo = ArchGuruRepo()
        stats = repo.get_stats()

        print("📊 ArchGuru Usage Statistics")
        print("=" * 40)
        print(f"Total Runs: {stats['total_runs']}")
        print(f"Average Latency: {stats['avg_latency_sec']}s")
        print(f"Recent Runs (7 days): {stats['recent_runs_7d']}")

        if stats['decision_types']:
            print(f"\n🎯 Decision Types:")
            for dt in stats['decision_types'][:5]:  # Top 5
                print(f"  {dt['label']}: {dt['count']} runs")

        if stats['model_usage']:
            print(f"\n🤖 Model Usage:")
            for model in stats['model_usage'][:5]:  # Top 5
                print(f"  {model['name']}: {model['responses']} responses")

        return 0

    except Exception as e:
        print(f"❌ Error retrieving stats: {str(e)}")
        return 1


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    # Handle stats command
    if args.stats:
        return asyncio.run(show_stats())

    # Validate required args for decision making
    if not args.type:
        parser.error("--type is required for decision making")

    return asyncio.run(run_decision(args))


if __name__ == "__main__":
    sys.exit(main())