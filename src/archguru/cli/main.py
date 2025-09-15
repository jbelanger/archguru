#!/usr/bin/env python3
"""
ArchGuru CLI - Universal AI Architecture Decision Platform
v0.1: Single model with autonomous research
"""
import argparse
import asyncio
import sys

from ..models.decision import DecisionRequest
from ..agents.pipeline import ModelResearchPipeline
from ..core.config import Config


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="archguru",
        description="Universal AI Architecture Decision Platform - Get architectural guidance with autonomous AI research"
    )

    parser.add_argument(
        "--type",
        choices=["project-structure", "database", "deployment", "api-design"],
        required=True,
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
        help="Enable verbose output"
    )

    return parser


async def run_decision(args) -> int:
    """Run the architectural decision process"""
    print("🏗️  ArchGuru v0.1 - AI Architecture Research Platform")
    print("=" * 60)

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
        pipeline = ModelResearchPipeline()
        result = await pipeline.run(request)

        print(f"\n🔬 Research Results:")
        if result.model_responses:
            response = result.model_responses[0]
            print(f"Model: {response.model_name}")
            print(f"Research steps: {len(response.research_steps)}")
            print(f"Response time: {response.response_time:.2f}s")

        print(f"\n📋 Recommendation:")
        print(result.consensus_recommendation or "No recommendation generated")

        if args.verbose and result.model_responses:
            response = result.model_responses[0]
            print(f"\n📊 Research Details:")
            for i, step in enumerate(response.research_steps, 1):
                print(f"  {i}. {step['function']}")
            print(f"\nReasoning: {response.reasoning}")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    return asyncio.run(run_decision(args))


if __name__ == "__main__":
    sys.exit(main())