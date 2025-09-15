#!/usr/bin/env python3
"""
ArchGuru CLI - Universal AI Architecture Decision Platform
Chapter 1 MVP: Basic CLI with model competition foundation
"""
import argparse
import asyncio
import sys

from ..models.decision import DecisionRequest
from ..agents.pipeline import ModelCompetitionPipeline
from ..core.config import Config


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="archguru",
        description="Universal AI Architecture Decision Platform - Get architectural guidance from competing AI models"
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
    print("🏗️  ArchGuru v0.1 - Universal AI Architecture Decision Platform")
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
        pipeline = ModelCompetitionPipeline()
        result = await pipeline.run(request)

        print(f"\n🏆 Competition Results:")
        print(f"Winning Model: {result.winning_model or 'No clear winner'}")
        print(f"Number of responses: {len(result.model_responses)}")

        print(f"\n📋 Consensus Recommendation:")
        print(result.consensus_recommendation or "No consensus reached")

        if args.verbose:
            print(f"\n📊 Detailed Results:")
            for response in result.model_responses:
                print(f"\n{response.team.upper()} Team ({response.model_name}):")
                print(f"  Response time: {response.response_time:.2f}s")
                print(f"  Recommendation: {response.recommendation[:100]}...")

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