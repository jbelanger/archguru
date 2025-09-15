# ArchGuru: Universal AI Architecture Decision Platform

## Project Overview

Building a **Universal AI Architecture Decision Platform** that uses multiple AI model teams to compete and provide the best architectural guidance for any technical decision. Starting with project structures, expanding to databases, deployment strategies, API designs, and any architectural choice. This is a learning project to master LangGraph through practical MVP development.

## What We're Building

A CLI platform (`archguru`) where AI model teams compete:

1. **Team A** - Configurable models (default: OpenAI) running research→community→generation pipeline
2. **Team B** - Configurable models (default: Claude) running research→community→generation pipeline
3. **Team C** - Configurable models (default: Llama) running research→community→generation pipeline
4. **Cross-Model Debates** - Models analyze and critique each other's recommendations
5. **Performance Analytics** - Track which models excel at which decision types

**Key Innovation**: Model-vs-Model competition where we evaluate which OpenRouter models provide the best architectural guidance across different domains.

## Architecture Decisions (from ADR v3.0)

- **Cross-Model Team Competition** with multiple OpenRouter models
- **Universal Decision Engine** handling any architectural decision type
- **Model Evaluation Framework** to rate which models excel at different decisions
- **LangGraph orchestration** for model team pipelines and cross-model debates
- **SQLite caching** across all decision types and models
- **OpenRouter API** as unified interface for model competition
- **Challenge/refinement loop** - users can contest recommendations
- **Rich CLI + Web API** with model performance analytics

## User Requirements (Updated)

1. **Scope**: Any architectural decision (project structure, databases, deployment, APIs, etc.)
2. **Output**: Architectural recommendations with explanations and trade-offs
3. **Model Competition**: Multiple OpenRouter models competing for best recommendations
4. **Performance Analytics**: Show which models excel at different decision types
5. **Data Sources**: GitHub, Reddit, StackOverflow, documentation, benchmarks
6. **Performance**: <3 minutes initial (multiple models), <45s cached
7. **Storage**: SQLite for model response caching and performance tracking
8. **Cross-Model Debates**: Models critique and respond to each other's recommendations
9. **Interface**: Rich CLI + Web dashboard + API endpoints
10. **Cost Optimization**: Balance model quality with API costs across multiple models

## Development Approach: Clean State Phases

**Clean Development Methodology** with "Phase Reset" approach:

- Each phase is a complete rewrite/rebuild of the codebase in `src/`
- NO legacy compatibility or backward support between phases
- Each phase starts fresh with current requirements only
- No "v0.1 compatibility mode" or legacy code paths
- Clean, focused implementation for each phase's goals
- Version tags (Phase 1 → Phase 2 → Phase 3) track major rebuilds
- Professional development through clean architecture at each phase

## Current Status - Phase 2

- ✅ Project setup (uv + Python 3.13)
- ✅ ADR v3.0 created (`docs/adr-001-architecture.md`) - Universal Architecture Decision Platform
- ✅ Product roadmap created (`docs/roadmap.md`) - Strategic development plan
- ✅ **Phase 2 Complete**: Multi-model team competition with cross-model debate
- ✅ OpenRouter dependencies added (openai, python-dotenv, langgraph)
- ✅ LangGraph pipeline for model competition
- ✅ Cross-model debate engine with arbiter evaluation
- ✅ Clean CLI implementation (no legacy compatibility)

## Development Commands

```bash
# Run the Phase 2 CLI (model competition)
uv run archguru --help
uv run archguru --type project-structure --language python --framework web
uv run archguru --type database --language python --verbose

# Development/testing
uv run python -m src.archguru.cli.main --help

# Add dependencies as needed
uv add package-name

# Add dev dependencies
uv add --dev package-name

# Check Python version
uv run python --version  # Should show 3.13.x

# Tag phases as we complete them
git tag Phase-1  # Single model research
git tag Phase-2  # Multi-model competition
```

## Environment Configuration

Configure models via environment variables in `.env`:

```bash
# Required
OPENROUTER_API_KEY=your_key_here

# Optional: Custom model teams (defaults to OpenAI, Claude, Llama)
ARCHGURU_TEAM_A_MODELS=openai/gpt-4o,openai/gpt-4o-mini
ARCHGURU_TEAM_B_MODELS=x-ai/grok-beta,anthropic/claude-3-haiku
ARCHGURU_TEAM_C_MODELS=deepseek/deepseek-chat,meta-llama/llama-3.1-8b-instruct

# Optional: Final arbiter model (default: openai/gpt-4o)
ARCHGURU_ARBITER_MODEL=anthropic/claude-3.5-sonnet

# Examples of other model combinations:
# ARCHGURU_TEAM_A_MODELS=google/gemini-pro-1.5,google/gemini-flash-1.5
# ARCHGURU_TEAM_B_MODELS=mistralai/mistral-large,mistralai/mistral-medium
# ARCHGURU_TEAM_C_MODELS=qwen/qwen-2.5-72b-instruct,alibaba/qwen-turbo
```

## File Structure

```
archguru/
├── .python-version                    # Python 3.13
├── pyproject.toml                     # Project config with CLI entry point
├── .env.example                       # OpenRouter API key and model configuration template
├── src/archguru/                      # THE REAL PRODUCT (iteratively built)
│   ├── __init__.py
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py                    # CLI entry point (archguru command)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                  # OpenRouter config
│   │   └── state.py                   # LangGraph state management
│   ├── models/
│   │   ├── __init__.py
│   │   └── decision.py                # Decision data models
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── model_teams.py             # Model team implementations
│   │   └── debate.py                  # Cross-model debate logic
│   ├── api/
│   │   ├── __init__.py
│   │   ├── openrouter.py              # OpenRouter client
│   │   └── external.py                # GitHub, Reddit APIs
│   └── utils/
│       ├── __init__.py
│       ├── cache.py                   # SQLite caching
│       └── evaluation.py             # Model performance tracking
├── docs/
│   ├── adr-001-architecture.md        # Updated to v3.0
│   └── roadmap.md                     # Strategic product development plan
└── CLAUDE.md                          # This file
```

## Next Steps

Start building **MVP v0.1** directly in `src/`:

1. Create CLI entry point in `src/archguru/cli/main.py`
2. Set up OpenRouter integration in `src/archguru/api/openrouter.py`
3. Build LangGraph pipeline in `src/archguru/agents/model_teams.py`
4. Configure `pyproject.toml` with CLI script entry point
5. Test: `uv run archguru --type project-structure --language python`

**No more chapter files - everything builds the real `archguru` CLI from now on.**

## Key Commands for Future Sessions

When resuming work:

1. `cd /Users/joel/Dev/archguru`
2. Read this file to understand context
3. Check `docs/roadmap.md` for current release milestone
4. Continue from where we left off

## Learning Goals

- Master LangGraph through building real MVP products
- Understand model competition and cross-model debate mechanics
- Build production-ready Universal Architecture Decision Platform
- Learn model evaluation and performance analytics
- Create something people will pay for while learning LangGraph

## Technical Stack

- **Python 3.13** with uv package manager
- **LangGraph** for model team orchestration and cross-model debates
- **OpenRouter** for multi-model access (configurable: GPT, Claude, Llama, Grok, DeepSeek, etc.)
- **SQLite** for model response caching and performance tracking
- **Rich/Typer** for production CLI interface
- **FastAPI** for web dashboard and API endpoints
- **GitHub/Reddit/StackOverflow APIs** for real data sources

## Remember for Future Sessions

- **Clean phase development**: Each phase is a fresh rebuild, no legacy compatibility
- **Real product development**: Every phase makes `archguru` CLI more powerful
- **Phase tagging**: Track major rebuilds with git tags (Phase-1, Phase-2, etc.)
- **Production mindset**: Build something people will pay for, not just a tutorial
- **Model competition focus**: Core value is comparing configurable OpenRouter models
- **Learning through building**: Learn LangGraph through real feature development

---

## Current Progress Summary

- **Architecture**: Multi-model team competition with cross-model debate
- **Development Approach**: Clean phase rebuilds in `src/` (no legacy compatibility)
- **Product Vision**: Universal architecture decision platform with model competition
- **Learning Method**: Real production development using LangGraph
- **Value Proposition**: Valuable product + deep LangGraph technical skills

_Last updated: 2025-09-15_
_Current status: **Phase 2 Complete** - Multi-model competition platform ready for testing_
