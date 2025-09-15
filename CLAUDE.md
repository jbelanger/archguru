# ArchGuru: Universal AI Architecture Decision Platform

## Project Overview

Building a **Universal AI Architecture Decision Platform** that uses multiple AI model teams to compete and provide the best architectural guidance for any technical decision. Starting with project structures, expanding to databases, deployment strategies, API designs, and any architectural choice. This is a learning project to master LangGraph through practical MVP development.

## What We're Building

A CLI platform (`archguru`) where AI model teams compete:

1. **OpenAI Team** - GPT models running research→community→generation pipeline
2. **Claude Team** - Claude models running research→community→generation pipeline
3. **Llama Team** - Llama models running research→community→generation pipeline
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

## Development Approach: Iterative MVP Releases

**Strategic Product Development** with "Real Product Evolution" methodology:

- Each release adds features to the same codebase in `src/`
- Single `archguru` CLI that grows more sophisticated each release
- No separate prototype files - everything builds on the real product
- Version tags (v0.1 → v1.4) track progress through git commits
- Professional development through actual production implementation

## Current Status

- ✅ Project setup (uv + Python 3.13)
- ✅ ADR v3.0 created (`docs/adr-001-architecture.md`) - Universal Architecture Decision Platform
- ✅ Product roadmap created (`docs/roadmap.md`) - Strategic development plan with 14 releases
- ✅ Architecture pivoted from individual agents to model team competition
- ✅ OpenRouter dependencies added (openai, python-dotenv)
- ✅ Learning chapters 1-7 completed (foundational LangGraph concepts)
- ⏳ Ready to start building real MVP with OpenRouter model competition

## Development Commands

```bash
# Run the main CLI (the real product)
uv run archguru --help
uv run archguru --type project-structure --language python --framework web

# Development/testing
uv run python -m src.archguru.cli.main --help

# Add dependencies as needed
uv add package-name

# Add dev dependencies
uv add --dev package-name

# Check Python version
uv run python --version  # Should show 3.13.x

# Tag versions as we build
git tag v0.1  # After Chapter 1
git tag v0.2  # After Chapter 2, etc.
```

## File Structure

```
archguru/
├── .python-version                    # Python 3.13
├── pyproject.toml                     # Project config with CLI entry point
├── .env.example                       # OpenRouter API key template
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
- **OpenRouter** for multi-model access (GPT, Claude, Llama, etc.)
- **SQLite** for model response caching and performance tracking
- **Rich/Typer** for production CLI interface
- **FastAPI** for web dashboard and API endpoints
- **GitHub/Reddit/StackOverflow APIs** for real data sources

## Remember for Future Sessions

- **Single codebase evolution**: All chapters add to `src/` - no separate files
- **Real product development**: Every commit makes `archguru` CLI more powerful
- **Version tagging**: Track progress with git tags (v0.1, v0.2, etc.)
- **Production mindset**: Build something people will pay for, not just a tutorial
- **Model competition focus**: Core value is comparing OpenRouter models
- **Iterative learning**: Learn LangGraph through real feature development

---

## Current Progress Summary

- **Architecture Evolution**: Individual agents → Model team competition
- **Development Approach**: Chapter files → Single iterative codebase in `src/`
- **Product Vision**: Project structure tool → Universal architecture decision platform
- **Learning Method**: Tutorial examples → Real production development
- **Value Proposition**: LangGraph learning → Valuable product + deep technical skills

_Last updated: 2025-09-15_
_Current status: Ready to start iterative development of `archguru` CLI in `src/`_
