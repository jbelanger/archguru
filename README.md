# ArchGuru <�

> **Universal AI Architecture Decision Platform** - Get the best architectural guidance from competing AI models

[![Development Status](https://img.shields.io/badge/Status-Under%20Development-yellow)](https://github.com/jbelanger/archguru)
[![Python](https://img.shields.io/badge/Python-3.13+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**� This project is currently under active development. The features described below represent the final vision - most are not yet implemented.**

## What is ArchGuru?

ArchGuru is a revolutionary CLI platform that uses **competing AI model teams** to provide expert architectural guidance for any technical decision. Instead of relying on a single AI model, ArchGuru pits multiple AI teams against each other to give you the most comprehensive and reliable architectural recommendations.

### The Problem We Solve

Making architectural decisions is hard:
- **Information Overload**: Too many options, frameworks, and best practices to evaluate
- **Biased Perspectives**: Single sources often have inherent biases toward specific technologies
- **Context Missing**: Generic advice doesn't account for your specific requirements and constraints
- **Expensive Consulting**: Expert architectural guidance costs thousands of dollars

### The ArchGuru Solution

**Model-vs-Model Competition** where AI teams compete to provide the best architectural guidance:

```bash
# Get project structure recommendations
archguru --type project-structure --language python --framework web

# Database architecture decisions
archguru --type database --app "real-time chat" --scale "10k users"

# Deployment strategy analysis
archguru --type deployment --app "e-commerce" --traffic "high" --budget "startup"

# API design recommendations
archguru --type api --pattern "microservices" --auth "oauth2"
```

## How It Works

### > Competing AI Model Teams
- **OpenAI Team**: GPT models analyze your requirements
- **Claude Team**: Anthropic's Claude models provide alternative perspectives
- **Llama Team**: Meta's Llama models offer additional insights
- **Cross-Model Debates**: Models critique and refine each other's recommendations

### =� Three-Phase Analysis Pipeline
Each model team runs a comprehensive analysis:

1. **Research Phase**: Analyzes GitHub repos, documentation, benchmarks
2. **Community Phase**: Evaluates Reddit discussions, StackOverflow trends, developer sentiment
3. **Generation Phase**: Synthesizes findings into actionable recommendations

### <� Performance Analytics
- Track which models excel at different decision types
- Learn from user feedback and real-world outcomes
- Continuously improve recommendation quality

## Supported Decision Types

| Decision Type | Description | Status |
|---------------|-------------|--------|
| **Project Structure** | File organization, build systems, dependency management | =� In Development |
| **Database Architecture** | SQL vs NoSQL, specific database selection, schema design | =� Planned |
| **Deployment Strategy** | Cloud vs on-premise, containers vs serverless | =� Planned |
| **API Design** | REST vs GraphQL vs gRPC, versioning strategies | =� Planned |
| **Authentication** | OAuth vs JWT vs sessions, identity providers | =� Planned |
| **Frontend Architecture** | SPA vs MPA, framework selection, state management | =� Planned |
| **Testing Strategy** | Unit vs integration vs E2E, framework selection | =� Planned |
| **Monitoring & Observability** | Logging, metrics, tracing solutions | =� Planned |

*More decision types added continuously based on community needs*

## Key Features

### <� **Multi-Model Competition**
Get perspectives from multiple AI models, not just one. Compare recommendations and see where models agree or disagree.

### = **Cross-Model Debates**
Models analyze and critique each other's recommendations, leading to more refined and thoughtful advice.

### =� **Performance Analytics**
Track which models provide better guidance for different types of decisions. Learn which AI excels at databases vs deployment vs API design.

### � **Intelligent Caching**
Fast responses through smart caching. First run takes ~3 minutes, cached results in ~30 seconds.

### <� **Rich CLI Experience**
Beautiful terminal output with tables, colors, and export options (Markdown, PDF, JSON).

### < **Universal Decision Engine**
One platform for all architectural decisions - from project structure to deployment strategy.

## Installation

```bash
# Install with uv (recommended)
uv add archguru

# Or with pip
pip install archguru

# Set up your OpenRouter API key
export OPENROUTER_API_KEY="your_key_here"

# Run your first architectural decision
archguru --type project-structure --language python --framework web
```

## Quick Start

```bash
# Get help
archguru --help

# Basic project structure decision
archguru --type project-structure \
         --language python \
         --framework web \
         --scale startup

# Database architecture with specific requirements
archguru --type database \
         --app "real-time chat" \
         --users "10k concurrent" \
         --consistency "eventual" \
         --budget "moderate"

# Compare deployment strategies
archguru --type deployment \
         --app "microservices" \
         --traffic "variable" \
         --team-size "5-10" \
         --expertise "intermediate"
```

## Development Status

**Current Version: v0.1-dev (MVP in Development)**

###  Completed
- Project setup and structure
- CLI framework with argument parsing
- OpenRouter API integration foundation
- Documentation and architecture design

### =� In Progress (MVP v0.1)
- Single model decision engine
- Basic project structure recommendations
- CLI output formatting

### =� Upcoming Milestones
- **v0.2**: Multi-phase pipeline (research � community � generation)
- **v0.3**: Two-model comparison (GPT vs Claude)
- **v0.4**: Smart model routing and selection
- **v0.5**: Production data integration (GitHub, Reddit APIs)
- **v0.6**: Three-model competition with scoring
- **v0.7**: Cross-model debates and critiques
- **v1.0**: Production caching and performance optimization

[View full roadmap →](docs/roadmap.md)

## Architecture

ArchGuru is built with a **Universal Decision Engine** that can handle any architectural decision type through:

- **LangGraph** for orchestrating model team pipelines and debates
- **OpenRouter** as unified interface for multiple AI model providers
- **SQLite** for caching model responses and performance analytics
- **Rich/Typer** for production-quality CLI experience
- **Plugin Architecture** for easily adding new decision types

[Read the full Architecture Decision Record �](docs/adr-001-architecture.md)

## Contributing

This project is under active development! We welcome:

- **Feature requests** for new decision types
- **Bug reports** and **feedback** on recommendations
- **Model performance insights** from real-world usage
- **Documentation improvements**
- **Code contributions** (see [development guide](CLAUDE.md))

## License

MIT License - see [LICENSE](LICENSE) for details.

## Learn More

- =� **[Development Guide](CLAUDE.md)** - Full project context and development approach
- <� **[Architecture Decision Records](docs/adr-001-architecture.md)** - Technical architecture and design decisions
- 🗺️ **[Product Roadmap](docs/roadmap.md)** - Strategic development plan and release milestones
- =� **[GitHub Repository](https://github.com/jbelanger/archguru)** - Source code and issue tracking

---

**Built with d as a learning project to master LangGraph while creating something valuable for developers worldwide.**

*Follow the development progress and get early access to new features!*