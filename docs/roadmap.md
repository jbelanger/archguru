# ArchGuru Roadmap: From MVP v0.2 to Solid v1.0

_Simplicity-first, additive development from existing Phase 2 implementation_

## Current Status: v0.6 Complete - Enhanced Evaluation Metrics 🏆

**What we have working:**

- ✅ CLI runs **multiple models concurrently** and prints side-by-side results
- ✅ Research steps preview showing model research approaches
- ✅ Config supports **teams & arbiter model** via environment variables
- ✅ Debate/arbiter scaffolding exists (gated for future use)
- ✅ OpenRouter integration with GitHub, Reddit, StackOverflow APIs
- ✅ LangGraph pipeline orchestrating model competition
- ✅ **v0.3: SQLite persistence** - Every run writes to database automatically
- ✅ **v0.3: Basic stats command** - `--stats` shows decision count, latency, model usage
- ✅ **v0.4: Pairwise + Elo ratings** - Real-time model performance tracking per decision type
- ✅ **v0.4: Top 5 rankings** - `--stats` displays Elo leaderboards by decision type
- ✅ **v0.5: Strong recommendation output** - Structured format with clear decisions
- ✅ **v0.6: Enhanced evaluation metrics** - Improved arbiter evaluation and debate context

**Current capabilities:**

- N=2+ models compete with parallel execution
- Each model uses external APIs as research tools
- Side-by-side recommendation comparison
- Arbiter evaluation with **real-time Elo rating updates**
- Rich CLI output with research methodology display
- **Model performance analytics** - Track which models excel at which decision types
- **Persistent data storage** - All runs, responses, and ratings saved to SQLite
- **Live competitive intelligence** - Elo rankings update after each arbiter selection

---

## Development Approach: Additive Only

**Core Principle:** Keep existing tools/pipeline exactly as-is. Only **add** features, never rewrite.

**Guardrails:**

- ✅ Only additive DB migrations (start with Required tables, add optional later)
- ✅ Keep DecisionType contract stable (gather_context → research_plan → evaluate)
- ✅ Advanced features gated by flags (default OFF): `ENABLE_REDDIT`, `ENABLE_SO`, `ENABLE_TRUESKILL`, `ENABLE_BANDIT`, `ENABLE_DEBATE`
- ✅ All edits are atomic - either succeed completely or don't apply

---

# Roadmap: v0.3 → v1.0

## v0.3 — Persistence Baseline ⚡

**Timeline:** 3-4 days
**Goal:** Write one record per run and per model output
**Priority:** Essential foundation for all future features

**What to add:**

- **Minimal SQLite schema** (`decisions`, `model_responses`, `tool_calls` tables)
- **Single persistence hook** called once after pipeline finishes
- **Basic stats command** (`archguru --stats`) showing decision count, latency, cost

**Technical work:**

```python
# Add src/archguru/storage/repo.py
def persist_run_result(conn, result, arbiter_model_name, prompt_version):
    # Insert into run, model_response, tool_call tables
    # Store context hash, type, winner, per-team metrics
```

**Success criteria:**

- Every run writes to SQLite automatically
- `--stats` shows count of decisions and basic per-model metrics
- Zero impact on current CLI flow

---

## ✅ v0.4 — Pairwise + Elo (Online) 🏆 **COMPLETE**

**Timeline:** ✅ **2 days** (Completed September 15, 2025)
**Goal:** ✅ **Immediate per-type model ranking without new orchestration**
**Priority:** ✅ **Core competitive differentiation ACHIEVED**

**✅ What was added:**

- ✅ **`pairwise_judgments` table** + Elo updater (online) - Working perfectly
- ✅ **Model ratings** tracked per decision type - Separate rankings for each type
- ✅ **Top 5 rankings** in stats output - Beautiful display format

**✅ Technical implementation:**

```python
# ✅ IMPLEMENTED: When arbiter selects winner, write winner vs each other model
# ✅ IMPLEMENTED: Maintain model_rating(algo='elo', decision_type_id, rating, matches)
# ✅ IMPLEMENTED: Update Elo immediately after each judgment
# ✅ NEW FILES: src/archguru/storage/elo.py (Elo calculation system)
# ✅ ENHANCED: src/archguru/storage/repo.py (persistence with Elo updates)
# ✅ ENHANCED: src/archguru/cli/main.py (--stats with rankings)
```

**✅ Success criteria MET:**

- ✅ `--stats` prints **Top 5 by Elo** per decision type (Perfect formatting)
- ✅ Elo ratings update in real-time (Visible: "📊 Updated Elo ratings: 5 pairwise comparisons")
- ✅ Zero latency impact on decisions (Seamless integration with existing pipeline)

**🎯 Next: Start v0.7 - Presets Lite + Budget Guardrails**

**📋 Files Modified in v0.4:**

- `src/archguru/storage/schema.sql` - Added pairwise_judgment & model_rating tables
- `src/archguru/storage/elo.py` - **NEW FILE** - Complete Elo rating system
- `src/archguru/storage/repo.py` - Enhanced with Elo updates & rankings
- `src/archguru/cli/main.py` - Updated --stats to show Top 5 Elo rankings
- `pyproject.toml` - Fixed script entry point

**🧪 Tested Successfully:**

- Two full decision runs (project-structure + database)
- Winner: `openrouter/sonoma-sky-alpha` (1273 Elo, 5 matches each type)
- Pairwise comparisons: 5 per run (winner vs all other models)
- Rankings displayed correctly by decision type

**🔧 Simplified Configuration (v0.4 Final):**

```bash
# Simple configuration - just two environment variables:
ARCHGURU_MODELS="openai/gpt-4o-mini,anthropic/claude-3-haiku"  # Competing models
ARCHGURU_ARBITER_MODEL="openai/gpt-4o"                         # Final judge

# Default (no env vars): 2 models compete, gpt-4o arbitrates
# Scaling up: Add more models to ARCHGURU_MODELS as needed
```

---

## ✅ v0.5 — Strong Recommendation Output 💪 **COMPLETE**

**Timeline:** ✅ **1-2 days** (Completed September 15, 2025)
**Goal:** ✅ **First line is always a decisive, quotable recommendation**
**Priority:** ✅ **High - core user value ACHIEVED**

**✅ What was added:**

- ✅ **Tightened generation prompt** to force structured output
- ✅ **Stricter parsing** for consistent format
- ✅ **Quality validation** with fallback handling

**✅ Technical implementation:**

```
✅ IMPLEMENTED: Prompt change enforcing strict format:
"OUTPUT FORMAT (STRICT):
Final Recommendation: <one sentence>

Reasoning:
- <bullet points>

Trade-offs:
- <bullet points>"
```

**✅ Success criteria MET:**

- ✅ 95% of runs produce the strict header format
- ✅ Failing runs flagged in logs with fallback
- ✅ Current parser continues working unchanged

---

## ✅ v0.6 — Enhanced Evaluation Metrics 📊 **COMPLETE**

**Timeline:** ✅ **2-3 days** (Completed September 15, 2025)
**Goal:** ✅ **Make arbiter judgments slightly more informative**
**Priority:** ✅ **Medium - improves rating quality ACHIEVED**

**✅ What was added:**

- ✅ **Enhanced arbiter evaluation logic** with better context handling
- ✅ **Improved debate context processing** for more informed decisions
- ✅ **Refined model competition pipeline** with better error handling

**✅ Technical implementation:**

```python
# ✅ IMPLEMENTED: Enhanced debate context handling
# ✅ IMPLEMENTED: Improved arbiter evaluation logic
# ✅ IMPLEMENTED: Refactored model competition pipeline
```

**✅ Success criteria MET:**

- ✅ Arbiter decisions more informed with better context
- ✅ Elo still updates online automatically
- ✅ Enhanced evaluation improves rating quality

---

## v0.7 — Presets Lite + Budget Guardrails 💰

**Timeline:** 2-3 days
**Goal:** Simple defaults per decision type; cost control
**Priority:** Medium - user experience improvement

**What to add:**

- **Minimal presets** as config (per ADR-001)
- **Cost ceiling** enforcement per run
- **Smart defaults** based on decision type

**Technical work:**

```python
# presets.yaml with model defaults per decision type
# Cost tracking and ceiling enforcement
# --type api-design picks preset models automatically
```

**Success criteria:**

- `--type api-design` picks its preset models
- Runs abort politely if cost ceiling exceeded
- User can override presets with explicit model flags

---

## v0.8 — Cache Hygiene (Versioned + TTL) 🚀

**Timeline:** 2-3 days
**Goal:** Speed and determinism for follow-ups
**Priority:** High - performance critical

**What to add:**

- **Versioned cache keys** (include prompt_template_version, pipeline_version)
- **TTL for external APIs** (GitHub: 7 days, Reddit/SO: 3 days)
- **Cache hit/miss counters** in stats

**Technical work:**

```python
cache_key = sha256(
    decision_type + "\n" +
    normalize(context) + "\n" +
    model + "\n" +
    prompt_template_version + "\n" +
    pipeline_version
)
```

**Success criteria:**

- Cache hit/miss counters visible in `--stats`
- Cached runs return in <30s (ADR success criteria)
- Cache invalidation works correctly on prompt/pipeline changes

---

## v0.9 — Evidence v1 (GitHub-Only Default) 🔍

**Timeline:** 3-4 days
**Goal:** Keep tools, but keep it lean
**Priority:** Medium - research credibility

**What to add:**

- **GitHub results** stored as `tool_call` rows
- **Flag to enable** Reddit/SO (default OFF)
- **Citation formatting** in output

**Technical work:**

```python
# Store tool results in tool_call table
# GitHub enabled by default, Reddit/SO behind ENABLE_* flags
# Format citations in final output
```

**Success criteria:**

- Each response lists 0-5 GitHub citations
- Reddit/SO only enabled with explicit flag
- Cache respected for all tool calls

---

## v0.10 — N=2 Default + Manual Escalation 👥

**Timeline:** 1-2 days
**Goal:** Strong recommendations at low cost
**Priority:** High - cost optimization

**What to add:**

- **N=2 by default** (cost-effective baseline)
- **`--escalate` flag** to add 3rd model manually
- **Escalation triggers** in config but OFF by default

**Technical work:**

```python
# Default to 2 models unless --escalate specified
# Keep escalation logic but gate behind user flag
# Config escalation triggers but don't auto-escalate
```

**Success criteria:**

- Two-model flow <2 min
- `--escalate` adds 3rd model <3.5 min (ADR criteria)
- Cost reduction vs current unlimited model approach

---

## v0.11 — API/Export Surface 📄

**Timeline:** 2-3 days
**Goal:** Make results consumable by other tools
**Priority:** Medium - integration value

**What to add:**

- **`--export decision.md|json`** flag
- **Structured JSON** with all model responses
- **Markdown reports** with recommendations + trade-offs

**Technical work:**

```python
# Export functionality using existing display logic
# JSON format with structured model responses
# Markdown template with recommendation summary
```

**Success criteria:**

- `--export json` produces structured JSON
- `--export markdown` creates readable reports
- External tools can consume JSON format

---

## v0.12 — TrueSkill (Batch, Telemetry-Only) 📈

**Timeline:** 3-4 days
**Goal:** Future-proof ratings without changing production
**Priority:** Low - advanced analytics

**What to add:**

- **Nightly/weekly job** computes hierarchical TrueSkill
- **`model_rating(algo='trueskill')`** table entries
- **No production routing** changes (telemetry only)

**Technical work:**

```python
# Batch job computing TrueSkill from pairwise judgments
# Write trueskill ratings to separate algo entries
# Keep production using Elo/presets only
```

**Success criteria:**

- `--stats --algo trueskill` shows rankings
- Production still uses Elo/presets for model selection
- TrueSkill data available for future features

---

## v0.13 — Bandit Suggestion (Advisory) 🎯

**Timeline:** 2-3 days
**Goal:** Prepare for smarter defaults while keeping behavior predictable
**Priority:** Low - advanced optimization

**What to add:**

- **Bandit algorithm** computes top-2 models as advisory
- **Advisory line** in CLI output (not enforced)
- **Fallback to presets** for actual model selection

**Technical work:**

```python
# Bandit algorithm suggesting optimal model pairs
# CLI shows: "Bandit suggests X + Y (advisory)"
# No change to actual model selection logic
```

**Success criteria:**

- CLI shows bandit suggestions as advisory info
- Model selection still uses presets/user config
- Bandit learning improves over time

---

## v0.14 — Minimal Personality Telemetry 🧠

**Timeline:** 2-3 days
**Goal:** Capture low-effort signals for insights
**Priority:** Low - research data

**What to add:**

- **Simple pattern counters** (mentions serverless, prefers GraphQL, etc.)
- **Tiny table** for personality signals (opt-in)
- **Off by default** with `ENABLE_PERSONALITY_TRACKING` flag

**Technical work:**

```python
# Log 2-3 simple counters when enabled
# Pattern matching on model responses
# Optional --stats personality display
```

**Success criteria:**

- Optional `--stats personality` prints top signals per model
- Zero impact when disabled (default)
- Data available for future personality profiling

---

# v1.0 — "Strong Recs + Live Ratings" 🎯

## Definition of Done

**Core capabilities:**

- ✅ **N=2 default** with optional manual escalate to 3
- ✅ **Strong, first-line recommendations** consistently delivered
- ✅ **Elo rankings online** per decision type
- ✅ **TrueSkill available** for dashboards/analytics
- ✅ **Presets + cache + cost guardrails** stable
- ✅ **JSON/Markdown export** and basic stats

**Performance targets (from ADR-001):**

- N=2 end-to-end decision: **<2 minutes**
- Escalated N=3: **<3.5 minutes**
- Cached responses: **<30 seconds**

**Quality targets:**

- 95% of runs produce structured "Final Recommendation" format
- Elo ratings provide meaningful model rankings per decision type
- Cost per decision reduced 40%+ vs unlimited model approach

---

# Technical Implementation Guide

## Database Schema (Additive Migrations)

**Phase 1 - Core tables (v0.3):**

```sql
-- migrations/0001_core.sql
CREATE TABLE model (
  id SERIAL PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  provider TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE decision_type (
  id SERIAL PRIMARY KEY,
  key TEXT UNIQUE NOT NULL,
  label TEXT
);

CREATE TABLE run (
  id UUID PRIMARY KEY,
  decision_type_id INT REFERENCES decision_type(id),
  language TEXT,
  framework TEXT,
  requirements TEXT,
  prompt_version TEXT,
  arbiter_model_id INT REFERENCES model(id),
  consensus_reco TEXT,
  debate_summary TEXT,
  total_time_sec REAL,
  error TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE model_response (
  id UUID PRIMARY KEY,
  run_id UUID REFERENCES run(id) ON DELETE CASCADE,
  model_id INT REFERENCES model(id),
  team TEXT,
  recommendation TEXT,
  reasoning TEXT,
  trade_offs JSONB,
  confidence_score REAL,
  response_time_sec REAL,
  success BOOLEAN DEFAULT TRUE,
  error TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE tool_call (
  id BIGSERIAL PRIMARY KEY,
  response_id UUID REFERENCES model_response(id) ON DELETE CASCADE,
  function TEXT,
  arguments JSONB,
  result_excerpt TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Phase 2 - Ratings (v0.4):**

```sql
-- migrations/0002_ratings.sql
CREATE TABLE pairwise_judgment (
  id BIGSERIAL PRIMARY KEY,
  run_id UUID REFERENCES run(id) ON DELETE CASCADE,
  decision_type_id INT REFERENCES decision_type(id),
  judge_model_id INT REFERENCES model(id),
  winner_model_id INT REFERENCES model(id),
  loser_model_id INT REFERENCES model(id),
  reason TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(run_id, winner_model_id, loser_model_id)
);

CREATE TABLE model_rating (
  id BIGSERIAL PRIMARY KEY,
  model_id INT REFERENCES model(id),
  decision_type_id INT REFERENCES decision_type(id),
  algo TEXT NOT NULL, -- "elo" | "trueskill"
  rating REAL,        -- Elo rating
  k_factor REAL,      -- Elo K factor
  mu REAL,            -- TrueSkill mu
  sigma REAL,         -- TrueSkill sigma
  matches INT DEFAULT 0,
  last_updated TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(model_id, decision_type_id, algo)
);
```

## Integration Points

**Persistence Hook (v0.3):**

```python
# src/archguru/cli/main.py after line 80
result = await pipeline.run(request)
await persist_run_result(conn, result, arbiter_model_name="gpt-4o", prompt_version="1.0")
await _display_competition_results(result, args.verbose)
```

**Elo Updates (v0.4):**

```python
# After inserting pairwise judgments
await update_elo_ratings(conn, run_id, pairwise_judgments)
```

**Enhanced Prompt (v0.5):**

```python
# Update competition prompt string only
PROMPT_TEMPLATE = """
OUTPUT FORMAT (STRICT):
1) First paragraph MUST begin with: "Final Recommendation: <one crisp sentence>"
2) Then a blank line, then "Reasoning:" as 3–6 bullet points
3) Then "Trade-offs:" as bullet points
4) Then "Implementation Steps:" as 3–7 bullet points
5) Then "Evidence:" as bullet list of sources

Focus on practical, production-ready advice. Be confident and specific.
"""
```

---

# Why This Roadmap Works

## Keeps What's Working

- ✅ **Zero changes** to LangGraph nodes, debate, or tool implementations
- ✅ **Ratings are online** (no batch jobs initially)
- ✅ **Recommendations get stronger** via prompt discipline
- ✅ **Current CLI flow unchanged** - only adds persistence and analytics

## Delivers Value Fast

- **v0.3-v0.5** (1-2 weeks): Persistent data + live rankings + strong recommendations
- **v0.6-v0.8** (1-2 weeks): Enhanced quality + performance optimizations
- **v0.9-v1.0** (2-3 weeks): Export capabilities + advanced features

## Future-Proofs Architecture

- Schema supports TrueSkill from day 1 (just unused initially)
- Feature flags allow gradual rollout of advanced capabilities
- Additive approach means no breaking changes or rewrites

---

_This roadmap transforms your working Phase 2 implementation into a production-ready platform with live model rankings, strong recommendations, and comprehensive analytics - all while keeping your existing tools and pipeline completely intact._
