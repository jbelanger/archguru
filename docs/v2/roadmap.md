## **MVP 1: Two-Model Debate Engine** (3-4 days)

### Day 1: Database Schema & Core State Machine

#### 1.1 Complete Database Schema (`src/archguru/storage/debate_schema.sql`)

```sql
-- Add to existing schema.sql

-- Debate tables
CREATE TABLE IF NOT EXISTS debate_match (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES run(id),
  model_a_id INTEGER NOT NULL REFERENCES model(id),
  model_b_id INTEGER NOT NULL REFERENCES model(id),
  winner_model_id INTEGER REFERENCES model(id),

  -- Configuration
  max_rounds INTEGER DEFAULT 3,
  convergence_threshold REAL DEFAULT 80.0,

  -- Results
  final_convergence_score REAL,
  total_rounds_completed INTEGER,
  debate_status TEXT, -- 'converged' | 'forced' | 'failed'

  -- Tournament context (for MVP2)
  match_number INTEGER DEFAULT 1,
  previous_match_id TEXT REFERENCES debate_match(id),

  -- Metadata
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  completed_at TIMESTAMP,
  total_cost_usd REAL
);

CREATE TABLE IF NOT EXISTS debate_round (
  id TEXT PRIMARY KEY,
  match_id TEXT NOT NULL REFERENCES debate_match(id),
  round_number INTEGER NOT NULL,
  round_type TEXT NOT NULL, -- 'initial' | 'challenge' | 'defense'

  -- Model A response
  model_a_response TEXT NOT NULL,
  model_a_confidence REAL,
  model_a_confidence_factors TEXT, -- JSON
  model_a_uncertainty_points TEXT, -- JSON
  model_a_tool_calls TEXT, -- JSON

  -- Model B response
  model_b_response TEXT NOT NULL,
  model_b_confidence REAL,
  model_b_confidence_factors TEXT, -- JSON
  model_b_uncertainty_points TEXT, -- JSON
  model_b_tool_calls TEXT, -- JSON

  -- Convergence
  convergence_score REAL,
  convergence_method TEXT, -- 'arbiter' for MVP1
  convergence_details TEXT, -- JSON

  -- Position tracking
  position_changed_a BOOLEAN DEFAULT FALSE,
  position_changed_b BOOLEAN DEFAULT FALSE,

  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(match_id, round_number)
);

-- Tournament tables (for MVP2, but add now)
CREATE TABLE IF NOT EXISTS tournament (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES run(id),
  total_models INTEGER NOT NULL,
  bracket_type TEXT DEFAULT 'single_elimination',
  total_budget_usd REAL,
  final_winner_id INTEGER REFERENCES model(id),
  total_matches INTEGER,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  completed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tournament_match (
  tournament_id TEXT NOT NULL REFERENCES tournament(id),
  match_id TEXT NOT NULL REFERENCES debate_match(id),
  match_order INTEGER NOT NULL,
  budget_allocated REAL,
  PRIMARY KEY (tournament_id, match_id)
);

-- OpenRouter cache (for MVP3, but add structure now)
CREATE TABLE IF NOT EXISTS openrouter_model (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  provider TEXT,
  context_length INTEGER,
  supports_tools BOOLEAN DEFAULT FALSE,
  prompt_price_usd REAL,
  completion_price_usd REAL,
  or_ranking REAL,
  avg_latency_ms INTEGER,
  availability_score REAL,
  last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_debate_round_lookup ON debate_round(match_id, round_number);
CREATE INDEX IF NOT EXISTS idx_debate_match_run ON debate_match(run_id);
CREATE INDEX IF NOT EXISTS idx_tournament_match ON tournament_match(tournament_id, match_order);
```

#### 1.2 Enhanced Model Response (`src/archguru/models/response.py`)

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class ConfidenceFactors:
    """Structured confidence breakdown"""
    evidence_quality: float      # 0-1: Quality of sources
    solution_fit: float          # 0-1: How well solution matches
    expertise: float             # 0-1: Model's domain familiarity

    def overall(self) -> float:
        return (
            self.evidence_quality * 0.4 +
            self.solution_fit * 0.4 +
            self.expertise * 0.2
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "evidence_quality": self.evidence_quality,
            "solution_fit": self.solution_fit,
            "expertise": self.expertise
        }

@dataclass
class DebateResponse:
    """Response in a debate round"""
    model_name: str
    content: str
    recommendation: str
    confidence: float
    confidence_factors: Optional[ConfidenceFactors] = None
    uncertainty_points: List[str] = field(default_factory=list)
    key_facts: List[str] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    def to_challenge_context(self) -> Dict[str, Any]:
        """Format for challenger to review"""
        return {
            'recommendation': self.recommendation,
            'confidence': self.confidence,
            'weak_points': self.uncertainty_points,
            'claimed_facts': self.key_facts,
            'sources_count': len(self.tool_calls)
        }
```

### Day 2: Debate State Machine & Round Orchestration

#### 2.1 Core Debate Engine (`src/archguru/debate/match.py`)

```python
import uuid
import json
import asyncio
from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

class MatchState(Enum):
    INITIALIZED = "initialized"
    RESEARCHING = "researching"
    CHALLENGING = "challenging"
    DEFENDING = "defending"
    CHECKING_CONVERGENCE = "checking_convergence"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class DebateRound:
    """Single round in a debate"""
    round_num: int
    round_type: str  # 'initial' | 'challenge' | 'defense'
    response_a: DebateResponse
    response_b: DebateResponse
    convergence_score: Optional[float] = None

class DebateMatch:
    def __init__(
        self,
        model_a: str,
        model_b: str,
        context: Dict[str, Any],
        max_rounds: int = 3,
        convergence_threshold: float = 80.0,
        client: Optional[OpenRouterClient] = None
    ):
        self.id = str(uuid.uuid4())
        self.model_a = model_a
        self.model_b = model_b
        self.context = context
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self.rounds: List[DebateRound] = []
        self.state = MatchState.INITIALIZED
        self.client = client or OpenRouterClient()
        self.arbiter_model = Config.get_arbiter_model()

    async def run(self) -> Dict[str, Any]:
        """Execute the complete debate match"""
        try:
            # Round 0: Initial research and positions
            self.state = MatchState.RESEARCHING
            initial_round = await self._run_initial_round()
            self.rounds.append(initial_round)

            # Run up to max_rounds of challenge/defense
            for cycle in range(self.max_rounds):
                # Challenge round (odd numbers)
                self.state = MatchState.CHALLENGING
                challenge_round = await self._run_challenge_round(cycle * 2 + 1)
                self.rounds.append(challenge_round)

                # Check convergence
                self.state = MatchState.CHECKING_CONVERGENCE
                convergence = await self._check_convergence(
                    challenge_round.response_a,
                    challenge_round.response_b
                )
                challenge_round.convergence_score = convergence

                if convergence >= self.convergence_threshold:
                    return await self._complete_match("converged")

                # Defense round (even numbers)
                if cycle < self.max_rounds - 1:  # Skip defense on last cycle
                    self.state = MatchState.DEFENDING
                    defense_round = await self._run_defense_round(cycle * 2 + 2)
                    self.rounds.append(defense_round)

                    # Check convergence again
                    convergence = await self._check_convergence(
                        defense_round.response_a,
                        defense_round.response_b
                    )
                    defense_round.convergence_score = convergence

                    if convergence >= self.convergence_threshold:
                        return await self._complete_match("converged")

            # Max rounds reached - force decision
            return await self._complete_match("forced")

        except Exception as e:
            self.state = MatchState.FAILED
            return self._handle_failure(e)
```

#### 2.2 Round Implementation (`src/archguru/debate/rounds.py`)

```python
from ..debate.prompts import (
    INITIAL_RESEARCH_PROMPT,
    CHALLENGE_PROMPT,
    DEFENSE_PROMPT
)

class RoundOrchestrator:
    def __init__(self, client: OpenRouterClient):
        self.client = client

    async def run_initial_round(
        self,
        model_a: str,
        model_b: str,
        context: Dict[str, Any]
    ) -> DebateRound:
        """Both models research and form initial positions"""

        prompt = INITIAL_RESEARCH_PROMPT.format(
            decision_type=context['decision_type'],
            language=context.get('language', 'Not specified'),
            framework=context.get('framework', 'Not specified'),
            requirements=context.get('requirements', 'None specified')
        )

        # Run both models in parallel
        task_a = self.client.get_model_response(model_a, prompt)
        task_b = self.client.get_model_response(model_b, prompt)

        response_a_raw, response_b_raw = await asyncio.gather(task_a, task_b)

        # Convert to DebateResponse with confidence
        response_a = self._parse_to_debate_response(response_a_raw, model_a)
        response_b = self._parse_to_debate_response(response_b_raw, model_b)

        return DebateRound(
            round_num=0,
            round_type="initial",
            response_a=response_a,
            response_b=response_b
        )

    async def run_challenge_round(
        self,
        round_num: int,
        model_a: str,
        model_b: str,
        prev_round: DebateRound
    ) -> DebateRound:
        """Each model challenges the other's position"""

        # Model A challenges B
        challenge_prompt_a = CHALLENGE_PROMPT.format(
            opponent_recommendation=prev_round.response_b.recommendation,
            opponent_confidence=prev_round.response_b.confidence,
            evidence_quality=prev_round.response_b.confidence_factors.evidence_quality,
            solution_fit=prev_round.response_b.confidence_factors.solution_fit,
            expertise=prev_round.response_b.confidence_factors.expertise,
            opponent_claims='\n'.join(f"- {fact}" for fact in prev_round.response_b.key_facts),
            opponent_sources=f"{len(prev_round.response_b.tool_calls)} research steps",
            opponent_uncertainties=', '.join(prev_round.response_b.uncertainty_points)
        )

        # Model B challenges A (similar)
        challenge_prompt_b = CHALLENGE_PROMPT.format(
            opponent_recommendation=prev_round.response_a.recommendation,
            # ... symmetric for A
        )

        # Execute challenges
        task_a = self.client.get_model_response(model_a, challenge_prompt_a)
        task_b = self.client.get_model_response(model_b, challenge_prompt_b)

        response_a_raw, response_b_raw = await asyncio.gather(task_a, task_b)

        return DebateRound(
            round_num=round_num,
            round_type="challenge",
            response_a=self._parse_to_debate_response(response_a_raw, model_a),
            response_b=self._parse_to_debate_response(response_b_raw, model_b)
        )
```

### Day 3: Convergence, CLI Integration & Persistence

#### 3.1 Convergence Check (`src/archguru/debate/convergence.py`)

```python
class ConvergenceChecker:
    def __init__(self, arbiter_model: str):
        self.arbiter_model = arbiter_model
        self.client = OpenRouterClient()

    async def check_convergence(
        self,
        response_a: DebateResponse,
        response_b: DebateResponse
    ) -> float:
        """Check convergence using arbiter (MVP1: simple arbiter check)"""

        prompt = f"""Analyze the convergence between these two architectural recommendations.

Model A Recommendation:
{response_a.recommendation}
Confidence: {response_a.confidence:.0%}

Model B Recommendation:
{response_b.recommendation}
Confidence: {response_b.confidence:.0%}

Rate their convergence on a 0-100 scale considering:
1. Core recommendation alignment (same solution?)
2. Reasoning alignment (same logic?)
3. Trade-off identification (same concerns?)

Return ONLY a JSON object:
{{
    "overall_convergence": <0-100>,
    "core_recommendation": <0-100>,
    "reasoning_alignment": <0-100>,
    "tradeoff_alignment": <0-100>,
    "key_disagreements": ["point1", "point2"]
}}"""

        response = await self.client.client.chat.completions.create(
            model=self.arbiter_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return result.get("overall_convergence", 0.0)
```

#### 3.2 CLI Integration (`src/archguru/cli/debate_command.py`)

```python
import click

@click.command()
@click.option('--type', '-t', required=True,
              type=click.Choice(['database', 'cache', 'messaging', 'api']))
@click.option('--models', '-m', multiple=True, help='Two models for debate')
@click.option('--debate', is_flag=True, help='Use debate mode instead of competition')
@click.option('--max-rounds', default=3, help='Maximum debate rounds')
@click.option('--convergence-threshold', default=80, help='Convergence % required')
@click.option('--show-debate', is_flag=True, help='Show full debate transcript')
async def archguru_debate(type, models, debate, max_rounds, convergence_threshold, show_debate):
    """Run architectural decision with debate mode"""

    if debate:
        # Validate we have exactly 2 models
        if not models or len(models) != 2:
            if not models:
                # Use defaults
                models = ["openai/gpt-4o-mini", "anthropic/claude-3-haiku"]
            else:
                click.echo("❌ Debate mode requires exactly 2 models")
                return 1

        # Run debate
        match = DebateMatch(
            model_a=models[0],
            model_b=models[1],
            context={'decision_type': type},
            max_rounds=max_rounds,
            convergence_threshold=convergence_threshold
        )

        result = await match.run()

        # Display results
        click.echo(f"\n🥊 Debate Results:")
        click.echo(f"Models: {models[0]} vs {models[1]}")
        click.echo(f"Rounds completed: {len(match.rounds)}")
        click.echo(f"Final convergence: {match.rounds[-1].convergence_score:.0f}%")
        click.echo(f"Status: {result['debate_status']}")

        if result['winner']:
            click.echo(f"🏆 Winner: {result['winner']}")

        click.echo(f"\n📋 Final Recommendation:")
        click.echo(result['consensus_recommendation'])

        if show_debate:
            for round in match.rounds:
                click.echo(f"\n--- Round {round.round_num} ({round.round_type}) ---")
                click.echo(f"{models[0]}: {round.response_a.recommendation[:200]}...")
                click.echo(f"{models[1]}: {round.response_b.recommendation[:200]}...")
                if round.convergence_score:
                    click.echo(f"Convergence: {round.convergence_score:.0f}%")

        # Persist to database
        await persist_debate_match(match, result)

    else:
        # Fall back to existing competition mode
        from ..cli.main import run_decision
        await run_decision(...)
```

---

## **MVP 2: Tournament Brackets** (2-3 days)

### Day 1: Tournament Structure & Bracket Management

#### 1.1 Tournament Orchestrator (`src/archguru/tournament/bracket.py`)

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import uuid

@dataclass
class TournamentConfig:
    models: List[str]
    max_rounds_per_match: int = 3
    convergence_threshold: float = 80.0
    budget_allocations: List[float] = None

    def __post_init__(self):
        if self.budget_allocations is None:
            # Front-loaded: 40%, 30%, 20%, 10%
            self.budget_allocations = [0.4, 0.3, 0.2, 0.1]

class TournamentBracket:
    """Manages single-elimination tournament"""

    def __init__(
        self,
        models: List[str],
        context: Dict[str, Any],
        config: Optional[TournamentConfig] = None
    ):
        self.id = str(uuid.uuid4())
        self.models = models
        self.context = context
        self.config = config or TournamentConfig(models=models)
        self.matches: List[DebateMatch] = []
        self.match_results: List[Dict[str, Any]] = []

    async def run(self) -> Dict[str, Any]:
        """Execute tournament bracket"""

        if len(self.models) < 2:
            raise ValueError("Tournament requires at least 2 models")

        if len(self.models) == 2:
            # Just run single debate
            match = DebateMatch(
                model_a=self.models[0],
                model_b=self.models[1],
                context=self.context,
                max_rounds=self.config.max_rounds_per_match,
                convergence_threshold=self.config.convergence_threshold
            )
            result = await match.run()
            self.matches.append(match)
            self.match_results.append(result)

            return {
                'tournament_id': self.id,
                'final_winner': result['winner'],
                'final_recommendation': result['consensus_recommendation'],
                'total_matches': 1,
                'bracket_type': 'single_match',
                'matches': self.match_results
            }

        # Run tournament with 3+ models
        return await self._run_elimination_bracket()

    async def _run_elimination_bracket(self) -> Dict[str, Any]:
        """Single elimination tournament"""

        # Match 1: First two models
        print(f"\n🥊 Match 1: {self.models[0]} vs {self.models[1]}")
        match1 = DebateMatch(
            model_a=self.models[0],
            model_b=self.models[1],
            context=self.context,
            max_rounds=self.config.max_rounds_per_match,
            convergence_threshold=self.config.convergence_threshold
        )

        result1 = await match1.run()
        self.matches.append(match1)
        self.match_results.append(result1)

        current_winner = result1['winner']
        current_position = result1['winner_position']

        # Advance through challengers
        for i, challenger in enumerate(self.models[2:], start=2):
            print(f"\n🥊 Match {i}: {current_winner} (defender) vs {challenger} (challenger)")

            # Prepare context with history
            enhanced_context = self._prepare_challenger_context(
                base_context=self.context,
                current_winner=current_winner,
                current_position=current_position,
                match_history=self.matches
            )

            # Run match with history awareness
            match = DebateMatch(
                model_a=current_winner,
                model_b=challenger,
                context=enhanced_context,
                max_rounds=self.config.max_rounds_per_match,
                convergence_threshold=self.config.convergence_threshold
            )

            # Set previous_match_id for tracking
            match.previous_match_id = self.matches[-1].id if self.matches else None

            result = await match.run()
            self.matches.append(match)
            self.match_results.append(result)

            # Update current winner
            if result['winner'] == challenger:
                print(f"  🔄 New leader: {challenger}")
                current_winner = challenger
                current_position = result['winner_position']
            else:
                print(f"  ✅ {current_winner} defends position")
                current_position = result['winner_position']  # May be refined

        return {
            'tournament_id': self.id,
            'final_winner': current_winner,
            'final_recommendation': current_position['recommendation'],
            'total_matches': len(self.matches),
            'bracket_type': 'single_elimination',
            'matches': self.match_results,
            'elimination_order': self._get_elimination_order()
        }
```

### Day 2: History Context & Enhanced Prompts

#### 2.1 Challenger Context (`src/archguru/tournament/context.py`)

```python
class TournamentContextManager:
    """Manages context and history for tournament progression"""

    def prepare_challenger_context(
        self,
        base_context: Dict[str, Any],
        current_winner: str,
        current_position: Dict[str, Any],
        match_history: List[DebateMatch]
    ) -> Dict[str, Any]:
        """Enhance context for challenger with tournament history"""

        # Extract key points from previous debates
        discussed_points = self._extract_key_points(match_history)
        previous_challenges = self._extract_challenges(match_history)

        enhanced_context = {
            **base_context,
            'tournament_context': {
                'current_leader': current_winner,
                'current_position': current_position,
                'points_discussed': discussed_points,
                'previous_challenges': previous_challenges,
                'match_number': len(match_history) + 1
            }
        }

        return enhanced_context

    def _extract_key_points(self, matches: List[DebateMatch]) -> List[str]:
        """Extract key discussion points from match history"""
        points = []

        for match in matches:
            for round in match.rounds:
                if round.round_type == 'challenge':
                    # Extract main challenges raised
                    if hasattr(round.response_a, 'key_challenges'):
                        points.extend(round.response_a.key_challenges[:2])
                    if hasattr(round.response_b, 'key_challenges'):
                        points.extend(round.response_b.key_challenges[:2])

        # Deduplicate and limit
        unique_points = list(dict.fromkeys(points))
        return unique_points[:10]  # Top 10 key points

    def _extract_challenges(self, matches: List[DebateMatch]) -> List[str]:
        """Extract challenges that have been addressed"""
        challenges = []

        for match in matches:
            for round in match.rounds:
                if round.round_type == 'defense':
                    # Track what was successfully defended
                    if round.response_a.position_changed:
                        challenges.append(f"Changed: {round.response_a.change_reason}")
                    if round.response_b.position_changed:
                        challenges.append(f"Changed: {round.response_b.change_reason}")

        return challenges[:5]
```

#### 2.2 Tournament-Aware Prompts (`src/archguru/tournament/prompts.py`)

```python
CHALLENGER_INITIAL_PROMPT = """You are entering an architectural decision debate tournament.

The current leader is {current_leader} with this position:
{current_position}

Previous debate highlights:
{match_history}

Key points already discussed:
{discussed_points}

You must provide a fresh perspective. Do not repeat arguments that have already been addressed.
Focus on:
1. New angles not yet considered
2. Unconsidered trade-offs
3. Superior alternatives with evidence

Research the topic and form your initial position.
Remember: You're not biased by the history - if the leader is wrong, prove it.

{base_prompt}"""

DEFENDER_CONTINUATION_PROMPT = """You've won the previous round with your recommendation.
A new challenger ({challenger}) is entering the debate.

Your current refined position:
{current_position}

Confidence: {confidence}%

Previous successful defenses:
{successful_defenses}

Briefly restate your position, incorporating any refinements from previous debates.
Be ready to defend against new challenges while maintaining consistency."""
```

### Day 3: CLI Integration & Persistence

#### 3.1 Enhanced CLI (`src/archguru/cli/main.py` updates)

```python
@click.option('--tournament-size', '-n', default=2,
              help='Number of models in tournament (2-5)')
@click.option('--tournament', is_flag=True,
              help='Run tournament mode with 3+ models')
async def archguru_main(..., tournament_size, tournament):
    """Enhanced CLI with tournament support"""

    if tournament or tournament_size > 2:
        # Get models for tournament
        if models and len(models) >= tournament_size:
            tournament_models = models[:tournament_size]
        else:
            # Use defaults or selection logic
            tournament_models = get_default_tournament_models(tournament_size)

        # Run tournament
        bracket = TournamentBracket(
            models=tournament_models,
            context={
                'decision_type': type,
                'language': language,
                'framework': framework,
                'requirements': requirements
            }
        )

        result = await bracket.run()

        # Display tournament results
        display_tournament_results(result)

        # Persist tournament
        await persist_tournament(bracket, result)

    elif debate:
        # Run simple 2-model debate (MVP1)
        ...
    else:
        # Run original competition mode
        ...
```

#### 3.2 Tournament Persistence (`src/archguru/storage/tournament_repo.py`)

```python
class TournamentRepository:
    """Persist tournament data"""

    async def persist_tournament(
        self,
        bracket: TournamentBracket,
        result: Dict[str, Any]
    ):
        """Save complete tournament to database"""

        with sqlite3.connect(self.db_path) as conn:
            # Create tournament record
            conn.execute("""
                INSERT INTO tournament (
                    id, run_id, total_models, bracket_type,
                    total_budget_usd, final_winner_id, total_matches
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                bracket.id,
                bracket.context.get('run_id'),
                len(bracket.models),
                result['bracket_type'],
                None,  # Budget tracking in later MVP
                self._get_model_id(conn, result['final_winner']),
                len(bracket.matches)
            ))

            # Save each match
            for i, match in enumerate(bracket.matches):
                # Persist debate_match record
                self._persist_debate_match(conn, match)

                # Link to tournament
                conn.execute("""
                    INSERT INTO tournament_match (
                        tournament_id, match_id, match_order, budget_allocated
                    ) VALUES (?, ?, ?, ?)
                """, (
                    bracket.id,
                    match.id,
                    i + 1,
                    None  # Budget in later MVP
                ))

                # Save all rounds
                for round in match.rounds:
                    self._persist_debate_round(conn, match.id, round)

            conn.commit()
```

## Key Implementation Notes:

### For MVP1:

- **No Redis needed** - SQLite handles everything
- **Simple arbiter convergence** - Just one LLM call
- **Hardcoded models** work fine with `--models` flag
- **Reuse existing** `OpenRouterClient` and tool infrastructure
- **Database complete** from day 1 - no future migrations

### For MVP2:

- **Build on MVP1** - Just add tournament layer
- **History context** prevents repetitive arguments
- **Same database** - tournament tables already exist
- **Progressive refinement** - Winner's position evolves
- **CLI backwards compatible** - Old commands still work

### Testing Strategy:

```bash
# MVP1 tests
archguru --type database --debate --models "gpt-4o-mini" "claude-3-haiku"
archguru --type api-design --debate --show-debate --max-rounds 2

# MVP2 tests
archguru --type database --tournament-size 3
archguru --type cache --tournament --models "gpt-4o-mini" "claude-3-haiku" "gemini-1.5-flash"
```

## **MVP 3: OpenRouter Intelligence Integration** (3-4 days)

### Day 1: OpenRouter Cache Schema & API Integration

#### 1.1 OpenRouter API Client (`src/archguru/intelligence/openrouter_api.py`)

```python
import aiohttp
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

class OpenRouterAPI:
    """Direct OpenRouter API integration for model intelligence"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/jbelanger/archguru",
            "X-Title": "ArchGuru"
        }

    async def fetch_models(self) -> List[Dict[str, Any]]:
        """Fetch all available models from OpenRouter"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/models",
                headers=self.headers
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("data", [])
                else:
                    print(f"Failed to fetch models: {resp.status}")
                    return []

    async def fetch_model_details(self, model_id: str) -> Dict[str, Any]:
        """Get detailed info for a specific model"""
        # OpenRouter doesn't have individual model endpoints,
        # so we fetch all and filter
        models = await self.fetch_models()
        for model in models:
            if model.get("id") == model_id:
                return model
        return {}

class OpenRouterIntelligence:
    """Manages OpenRouter model intelligence and caching"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or Path.home() / ".archguru" / "archguru.db"
        self.api = OpenRouterAPI(Config.OPENROUTER_API_KEY)
        self.cache_ttl_hours = 24

    async def refresh_model_cache(self, force: bool = False) -> int:
        """Refresh OpenRouter model data in cache"""

        # Check if cache needs refresh
        if not force and not self._cache_needs_refresh():
            print("Cache is fresh, skipping refresh")
            return 0

        print("🔄 Refreshing OpenRouter model cache...")
        models = await self.api.fetch_models()

        if not models:
            print("❌ No models fetched from OpenRouter")
            return 0

        with sqlite3.connect(self.db_path) as conn:
            updated_count = 0

            for model in models:
                # Extract relevant fields
                model_data = self._parse_model_data(model)

                # Upsert into cache
                conn.execute("""
                    INSERT INTO openrouter_model (
                        id, name, provider, context_length, supports_tools,
                        prompt_price_usd, completion_price_usd, or_ranking,
                        avg_latency_ms, availability_score, architecture,
                        training_cutoff, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(id) DO UPDATE SET
                        name=excluded.name,
                        context_length=excluded.context_length,
                        supports_tools=excluded.supports_tools,
                        prompt_price_usd=excluded.prompt_price_usd,
                        completion_price_usd=excluded.completion_price_usd,
                        or_ranking=excluded.or_ranking,
                        avg_latency_ms=excluded.avg_latency_ms,
                        availability_score=excluded.availability_score,
                        architecture=excluded.architecture,
                        training_cutoff=excluded.training_cutoff,
                        last_updated=CURRENT_TIMESTAMP
                """, model_data)
                updated_count += 1

            conn.commit()
            print(f"✅ Updated {updated_count} models in cache")
            return updated_count

    def _parse_model_data(self, model: Dict[str, Any]) -> tuple:
        """Parse OpenRouter model data into our schema"""

        # Extract pricing (OpenRouter uses per-token pricing)
        pricing = model.get("pricing", {})
        prompt_price = float(pricing.get("prompt", 0)) * 1_000_000  # Convert to per-million
        completion_price = float(pricing.get("completion", 0)) * 1_000_000

        # Determine if model supports tools (function calling)
        # This is a heuristic - OpenRouter doesn't always expose this
        supports_tools = self._detect_tool_support(model)

        # Calculate OR ranking (if not provided, use a heuristic)
        or_ranking = self._calculate_or_ranking(model)

        return (
            model.get("id"),                              # id
            model.get("name", model.get("id")),           # name
            model.get("id", "").split("/")[0],            # provider
            model.get("context_length", 4096),            # context_length
            supports_tools,                               # supports_tools
            prompt_price,                                  # prompt_price_usd
            completion_price,                              # completion_price_usd
            or_ranking,                                    # or_ranking
            None,                                          # avg_latency_ms (not provided)
            1.0,                                           # availability_score (assume 100%)
            model.get("architecture", "chat"),            # architecture
            None,                                          # training_cutoff (not provided)
        )

    def _detect_tool_support(self, model: Dict[str, Any]) -> bool:
        """Heuristic to detect if model supports function calling"""
        model_id = model.get("id", "").lower()

        # Known models with tool support
        tool_capable = [
            "gpt-4", "gpt-3.5", "claude-3", "claude-2",
            "command", "gemini", "mistral-large"
        ]

        return any(capable in model_id for capable in tool_capable)

    def _calculate_or_ranking(self, model: Dict[str, Any]) -> float:
        """Calculate a ranking score based on available metrics"""
        # Simple heuristic based on context length and pricing
        context = model.get("context_length", 4096)
        pricing = model.get("pricing", {})
        prompt_price = float(pricing.get("prompt", 0.001))

        # Higher context and lower price = better score
        if prompt_price > 0:
            score = (context / 1000) / (prompt_price * 100)
        else:
            score = context / 1000

        return min(100, score)  # Cap at 100
```

### Day 2: Model Selection Algorithm

#### 2.1 Intelligent Model Selector (`src/archguru/intelligence/selector.py`)

```python
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import sqlite3

@dataclass
class ModelCandidate:
    """Model candidate with scoring information"""
    id: str
    name: str
    provider: str
    context_length: int
    supports_tools: bool
    prompt_price: float
    completion_price: float
    or_ranking: float
    elo_rating: float
    matches_played: int
    debate_score: float = 0.0

    @property
    def estimated_cost_per_round(self) -> float:
        """Estimate cost for one debate round"""
        # Assume ~2000 prompt tokens and ~800 completion tokens per round
        prompt_cost = (2000 * self.prompt_price) / 1_000_000
        completion_cost = (800 * self.completion_price) / 1_000_000
        return prompt_cost + completion_cost

class ModelSelector:
    """Intelligent model selection using OR data + our rankings"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or Path.home() / ".archguru" / "archguru.db"
        self.intelligence = OpenRouterIntelligence(db_path)

    async def select_debate_pair(
        self,
        decision_type: str,
        user_models: Optional[List[str]] = None,
        force_diversity: bool = True,
        min_context: int = 16000,
        require_tools: bool = True
    ) -> Tuple[str, str]:
        """Select optimal model pair for debate"""

        # User override takes precedence
        if user_models and len(user_models) >= 2:
            return await self._validate_models(user_models[:2])

        # Refresh cache if needed (non-blocking check)
        await self._refresh_if_stale()

        # Get suitable candidates
        candidates = await self._get_debate_candidates(
            decision_type,
            min_context,
            require_tools
        )

        if len(candidates) < 2:
            print(f"⚠️  Only {len(candidates)} suitable models found, using defaults")
            return ("openai/gpt-4o-mini", "anthropic/claude-3-haiku")

        # Score each candidate
        for candidate in candidates:
            candidate.debate_score = self._calculate_debate_score(
                candidate,
                decision_type
            )

        # Sort by score
        candidates.sort(key=lambda c: c.debate_score, reverse=True)

        # Select primary (strongest within reason)
        model_a = candidates[0]

        # Select challenger (complementary)
        model_b = self._select_complementary(
            candidates[1:],  # Exclude primary
            primary=model_a,
            force_diversity=force_diversity
        )

        print(f"🎯 Selected models for debate:")
        print(f"   Primary: {model_a.id} (score: {model_a.debate_score:.2f})")
        print(f"   Challenger: {model_b.id} (score: {model_b.debate_score:.2f})")

        return (model_a.id, model_b.id)

    async def _get_debate_candidates(
        self,
        decision_type: str,
        min_context: int,
        require_tools: bool
    ) -> List[ModelCandidate]:
        """Get all suitable model candidates"""

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get decision type ID
            dt_result = conn.execute(
                "SELECT id FROM decision_type WHERE key = ?",
                (decision_type,)
            ).fetchone()

            if not dt_result:
                decision_type_id = None
            else:
                decision_type_id = dt_result['id']

            # Query suitable models with Elo ratings
            query = """
                SELECT
                    om.*,
                    COALESCE(mr.rating, 1200) as elo_rating,
                    COALESCE(mr.matches, 0) as matches_played
                FROM openrouter_model om
                LEFT JOIN model m ON om.id = m.name
                LEFT JOIN model_rating mr ON m.id = mr.model_id
                    AND mr.decision_type_id = ? AND mr.algo = 'elo'
                WHERE om.context_length >= ?
            """

            params = [decision_type_id, min_context]

            if require_tools:
                query += " AND om.supports_tools = 1"

            query += " AND om.availability_score > 0.5"

            cursor = conn.execute(query, params)
            results = cursor.fetchall()

            candidates = []
            for row in results:
                candidates.append(ModelCandidate(
                    id=row['id'],
                    name=row['name'],
                    provider=row['provider'],
                    context_length=row['context_length'],
                    supports_tools=bool(row['supports_tools']),
                    prompt_price=row['prompt_price_usd'],
                    completion_price=row['completion_price_usd'],
                    or_ranking=row['or_ranking'] or 50.0,
                    elo_rating=row['elo_rating'],
                    matches_played=row['matches_played']
                ))

            return candidates

    def _calculate_debate_score(
        self,
        candidate: ModelCandidate,
        decision_type: str
    ) -> float:
        """Calculate composite score for debate suitability"""

        # Normalize components to 0-1 range
        or_score = min(1.0, candidate.or_ranking / 100)
        elo_score = min(1.0, candidate.elo_rating / 1600)

        # Price score (inverse - lower is better)
        # Assume $0.01 per 1K tokens is expensive, $0.0001 is cheap
        total_price = candidate.prompt_price + candidate.completion_price
        price_score = 1.0 - min(1.0, total_price / 10.0)

        # Experience bonus
        experience_score = min(1.0, candidate.matches_played / 10)

        # Composite score with weights
        score = (
            0.30 * or_score +           # OpenRouter quality
            0.25 * elo_score +          # Our domain expertise
            0.20 * price_score +        # Cost efficiency
            0.15 * 1.0 +                # Tool support (already filtered)
            0.10 * experience_score     # Proven track record
        )

        return score * 100  # Scale to 0-100

    def _select_complementary(
        self,
        candidates: List[ModelCandidate],
        primary: ModelCandidate,
        force_diversity: bool
    ) -> ModelCandidate:
        """Select a complementary challenger model"""

        if not candidates:
            raise ValueError("No candidates available for challenger")

        if force_diversity:
            # Prefer different provider
            different_provider = [
                c for c in candidates
                if c.provider != primary.provider
            ]

            if different_provider:
                # Pick best from different provider
                return different_provider[0]

        # Otherwise just pick the next best
        return candidates[0]
```

### Day 3: Tournament Integration & Model Ranking Display

#### 3.1 Tournament with Dynamic Selection (`src/archguru/tournament/smart_bracket.py`)

```python
class SmartTournamentBracket(TournamentBracket):
    """Tournament with intelligent model selection"""

    def __init__(
        self,
        decision_type: str,
        context: Dict[str, Any],
        n_models: int = 3,
        user_models: Optional[List[str]] = None
    ):
        self.selector = ModelSelector()
        self.decision_type = decision_type
        self.n_models = n_models
        self.user_models = user_models

        # Will be populated by select_models()
        super().__init__([], context)

    async def select_models(self) -> List[str]:
        """Select models for tournament bracket"""

        selected = []

        if self.user_models and len(self.user_models) >= self.n_models:
            # Use user-provided models
            selected = self.user_models[:self.n_models]
        else:
            # Intelligent selection for tournament
            selected = await self._select_tournament_models()

        self.models = selected
        return selected

    async def _select_tournament_models(self) -> List[str]:
        """Select diverse models for tournament"""

        # First pair: strongest available models
        model1, model2 = await self.selector.select_debate_pair(
            self.decision_type,
            force_diversity=False  # Get absolute best
        )
        selected = [model1, model2]

        # Additional challengers: diverse perspectives
        for i in range(self.n_models - 2):
            challenger = await self._select_next_challenger(selected)
            selected.append(challenger)

        return selected

    async def _select_next_challenger(
        self,
        already_selected: List[str]
    ) -> str:
        """Select next challenger for diversity"""

        candidates = await self.selector._get_debate_candidates(
            self.decision_type,
            min_context=16000,
            require_tools=True
        )

        # Filter out already selected
        available = [
            c for c in candidates
            if c.id not in already_selected
        ]

        if not available:
            raise ValueError("No more models available for tournament")

        # Prefer different providers for diversity
        selected_providers = [
            c.id.split("/")[0] for c in already_selected
        ]

        different_provider = [
            c for c in available
            if c.id.split("/")[0] not in selected_providers
        ]

        if different_provider:
            # Sort by score and pick best
            different_provider.sort(
                key=lambda c: c.debate_score,
                reverse=True
            )
            return different_provider[0].id

        # Otherwise pick next best regardless of provider
        available.sort(key=lambda c: c.debate_score, reverse=True)
        return available[0].id
```

#### 3.2 Model Rankings Display (`src/archguru/intelligence/rankings.py`)

```python
class ModelRankingsDisplay:
    """Display model rankings and recommendations"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or Path.home() / ".archguru" / "archguru.db"

    async def show_recommendations(
        self,
        decision_type: str,
        limit: int = 10
    ) -> str:
        """Show recommended models for a decision type"""

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get decision type ID
            dt_result = conn.execute(
                "SELECT id FROM decision_type WHERE key = ?",
                (decision_type,)
            ).fetchone()

            decision_type_id = dt_result['id'] if dt_result else None

            # Query with composite scoring
            results = conn.execute("""
                SELECT
                    om.*,
                    m.id as model_internal_id,
                    COALESCE(mr.rating, 1200) as elo,
                    COALESCE(mr.matches, 0) as matches,
                    (om.prompt_price_usd + om.completion_price_usd) as total_price
                FROM openrouter_model om
                LEFT JOIN model m ON om.id = m.name
                LEFT JOIN model_rating mr ON m.id = mr.model_id
                    AND mr.decision_type_id = ? AND mr.algo = 'elo'
                WHERE om.supports_tools = 1
                ORDER BY
                    (om.or_ranking * 0.4 + COALESCE(mr.rating/2000, 0.6) * 0.6) DESC
                LIMIT ?
            """, (decision_type_id, limit)).fetchall()

            output = f"🏆 Top models for {decision_type} decisions:\n\n"

            for i, row in enumerate(results, 1):
                output += f"{i}. {row['id']}\n"
                output += f"   OR Score: {row['or_ranking']:.1f} | "
                output += f"Elo: {row['elo']:.0f} ({row['matches']} matches) | "
                output += f"Cost: ${row['total_price']/1000:.4f}/1K tokens\n"

                # Show specialization
                if row['matches'] > 10:
                    if row['elo'] > 1400:
                        output += f"   ✨ Proven expert in {decision_type}\n"
                    elif row['elo'] < 1000:
                        output += f"   ⚠️ Struggles with {decision_type}\n"

                # Show provider and context
                output += f"   Provider: {row['provider']} | "
                output += f"Context: {row['context_length']:,} tokens\n\n"

            return output
```

---

## **MVP 4: Advanced Convergence & Research** (2-3 days)

### Day 1: Three-Layer Convergence System

#### 1.1 Multi-Method Convergence (`src/archguru/convergence/analyzer.py`)

```python
import re
import json
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from difflib import SequenceMatcher
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

@dataclass
class ConvergenceResult:
    """Result of convergence measurement"""
    score: float                    # 0-100 percentage
    method: str                      # 'structural' | 'semantic' | 'arbiter'

    # Detailed breakdown (when method=ARBITER)
    core_recommendation_alignment: Optional[float] = None
    reasoning_alignment: Optional[float] = None
    tradeoff_alignment: Optional[float] = None
    implementation_alignment: Optional[float] = None

    # Key differences
    key_disagreements: List[str] = field(default_factory=list)

    def is_converged(self, threshold: float = 80.0) -> bool:
        return self.score >= threshold

class ThreeLayerConvergence:
    """Three-layer convergence measurement system"""

    def __init__(self):
        self.arbiter_model = Config.get_arbiter_model()
        self.client = OpenRouterClient()
        # Initialize sentence transformer for semantic similarity
        self.semantic_model = None  # Lazy load

    async def measure_convergence(
        self,
        response_a: DebateResponse,
        response_b: DebateResponse
    ) -> ConvergenceResult:
        """Multi-method convergence measurement"""

        # Layer 1: Structural alignment (fast, cheap)
        structural_score = self._check_structural_alignment(
            response_a.recommendation,
            response_b.recommendation
        )

        if structural_score > 95:  # Nearly identical
            print(f"  ✅ Structural convergence: {structural_score:.0f}%")
            return ConvergenceResult(
                score=structural_score,
                method="structural"
            )

        # Layer 2: Semantic similarity (medium cost)
        semantic_score = await self._check_semantic_similarity(
            response_a,
            response_b
        )

        if semantic_score > 90:  # Very similar meaning
            print(f"  ✅ Semantic convergence: {semantic_score:.0f}%")
            return ConvergenceResult(
                score=semantic_score,
                method="semantic"
            )

        # Layer 3: Arbiter judgment (highest cost, most accurate)
        print(f"  🤔 Structural: {structural_score:.0f}%, Semantic: {semantic_score:.0f}%")
        print(f"  📊 Using arbiter for detailed analysis...")

        arbiter_result = await self._arbiter_convergence_check(
            response_a,
            response_b
        )

        return arbiter_result

    def _check_structural_alignment(
        self,
        text_a: str,
        text_b: str
    ) -> float:
        """Check structural similarity using string matching"""

        # Normalize texts
        norm_a = self._normalize_text(text_a)
        norm_b = self._normalize_text(text_b)

        # Use difflib for similarity ratio
        matcher = SequenceMatcher(None, norm_a, norm_b)
        ratio = matcher.ratio()

        # Also check key terms overlap
        terms_a = set(norm_a.lower().split())
        terms_b = set(norm_b.lower().split())

        if len(terms_a) > 0 and len(terms_b) > 0:
            overlap = len(terms_a & terms_b) / len(terms_a | terms_b)
        else:
            overlap = 0

        # Weighted combination
        score = (ratio * 0.6 + overlap * 0.4) * 100

        return min(100, score)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    async def _check_semantic_similarity(
        self,
        response_a: DebateResponse,
        response_b: DebateResponse
    ) -> float:
        """Check semantic similarity using embeddings"""

        # Lazy load model
        if self.semantic_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                print("  ⚠️  sentence-transformers not installed, skipping semantic check")
                return 0.0

        # Combine key elements for comparison
        text_a = f"{response_a.recommendation} {response_a.content[:500]}"
        text_b = f"{response_b.recommendation} {response_b.content[:500]}"

        # Generate embeddings
        embeddings = self.semantic_model.encode([text_a, text_b])

        # Calculate cosine similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        # Convert to percentage
        score = similarity * 100

        return min(100, max(0, score))

    async def _arbiter_convergence_check(
        self,
        response_a: DebateResponse,
        response_b: DebateResponse
    ) -> ConvergenceResult:
        """Detailed convergence check using arbiter"""

        prompt = f"""Analyze the convergence between these two architectural recommendations.

Model A Recommendation:
{response_a.recommendation}
Confidence: {response_a.confidence:.0%}
Key Points: {', '.join(response_a.key_facts[:3])}

Model B Recommendation:
{response_b.recommendation}
Confidence: {response_b.confidence:.0%}
Key Points: {', '.join(response_b.key_facts[:3])}

Rate their convergence on:
1. Core recommendation (same solution?): 0-100
2. Key reasoning points (same logic?): 0-100
3. Trade-off identification (same concerns?): 0-100
4. Implementation approach (same steps?): 0-100

Return ONLY a JSON object:
{{
    "overall_convergence": <0-100>,
    "core_recommendation": <0-100>,
    "reasoning_alignment": <0-100>,
    "tradeoff_alignment": <0-100>,
    "implementation_alignment": <0-100>,
    "key_disagreements": ["point1", "point2", "point3"]
}}"""

        try:
            response = await self.client.client.chat.completions.create(
                model=self.arbiter_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            return ConvergenceResult(
                score=result.get("overall_convergence", 0),
                method="arbiter",
                core_recommendation_alignment=result.get("core_recommendation"),
                reasoning_alignment=result.get("reasoning_alignment"),
                tradeoff_alignment=result.get("tradeoff_alignment"),
                implementation_alignment=result.get("implementation_alignment"),
                key_disagreements=result.get("key_disagreements", [])
            )

        except Exception as e:
            print(f"  ❌ Arbiter convergence check failed: {e}")
            # Fallback to structural score
            return ConvergenceResult(
                score=self._check_structural_alignment(
                    response_a.recommendation,
                    response_b.recommendation
                ),
                method="structural",
                key_disagreements=["Arbiter check failed"]
            )
```

### Day 2: Research Budget Management & Verification

#### 2.1 Research Budget Manager (`src/archguru/debate/research_budget.py`)

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class ResearchType(Enum):
    INITIAL = "initial"              # Unlimited (within reason)
    CHALLENGE = "challenge"          # Max 2 queries
    DEFENSE = "defense"              # Max 2 queries
    VERIFICATION = "verification"    # Specific claim verification

@dataclass
class ResearchBudget:
    """Research budget for a debate round"""
    round_type: ResearchType
    max_queries: Optional[int]
    queries_used: int = 0

    def can_research(self) -> bool:
        if self.max_queries is None:
            return True  # Unlimited for initial
        return self.queries_used < self.max_queries

    def use_query(self):
        self.queries_used += 1

class ResearchBudgetManager:
    """Manages research budgets across debate rounds"""

    DEFAULT_BUDGETS = {
        ResearchType.INITIAL: None,      # Unlimited
        ResearchType.CHALLENGE: 2,       # 2 queries max
        ResearchType.DEFENSE: 2,         # 2 queries max
        ResearchType.VERIFICATION: 1     # 1 query for specific verification
    }

    def __init__(self):
        self.round_budgets: Dict[int, ResearchBudget] = {}

    def get_budget(self, round_num: int, round_type: str) -> ResearchBudget:
        """Get research budget for a round"""

        if round_num not in self.round_budgets:
            research_type = self._map_round_type(round_type)
            max_queries = self.DEFAULT_BUDGETS.get(research_type)

            self.round_budgets[round_num] = ResearchBudget(
                round_type=research_type,
                max_queries=max_queries
            )

        return self.round_budgets[round_num]

    def _map_round_type(self, round_type: str) -> ResearchType:
        """Map round type string to ResearchType enum"""
        mapping = {
            'initial': ResearchType.INITIAL,
            'challenge': ResearchType.CHALLENGE,
            'defense': ResearchType.DEFENSE
        }
        return mapping.get(round_type, ResearchType.INITIAL)

class VerificationTools:
    """Tools for claim verification in challenges"""

    def __init__(self, client: OpenRouterClient):
        self.client = client

    async def verify_claim(
        self,
        claim: str,
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """Verify a specific claim made by opponent"""

        verification_prompt = f"""
        Verify this claim: "{claim}"
        {f'Source cited: {source}' if source else ''}

        Search for evidence that supports or contradicts this claim.
        Focus on authoritative sources and real examples.
        """

        # Use limited research budget for verification
        tools = self.client.get_research_tools()

        # Construct verification query
        if "github" in claim.lower() or "repository" in claim.lower():
            # Search GitHub for examples
            query = f"search_github_repos for: {claim[:100]}"
        elif "community" in claim.lower() or "discussion" in claim.lower():
            # Search Reddit for discussions
            query = f"search_reddit_discussions for: {claim[:100]}"
        else:
            # Search StackOverflow for technical validation
            query = f"search_stackoverflow for: {claim[:100]}"

        # Execute verification
        result = await self.client.execute_tool_call({
            'function': {'name': query.split(' for:')[0], 'arguments': {'query': claim}}
        })

        return {
            'claim': claim,
            'verification_result': result,
            'verified': self._assess_verification(claim, result)
        }

    def _assess_verification(self, claim: str, result: str) -> bool:
        """Assess if verification supports the claim"""
        # Simple heuristic - check if results contain relevant content
        if not result or "error" in result.lower():
            return False

        # More sophisticated verification could use NLP here
        return len(result) > 100  # Has substantial results
```

### Day 3: Position Change Detection & Non-Convergent Handling

#### 3.1 Position Change Detector (`src/archguru/debate/position_tracker.py`)

```python
class PositionTracker:
    """Track position changes across debate rounds"""

    def __init__(self):
        self.positions: Dict[str, List[str]] = {}  # model -> [positions]

    def detect_position_change(
        self,
        model: str,
        original_position: str,
        new_position: str,
        threshold: float = 0.7
    ) -> Tuple[bool, Optional[str]]:
        """Detect if model changed position significantly"""

        # Structural similarity check
        similarity = SequenceMatcher(
            None,
            self._extract_core(original_position),
            self._extract_core(new_position)
        ).ratio()

        changed = similarity < threshold

        if changed:
            # Identify what changed
            change_reason = self._identify_change_reason(
                original_position,
                new_position
            )
            return True, change_reason

        return False, None

    def _extract_core(self, position: str) -> str:
        """Extract core recommendation from position"""
        # Get first sentence or key recommendation
        lines = position.split('\n')
        for line in lines:
            if line.strip() and not line.endswith(':'):
                return line.strip()[:200]
        return position[:200]

    def _identify_change_reason(
        self,
        original: str,
        new: str
    ) -> str:
        """Identify reason for position change"""

        # Look for explicit change indicators
        if "however" in new.lower() or "actually" in new.lower():
            return "Reconsidered based on challenges"
        elif "updated" in new.lower() or "revised" in new.lower():
            return "Refined position after debate"
        else:
            return "Position evolved through discussion"

class NonConvergentHandler:
    """Handle debates that don't converge"""

    async def handle_non_convergence(
        self,
        match: DebateMatch
    ) -> Dict[str, Any]:
        """Handle debates that don't converge"""

        # Get final positions
        final_round = match.rounds[-1]

        # Analyze fundamental disagreement
        analysis = await self._analyze_disagreement(
            final_round.response_a,
            final_round.response_b,
            match.rounds
        )

        # Prepare result for user
        result = {
            'debate_status': 'non_convergent',
            'model_a': match.model_a,
            'model_b': match.model_b,
            'position_a': final_round.response_a.recommendation,
            'position_b': final_round.response_b.recommendation,
            'fundamental_disagreement': analysis['core_disagreement'],
            'rounds_completed': len(match.rounds),
            'convergence_progression': [
                r.convergence_score for r in match.rounds
                if hasattr(r, 'convergence_score')
            ],
            'arbiter_forced_choice': analysis['recommended_position'],
            'why_no_convergence': analysis['disagreement_reason']
        }

        # Special formatting for non-convergent debates
        self._display_non_convergent_result(result)

        return result

    async def _analyze_disagreement(
        self,
        response_a: DebateResponse,
        response_b: DebateResponse,
        rounds: List[DebateRound]
    ) -> Dict[str, Any]:
        """Analyze why models couldn't converge"""

        # Check if it's fundamental philosophical difference
        if self._is_philosophical_difference(response_a, response_b):
            return {
                'core_disagreement': 'Fundamental philosophical difference',
                'disagreement_reason': 'Models have incompatible design philosophies',
                'recommended_position': self._pick_pragmatic_choice(response_a, response_b)
            }

        # Check if it's trade-off prioritization difference
        if self._is_tradeoff_difference(response_a, response_b):
            return {
                'core_disagreement': 'Different trade-off priorities',
                'disagreement_reason': 'Models prioritize different quality attributes',
                'recommended_position': self._pick_balanced_choice(response_a, response_b)
            }

        # Default: unclear disagreement
        return {
            'core_disagreement': 'Complex multi-factor disagreement',
            'disagreement_reason': 'Models disagree on multiple technical aspects',
            'recommended_position': await self._arbiter_tiebreak(response_a, response_b)
        }

    def _display_non_convergent_result(self, result: Dict[str, Any]):
        """Display non-convergent debate results"""

        print(f"""
⚠️  Models could not reach consensus after {result['rounds_completed']} rounds

Fundamental disagreement: {result['fundamental_disagreement']}

{result['model_a']} maintains: {result['position_a'][:150]}...
{result['model_b']} maintains: {result['position_b'][:150]}...

Convergence progression: {' → '.join(f"{s:.0f}%" for s in result['convergence_progression'])}

Why no convergence: {result['why_no_convergence']}

Arbiter's assessment: {result['arbiter_forced_choice'][:200]}...

💡 This disagreement itself is valuable information - it indicates a genuinely
   difficult architectural decision where reasonable experts disagree.

View full debate with --show-debate to understand both perspectives.
""")
```

## Key Implementation Notes:

### MVP 3 Highlights:

- **OpenRouter Integration**: Real-time model data without external dependencies
- **Smart Selection**: Combines OR rankings with our Elo ratings
- **Provider Diversity**: Automatic selection of complementary models
- **No New Dependencies**: Uses existing SQLite, no Redis needed
- **Graceful Fallbacks**: Works even if OR API is down (cached data)

### MVP 4 Highlights:

- **3-Layer Convergence**: Fast structural → semantic → detailed arbiter
- **Research Budgets**: Prevents excessive API costs
- **Claim Verification**: Models can fact-check each other
- **Position Evolution**: Track how recommendations change
- **Non-Convergent Insights**: Disagreement is valuable information

### Testing Commands:

```bash
# MVP 3: Test dynamic selection
archguru --type database --debate --show-selection
archguru --type api-design --tournament-size 3 --show-models

# MVP 4: Test convergence and research
archguru --type cache --debate --show-convergence
archguru --type messaging --debate --max-rounds 5 --show-debate

# Test non-convergent handling
archguru --type database --models "gpt-4" "claude-3-opus" --debate
```

### Performance Considerations:

- Semantic similarity is optional (graceful degradation if not installed)
- Arbiter calls are minimized (only when needed)
- Research budgets prevent runaway costs
- Cache prevents redundant OR API calls

## **MVP 5: Production Polish & Caching** (2-3 days)

### Day 1: SQLite-Based Debate Chain Caching

#### 1.1 Cache Schema & Manager (`src/archguru/cache/debate_cache.py`)

```python
import hashlib
import json
import sqlite3
import zlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path

class DebateCacheManager:
    """SQLite-based cache for debate chains"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or Path.home() / ".archguru" / "archguru.db"
        self._init_cache_tables()

    def _init_cache_tables(self):
        """Initialize cache tables in SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS debate_cache (
                    cache_key TEXT PRIMARY KEY,
                    decision_type TEXT NOT NULL,
                    models TEXT NOT NULL,  -- JSON array
                    context_hash TEXT NOT NULL,

                    -- Cached data (compressed)
                    debate_data BLOB NOT NULL,
                    rounds_data BLOB NOT NULL,

                    -- Metadata
                    winner TEXT,
                    final_convergence REAL,
                    total_rounds INTEGER,
                    debate_status TEXT,
                    total_cost_usd REAL,

                    -- Cache management
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1,
                    data_size_bytes INTEGER,
                    compression_ratio REAL,

                    -- Versioning
                    prompt_version TEXT,
                    cache_version TEXT DEFAULT '1.0'
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_access
                ON debate_cache(accessed_at DESC)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_type_models
                ON debate_cache(decision_type, models)
            """)

    def get_cache_key(
        self,
        decision_type: str,
        context: Dict[str, Any],
        models: List[str],
        version: str = "1.0"
    ) -> str:
        """Generate deterministic cache key for debate chain"""

        # Order models consistently for cache hits
        models_sorted = sorted(models)

        # Normalize context for consistent hashing
        context_normalized = self._normalize_context(context)

        components = [
            decision_type,
            context_normalized,
            '|'.join(models_sorted),
            version  # Prompt version to invalidate on changes
        ]

        return hashlib.sha256('\n'.join(components).encode()).hexdigest()

    def _normalize_context(self, context: Dict[str, Any]) -> str:
        """Normalize context for consistent hashing"""
        # Extract key components and sort
        normalized = {
            'language': context.get('language', '').lower().strip(),
            'framework': context.get('framework', '').lower().strip(),
            'requirements': context.get('requirements', '').lower().strip()[:500]
        }

        # Create consistent string representation
        return json.dumps(normalized, sort_keys=True)

    async def get_cached_debate(
        self,
        cache_key: str,
        max_age_days: int = 7
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached debate if available and fresh"""

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            result = conn.execute("""
                SELECT * FROM debate_cache
                WHERE cache_key = ?
                AND created_at > datetime('now', '-' || ? || ' days')
            """, (cache_key, max_age_days)).fetchone()

            if result:
                # Update access metadata
                conn.execute("""
                    UPDATE debate_cache
                    SET accessed_at = CURRENT_TIMESTAMP,
                        access_count = access_count + 1
                    WHERE cache_key = ?
                """, (cache_key,))

                # Decompress data
                debate_data = zlib.decompress(result['debate_data'])
                rounds_data = zlib.decompress(result['rounds_data'])

                return {
                    'debate': json.loads(debate_data),
                    'rounds': json.loads(rounds_data),
                    'metadata': {
                        'winner': result['winner'],
                        'final_convergence': result['final_convergence'],
                        'total_rounds': result['total_rounds'],
                        'debate_status': result['debate_status'],
                        'total_cost_usd': result['total_cost_usd'],
                        'cached_at': result['created_at'],
                        'access_count': result['access_count']
                    }
                }

        return None

    async def cache_debate(
        self,
        match: 'DebateMatch',
        result: Dict[str, Any],
        cache_ttl_days: int = 7
    ) -> bool:
        """Cache a complete debate chain"""

        try:
            # Prepare data for caching
            debate_data = {
                'id': match.id,
                'model_a': match.model_a,
                'model_b': match.model_b,
                'context': match.context,
                'max_rounds': match.max_rounds,
                'convergence_threshold': match.convergence_threshold,
                'state': match.state.value,
                'result': result
            }

            rounds_data = []
            for round in match.rounds:
                rounds_data.append({
                    'round_num': round.round_num,
                    'round_type': round.round_type,
                    'response_a': self._serialize_response(round.response_a),
                    'response_b': self._serialize_response(round.response_b),
                    'convergence_score': round.convergence_score
                })

            # Compress data
            debate_compressed = zlib.compress(
                json.dumps(debate_data).encode(),
                level=9  # Maximum compression
            )
            rounds_compressed = zlib.compress(
                json.dumps(rounds_data).encode(),
                level=9
            )

            # Calculate compression ratio
            original_size = len(json.dumps(debate_data)) + len(json.dumps(rounds_data))
            compressed_size = len(debate_compressed) + len(rounds_compressed)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1

            # Generate cache key
            cache_key = self.get_cache_key(
                match.context['decision_type'],
                match.context,
                [match.model_a, match.model_b],
                "1.0"
            )

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO debate_cache (
                        cache_key, decision_type, models, context_hash,
                        debate_data, rounds_data,
                        winner, final_convergence, total_rounds, debate_status, total_cost_usd,
                        data_size_bytes, compression_ratio, prompt_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    cache_key,
                    match.context['decision_type'],
                    json.dumps([match.model_a, match.model_b]),
                    self._normalize_context(match.context),
                    debate_compressed,
                    rounds_compressed,
                    result.get('winner'),
                    match.rounds[-1].convergence_score if match.rounds else None,
                    len(match.rounds),
                    result.get('debate_status'),
                    result.get('total_cost'),
                    compressed_size,
                    compression_ratio,
                    "1.0"
                ))

                print(f"  💾 Cached debate: {cache_key[:8]}... ({compressed_size/1024:.1f}KB, ratio: {compression_ratio:.1f}x)")
                return True

        except Exception as e:
            print(f"  ⚠️  Failed to cache debate: {e}")
            return False

    def _serialize_response(self, response: 'DebateResponse') -> Dict[str, Any]:
        """Serialize a debate response for caching"""
        return {
            'model_name': response.model_name,
            'content': response.content[:1000],  # Limit size
            'recommendation': response.recommendation,
            'confidence': response.confidence,
            'confidence_factors': response.confidence_factors.to_dict() if response.confidence_factors else None,
            'uncertainty_points': response.uncertainty_points[:5],
            'key_facts': response.key_facts[:5],
            'tool_calls_count': len(response.tool_calls)
        }
```

### Day 2: Failure Recovery & Resilience

#### 2.1 Failure Recovery System (`src/archguru/resilience/recovery.py`)

```python
from enum import Enum
from typing import Optional, Dict, Any, List
import asyncio

class FailureType(Enum):
    MODEL_TIMEOUT = "model_timeout"
    MODEL_ERROR = "model_error"
    RATE_LIMIT = "rate_limit"
    CONVERGENCE_FAILURE = "convergence_failure"
    TOOL_FAILURE = "tool_failure"

class FailureRecoveryHandler:
    """Comprehensive failure recovery for debates"""

    def __init__(self):
        self.fallback_models = {
            "openai/gpt-4o": ["openai/gpt-4o-mini", "openai/gpt-3.5-turbo"],
            "anthropic/claude-3-opus": ["anthropic/claude-3-sonnet", "anthropic/claude-3-haiku"],
            "google/gemini-pro": ["google/gemini-1.5-flash", "google/gemini-1.0-pro"]
        }

        self.retry_configs = {
            FailureType.MODEL_TIMEOUT: {'max_retries': 2, 'backoff': 2.0},
            FailureType.RATE_LIMIT: {'max_retries': 3, 'backoff': 5.0},
            FailureType.MODEL_ERROR: {'max_retries': 1, 'backoff': 1.0}
        }

    async def handle_model_failure(
        self,
        match: 'DebateMatch',
        failed_model: str,
        error: Exception,
        round_num: int
    ) -> Dict[str, Any]:
        """Handle model failure during debate"""

        failure_type = self._classify_failure(error)

        print(f"  ⚠️  {failed_model} failed: {failure_type.value}")

        # Round 0 failure - critical, need backup
        if round_num == 0:
            return await self._handle_initial_failure(
                match, failed_model, failure_type
            )

        # Mid-debate failure - try recovery strategies
        elif round_num <= 2:
            return await self._handle_mid_debate_failure(
                match, failed_model, failure_type, round_num
            )

        # Late debate failure - opponent wins
        else:
            return self._handle_late_failure(
                match, failed_model, round_num
            )

    def _classify_failure(self, error: Exception) -> FailureType:
        """Classify the type of failure"""
        error_str = str(error).lower()

        if "timeout" in error_str or "timed out" in error_str:
            return FailureType.MODEL_TIMEOUT
        elif "rate" in error_str or "429" in error_str:
            return FailureType.RATE_LIMIT
        elif "tool" in error_str or "function" in error_str:
            return FailureType.TOOL_FAILURE
        else:
            return FailureType.MODEL_ERROR

    async def _handle_initial_failure(
        self,
        match: 'DebateMatch',
        failed_model: str,
        failure_type: FailureType
    ) -> Dict[str, Any]:
        """Handle failure in initial round"""

        # Try to get a backup model
        backup = self._get_backup_model(failed_model)

        if backup:
            print(f"  🔄 Switching to backup: {backup}")

            # Update match with backup
            if match.model_a == failed_model:
                match.model_a = backup
            else:
                match.model_b = backup

            # Restart the match
            return await match.run()
        else:
            # No backup available - match fails
            return {
                'debate_status': 'failed',
                'failure_reason': f"{failed_model} failed with no backup",
                'winner': None,
                'consensus_recommendation': "Debate could not be completed due to model failure"
            }

    async def _handle_mid_debate_failure(
        self,
        match: 'DebateMatch',
        failed_model: str,
        failure_type: FailureType,
        round_num: int
    ) -> Dict[str, Any]:
        """Handle failure during debate rounds"""

        # For rate limits, wait and retry
        if failure_type == FailureType.RATE_LIMIT:
            retry_config = self.retry_configs[failure_type]

            for attempt in range(retry_config['max_retries']):
                wait_time = retry_config['backoff'] * (attempt + 1)
                print(f"  ⏳ Waiting {wait_time}s before retry {attempt + 1}...")
                await asyncio.sleep(wait_time)

                try:
                    # Try to continue from current round
                    return await match._continue_from_round(round_num)
                except Exception as e:
                    print(f"  ❌ Retry {attempt + 1} failed: {e}")

        # For other failures, opponent wins this round
        winner = match.model_a if failed_model == match.model_b else match.model_b

        return {
            'debate_status': 'partial',
            'winner': winner,
            'rounds_completed': round_num,
            'failure_in_round': round_num,
            'consensus_recommendation': f"{winner} wins by opponent failure in round {round_num}"
        }

    def _get_backup_model(self, failed_model: str) -> Optional[str]:
        """Get backup model for a failed model"""

        # Check direct fallbacks
        if failed_model in self.fallback_models:
            for backup in self.fallback_models[failed_model]:
                # Could check availability here
                return backup

        # Try to find same-provider alternative
        provider = failed_model.split('/')[0]
        for model, fallbacks in self.fallback_models.items():
            if model.startswith(provider):
                return fallbacks[0] if fallbacks else None

        # Last resort: use a default
        return "openai/gpt-3.5-turbo"

class CircuitBreaker:
    """Circuit breaker for model failures"""

    def __init__(self, failure_threshold: int = 3, reset_timeout: int = 60):
        self.failure_counts: Dict[str, int] = {}
        self.circuit_open: Dict[str, bool] = {}
        self.last_failure_time: Dict[str, float] = {}
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout

    def record_failure(self, model: str):
        """Record a model failure"""
        self.failure_counts[model] = self.failure_counts.get(model, 0) + 1
        self.last_failure_time[model] = time.time()

        if self.failure_counts[model] >= self.failure_threshold:
            print(f"  🔴 Circuit breaker OPEN for {model}")
            self.circuit_open[model] = True

    def is_available(self, model: str) -> bool:
        """Check if model is available"""
        if model not in self.circuit_open:
            return True

        if not self.circuit_open[model]:
            return True

        # Check if enough time has passed to reset
        if time.time() - self.last_failure_time.get(model, 0) > self.reset_timeout:
            print(f"  🟢 Circuit breaker RESET for {model}")
            self.circuit_open[model] = False
            self.failure_counts[model] = 0
            return True

        return False
```

### Day 3: Export Formats & Performance Dashboard

#### 3.1 Export System (`src/archguru/export/formatter.py`)

````python
from typing import Dict, Any, List
import json
from datetime import datetime

class DebateExporter:
    """Export debates in various formats"""

    def to_markdown(self, match: 'DebateMatch', result: Dict[str, Any]) -> str:
        """Export full debate transcript in Markdown"""

        md = f"""# Architectural Decision Debate

**Date**: {datetime.now().isoformat()}
**Decision Type**: {match.context.get('decision_type')}
**Models**: {match.model_a} vs {match.model_b}
**Status**: {result.get('debate_status')}
**Final Convergence**: {match.rounds[-1].convergence_score if match.rounds else 0:.0f}%

---

## Context

- **Language/Stack**: {match.context.get('language', 'Not specified')}
- **Framework**: {match.context.get('framework', 'Not specified')}
- **Requirements**: {match.context.get('requirements', 'None specified')}

---

## Debate Rounds

"""

        for round in match.rounds:
            md += f"""
### Round {round.round_num} - {round.round_type.title()}

#### {match.model_a}
**Confidence**: {round.response_a.confidence:.0%}

{round.response_a.recommendation}

<details>
<summary>Full Response</summary>

{round.response_a.content}

</details>

#### {match.model_b}
**Confidence**: {round.response_b.confidence:.0%}

{round.response_b.recommendation}

<details>
<summary>Full Response</summary>

{round.response_b.content}

</details>

**Convergence Score**: {round.convergence_score:.0f}%

---
"""

        md += f"""
## Final Result

**Winner**: {result.get('winner', 'No winner')}

**Consensus Recommendation**:
{result.get('consensus_recommendation')}

**Total Cost**: ${result.get('total_cost', 0):.4f}
"""

        return md

    def to_summary(self, match: 'DebateMatch', result: Dict[str, Any]) -> str:
        """Executive summary with key points"""

        summary = f"""ARCHITECTURAL DECISION SUMMARY
{'=' * 40}

Decision: {match.context.get('decision_type')}
Models: {match.model_a} vs {match.model_b}
Rounds: {len(match.rounds)}
Convergence: {match.rounds[-1].convergence_score if match.rounds else 0:.0f}%

WINNER: {result.get('winner', 'No consensus')}

FINAL RECOMMENDATION:
{result.get('consensus_recommendation', 'No recommendation')[:500]}

KEY INSIGHTS:
"""

        # Extract key points from debate
        for i, round in enumerate(match.rounds):
            if round.round_type == 'challenge':
                if round.response_a.uncertainty_points:
                    summary += f"- {match.model_a} identified: {round.response_a.uncertainty_points[0]}\n"
                if round.response_b.uncertainty_points:
                    summary += f"- {match.model_b} identified: {round.response_b.uncertainty_points[0]}\n"

        summary += f"""
CONVERGENCE PROGRESSION:
{' → '.join(f"{r.convergence_score:.0f}%" for r in match.rounds if hasattr(r, 'convergence_score'))}

Cost: ${result.get('total_cost', 0):.4f}
Time: {result.get('total_time', 0):.1f}s
"""

        return summary

    def to_mermaid(self, tournament: 'TournamentBracket') -> str:
        """Tournament bracket visualization in Mermaid"""

        mermaid = """```mermaid
graph TD
"""

        # Generate tournament bracket
        for i, match_result in enumerate(tournament.match_results):
            if i == 0:
                mermaid += f"""
    M1["{tournament.models[0]} vs {tournament.models[1]}"]
    M1 --> W1["{match_result['winner']}"]
"""
            else:
                mermaid += f"""
    W{i}["{tournament.match_results[i-1]['winner']}"] --> M{i+1}
    C{i+1}["{tournament.models[i+1]}"] --> M{i+1}
    M{i+1}[Match {i+1}] --> W{i+1}["{match_result['winner']}"]
"""

        mermaid += """
```"""

        return mermaid

    def to_decision_record(
        self,
        match: 'DebateMatch',
        result: Dict[str, Any]
    ) -> str:
        """Formal decision record format"""

        record = f"""ARCHITECTURE DECISION RECORD (ADR-{datetime.now().strftime('%Y%m%d')})

Title: {match.context.get('decision_type').replace('-', ' ').title()}
Date: {datetime.now().isoformat()}
Status: Decided

CONTEXT:
{match.context.get('requirements', 'General architectural decision')}

DECISION:
{result.get('consensus_recommendation')}

PARTICIPANTS:
- {match.model_a}
- {match.model_b}

DEBATE SUMMARY:
- Total Rounds: {len(match.rounds)}
- Final Convergence: {match.rounds[-1].convergence_score if match.rounds else 0:.0f}%
- Winner: {result.get('winner')}

CONSEQUENCES:
[To be documented based on implementation]

ALTERNATIVES CONSIDERED:
"""

        # Add alternative positions from debate
        for round in match.rounds:
            if round.round_type == 'initial':
                record += f"""
- {match.model_a}'s approach: {round.response_a.recommendation[:200]}
- {match.model_b}'s approach: {round.response_b.recommendation[:200]}
"""

        return record
````

#### 3.2 Performance Dashboard (`src/archguru/metrics/dashboard.py`)

```python
class PerformanceDashboard:
    """Performance metrics and monitoring dashboard"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or Path.home() / ".archguru" / "archguru.db"

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Debate performance metrics
            debate_stats = conn.execute("""
                SELECT
                    COUNT(*) as total_debates,
                    AVG(total_rounds_completed) as avg_rounds,
                    AVG(final_convergence_score) as avg_convergence,
                    AVG(total_cost_usd) as avg_cost,
                    COUNT(CASE WHEN debate_status = 'converged' THEN 1 END) as converged_count,
                    COUNT(CASE WHEN debate_status = 'forced' THEN 1 END) as forced_count,
                    COUNT(CASE WHEN debate_status = 'failed' THEN 1 END) as failed_count
                FROM debate_match
                WHERE created_at > datetime('now', '-30 days')
            """).fetchone()

            # Cache performance
            cache_stats = conn.execute("""
                SELECT
                    COUNT(*) as cache_entries,
                    SUM(access_count) as total_hits,
                    AVG(compression_ratio) as avg_compression,
                    SUM(data_size_bytes) / 1024.0 / 1024.0 as cache_size_mb
                FROM debate_cache
            """).fetchone()

            # Model performance by decision type
            model_performance = conn.execute("""
                SELECT
                    m.name as model,
                    dt.label as decision_type,
                    mr.rating as elo_rating,
                    mr.matches as matches_played,
                    COUNT(CASE WHEN dm.winner_model_id = m.id THEN 1 END) as wins
                FROM model m
                JOIN model_rating mr ON m.id = mr.model_id
                JOIN decision_type dt ON mr.decision_type_id = dt.id
                LEFT JOIN debate_match dm ON dm.model_a_id = m.id OR dm.model_b_id = m.id
                WHERE mr.algo = 'elo'
                GROUP BY m.id, dt.id
                ORDER BY mr.rating DESC
                LIMIT 20
            """).fetchall()

            return {
                'debate_performance': dict(debate_stats) if debate_stats else {},
                'cache_performance': dict(cache_stats) if cache_stats else {},
                'model_rankings': [dict(row) for row in model_performance],
                'system_health': self._calculate_health_score(debate_stats, cache_stats)
            }

    def _calculate_health_score(self, debate_stats, cache_stats) -> Dict[str, Any]:
        """Calculate system health score"""

        health = {
            'overall': 'healthy',
            'issues': []
        }

        if debate_stats:
            # Check failure rate
            total = debate_stats['total_debates']
            if total > 0:
                failure_rate = debate_stats['failed_count'] / total
                if failure_rate > 0.1:
                    health['issues'].append(f"High failure rate: {failure_rate:.0%}")
                    health['overall'] = 'degraded'

                # Check convergence rate
                convergence_rate = debate_stats['converged_count'] / total
                if convergence_rate < 0.5:
                    health['issues'].append(f"Low convergence rate: {convergence_rate:.0%}")

        if cache_stats and cache_stats['cache_entries']:
            # Check cache size
            if cache_stats['cache_size_mb'] > 100:
                health['issues'].append(f"Large cache: {cache_stats['cache_size_mb']:.1f}MB")

        return health

    def display_dashboard(self, metrics: Dict[str, Any]):
        """Display performance dashboard"""

        print("""
╔══════════════════════════════════════════════════════════╗
║            ARCHGURU PERFORMANCE DASHBOARD                 ║
╚══════════════════════════════════════════════════════════╝
""")

        # Debate Performance
        debate = metrics['debate_performance']
        print(f"""
📊 DEBATE PERFORMANCE (Last 30 days)
    Total Debates: {debate.get('total_debates', 0)}
    Avg Rounds: {debate.get('avg_rounds', 0):.1f}
    Avg Convergence: {debate.get('avg_convergence', 0):.0f}%
    Avg Cost: ${debate.get('avg_cost', 0):.4f}

    Outcomes:
    ✅ Converged: {debate.get('converged_count', 0)}
    ⚖️  Forced: {debate.get('forced_count', 0)}
    ❌ Failed: {debate.get('failed_count', 0)}
""")

        # Cache Performance
        cache = metrics['cache_performance']
        if cache.get('cache_entries'):
            hit_rate = cache.get('total_hits', 0) / max(cache.get('cache_entries', 1), 1)
            print(f"""
💾 CACHE PERFORMANCE
    Entries: {cache.get('cache_entries', 0)}
    Total Hits: {cache.get('total_hits', 0)}
    Hit Rate: {hit_rate:.1f}x per entry
    Compression: {cache.get('avg_compression', 0):.1f}x
    Cache Size: {cache.get('cache_size_mb', 0):.1f}MB
""")

        # Top Models
        if metrics['model_rankings']:
            print("\n🏆 TOP MODELS BY ELO:")
            for i, model in enumerate(metrics['model_rankings'][:5], 1):
                print(f"    {i}. {model['model'][:30]:<30} {model['elo_rating']:.0f} Elo")

        # System Health
        health = metrics['system_health']
        health_icon = "🟢" if health['overall'] == 'healthy' else "🟡"
        print(f"""
{health_icon} SYSTEM HEALTH: {health['overall'].upper()}""")

        if health['issues']:
            print("    Issues:")
            for issue in health['issues']:
                print(f"    - {issue}")
```

### CLI Integration for MVP 5

#### Enhanced CLI Commands (`src/archguru/cli/main.py` additions)

```python
@click.option('--cache/--no-cache', default=True,
              help='Use cached debates when available')
@click.option('--export', type=click.Choice(['markdown', 'summary', 'mermaid', 'adr']),
              help='Export format for results')
@click.option('--dashboard', is_flag=True,
              help='Show performance dashboard')
@click.option('--clear-cache', is_flag=True,
              help='Clear debate cache')
async def archguru_main(..., cache, export, dashboard, clear_cache):
    """Enhanced CLI with caching and export"""

    if dashboard:
        # Show performance dashboard
        dashboard = PerformanceDashboard()
        metrics = await dashboard.get_performance_metrics()
        dashboard.display_dashboard(metrics)
        return 0

    if clear_cache:
        # Clear cache with confirmation
        if click.confirm("Clear all cached debates?"):
            cache_manager = DebateCacheManager()
            await cache_manager.clear_cache()
            print("✅ Cache cleared")
        return 0

    # Check cache before running debate
    if cache and debate:
        cache_manager = DebateCacheManager()
        cache_key = cache_manager.get_cache_key(
            type, context, models, "1.0"
        )

        cached = await cache_manager.get_cached_debate(cache_key)
        if cached:
            print(f"📦 Using cached debate (accessed {cached['metadata']['access_count']} times)")
            result = cached['debate']['result']

            # Display cached results
            display_debate_results(result)

            if export:
                await export_results(cached['debate'], result, export)

            return 0

    # Run new debate
    match = DebateMatch(...)
    result = await match.run()

    # Cache the result
    if cache:
        await cache_manager.cache_debate(match, result)

    # Export if requested
    if export:
        exporter = DebateExporter()

        if export == 'markdown':
            output = exporter.to_markdown(match, result)
            filename = f"debate_{match.id[:8]}.md"
        elif export == 'summary':
            output = exporter.to_summary(match, result)
            filename = f"summary_{match.id[:8]}.txt"
        elif export == 'adr':
            output = exporter.to_decision_record(match, result)
            filename = f"adr_{datetime.now().strftime('%Y%m%d')}.md"

        with open(filename, 'w') as f:
            f.write(output)

        print(f"📄 Exported to {filename}")
```

## Key Implementation Notes for MVP 5:

### Caching Strategy:

- **SQLite-only**: No Redis dependency, keeping it simple
- **Compression**: zlib achieves ~5-10x compression for debate data
- **Smart Keys**: Deterministic cache keys for consistent hits
- **TTL Management**: 7-day default with access tracking
- **Size Management**: Can implement cache eviction if needed

### Resilience Features:

- **Circuit Breaker**: Prevents cascade failures
- **Fallback Models**: Automatic substitution on failure
- **Retry Logic**: Exponential backoff for rate limits
- **Graceful Degradation**: Partial results better than none

### Export Capabilities:

- **Markdown**: Full debate transcript with collapsible sections
- **Summary**: Executive summary for quick review
- **Mermaid**: Visual tournament brackets
- **ADR**: Formal decision records

### Performance Monitoring:

- **Real-time Metrics**: Debate success rates, convergence stats
- **Cache Analytics**: Hit rates, compression ratios
- **Model Rankings**: Elo-based leaderboards
- **System Health**: Automatic issue detection

### Testing Commands:

```bash
# Test caching
archguru --type database --debate --models "gpt-4o-mini" "claude-3-haiku"
archguru --type database --debate --models "gpt-4o-mini" "claude-3-haiku"  # Should hit cache

# Test exports
archguru --type api-design --tournament --export markdown
archguru --type cache --debate --export summary

# Performance monitoring
archguru --dashboard
archguru --stats  # Existing stats command

# Cache management
archguru --clear-cache
archguru --type database --debate --no-cache  # Skip cache
```

## Complete MVP Roadmap Summary:

1. **MVP 1**: Core 2-model debate engine with full DB schema ✅
2. **MVP 2**: Tournament brackets for 3+ models ✅
3. **MVP 3**: OpenRouter intelligence for dynamic selection ✅
4. **MVP 4**: Advanced convergence & research budgets ✅
5. **MVP 5**: Production polish with caching & exports ✅

The system is now complete with:

- No Docker/Redis dependencies (pure SQLite)
- Progressive enhancement (each MVP builds on previous)
- Backward compatibility maintained throughout
- Production-ready features (caching, exports, monitoring)
- Graceful failure handling at every level
