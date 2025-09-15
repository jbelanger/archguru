# ADR-002: Advanced Evaluation Algorithms

## Status
Proposed (September 2025)

## Context
ADR-001 establishes individual LLM competition with foundational evaluation infrastructure including analytics tables, judge protocols, structured rationale schemas, and basic orchestration policy. This ADR specifies the advanced algorithms for model skill estimation, intelligent selection, and behavioral profiling that leverage the established infrastructure to optimize model performance and selection.

## Decision

### Advanced Evaluation Components

* **TrueSkill Model Rating**: Hierarchical skill estimation with decision-type specialization
* **Contextual Bandit Selection**: Cache-aware model selection with exploration guarantees
* **Model Personality Profiling**: Behavioral axes derived from decision patterns
* **Hierarchical Preference Calibration**: Cross-type score normalization with anchors
* **Cross-Domain Bridge Analysis**: Multi-criteria decision analysis for spanning decisions

---

## 1. TrueSkill Model Rating System

### Hierarchical Skill Estimation

Implement TrueSkill with both global and decision-type-specific ratings to capture model specialization patterns.

**Core Algorithm:**
```python
class HierarchicalTrueSkill:
    def __init__(self):
        self.mu_0 = 25.0      # Initial skill mean
        self.sigma_0 = 8.33   # Initial skill variance
        self.beta = 4.17      # Performance variance
        self.tau = 0.08       # Skill change over time
        self.draw_prob = 0.10
        self.decay_factor = 0.98  # Weekly decay

    def update_skills(self, outcome, global_skills, type_skills):
        """Update both global and type-specific skills"""
        # Global skill update (30% weight)
        global_delta = self.compute_skill_delta(outcome, global_skills)

        # Type-specific skill update (70% weight)
        type_delta = self.compute_skill_delta(outcome, type_skills)

        # Hierarchical combination
        final_delta = 0.3 * global_delta + 0.7 * type_delta
        return final_delta
```

**Weekly Ranking Protocol:**
```python
def weekly_rerank():
    """Monday 00:00 UTC: Batch recomputation for production defaults"""
    for decision_type in DECISION_TYPES:
        # Apply decay to historical results
        apply_skill_decay(decision_type, decay_factor=0.98)

        # Recompute rankings
        rankings = compute_trueskill_rankings(decision_type)

        # Update production defaults (frozen for week)
        update_production_presets(decision_type, rankings)

        # Log ranking changes for analysis
        log_ranking_stability(decision_type, rankings)

def update_skills_online(outcome):
    """Real-time updates for dashboards only"""
    update_skills_incremental(outcome)
    refresh_telemetry_only()  # Don't affect production selection
```

---

## 2. Contextual Bandit Model Selection

### Cache-Aware Selection with Exploration Guarantees

Implement LinUCB with contextual features from cache similarity, decision type, and historical performance.

**Core Algorithm:**
```python
class ContextualBanditSelector:
    def __init__(self, exploration_floor=0.1):
        self.exploration_floor = exploration_floor
        self.confidence_multiplier = 1.96  # 95% confidence bounds
        self.max_exploitation_streak = 5
        self.theta = {}  # Model parameter vectors
        self.A = {}      # Covariance matrices

    def select_models(self, context, n_models=2):
        """Select models using Upper Confidence Bound"""
        features = self.build_context_features(context)

        # Force exploration conditions
        if self.should_explore(context):
            return self.explore_models(n_models)

        # Exploitation with confidence bounds
        model_scores = []
        for model in self.available_models:
            mu = np.dot(self.theta[model], features)
            sigma = self.compute_confidence_bound(model, features)
            ucb_score = mu + self.confidence_multiplier * sigma
            model_scores.append((model, ucb_score))

        return self.top_k_models(model_scores, n_models)

    def build_context_features(self, context):
        """Extract features for bandit decision"""
        return np.concatenate([
            self.decision_type_onehot(context.decision_type),
            self.tech_stack_embedding(context.tech_stack),
            self.cache_similarity_features(context),
            self.complexity_features(context),
            self.stakeholder_features(context)
        ])

    def cache_similarity_features(self, context):
        """Leverage SQLite cache for similarity"""
        similar_contexts = self.query_cache_similarity(
            context.decision_type,
            context.tech_stack,
            limit=10
        )

        return np.array([
            len(similar_contexts) / 10,  # Cache hit rate
            np.mean([c.arbiter_confidence for c in similar_contexts]),
            np.std([c.model_agreement for c in similar_contexts])
        ])
```

**Exploration Strategy:**
```python
def should_explore(self, context):
    """Determine when to force exploration"""
    return any([
        context.novelty_score > 0.8,         # Novel context
        context.user_stakes == "high",       # High stakes
        np.random.random() < self.exploration_floor,  # Random exploration
        self.max_exploitation_streak_reached(),
        self.model_starvation_detected()      # Unused models
    ])

def update_model_performance(self, outcome, context, selected_models):
    """Update bandit parameters based on outcome"""
    features = self.build_context_features(context)
    reward = self.compute_reward(outcome)  # From arbiter selection + user feedback

    for model in selected_models:
        # Update parameter estimates
        self.A[model] += np.outer(features, features)
        self.b[model] += reward * features
        self.theta[model] = np.linalg.solve(self.A[model], self.b[model])
```

---

## 3. Model Personality Profiling

### Behavioral Axis Extraction

Derive stable personality dimensions from historical decision patterns to enable model selection insights.

**Core Personality Axes:**
```python
class PersonalityProfiler:
    def __init__(self):
        self.axes = {
            'risk_tolerance': {
                'signals': ['serverless_frequency', 'bleeding_edge_adoption', 'fallback_mentions'],
                'range': [0, 1],  # 0=conservative, 1=aggressive
                'weight': 0.25
            },
            'complexity_preference': {
                'signals': ['microservices_bias', 'abstraction_layers', 'tool_diversity'],
                'range': [0, 1],  # 0=simple, 1=sophisticated
                'weight': 0.25
            },
            'vendor_neutrality': {
                'signals': ['aws_mentions', 'open_source_ratio', 'vendor_warnings'],
                'range': [0, 1],  # 0=opinionated, 1=agnostic
                'weight': 0.20
            },
            'evidence_style': {
                'signals': ['benchmark_ratio', 'citation_depth', 'quantitative_focus'],
                'range': [0, 1],  # 0=anecdotal, 1=rigorous
                'weight': 0.20
            },
            'innovation_bias': {
                'signals': ['new_tech_adoption', 'experimental_mentions', 'stability_warnings'],
                'range': [0, 1],  # 0=stable, 1=cutting-edge
                'weight': 0.10
            }
        }
        self.ema_alpha = 0.1  # Slow updates for stability

    def extract_signals(self, decision_outcome):
        """Extract behavioral signals from model decisions"""
        signals = {}

        # Parse recommendation text for patterns
        text = decision_outcome.rationale + decision_outcome.recommendation

        # Risk tolerance signals
        signals['serverless_frequency'] = len(re.findall(r'serverless|lambda|faas', text, re.I)) / len(text.split())
        signals['bleeding_edge_adoption'] = self.score_technology_recency(decision_outcome.technologies)
        signals['fallback_mentions'] = len(re.findall(r'fallback|backup|contingency', text, re.I))

        # Complexity preference signals
        signals['microservices_bias'] = 1.0 if 'microservice' in text.lower() else 0.0
        signals['abstraction_layers'] = self.count_abstraction_mentions(text)
        signals['tool_diversity'] = len(set(decision_outcome.recommended_tools))

        # Evidence style signals
        signals['benchmark_ratio'] = len(decision_outcome.benchmark_citations) / max(1, len(decision_outcome.all_citations))
        signals['citation_depth'] = len(decision_outcome.all_citations) / len(text.split())

        return signals

    def update_personality(self, model_id, decision_outcome):
        """Update personality profile via exponential moving average"""
        signals = self.extract_signals(decision_outcome)

        for axis_name, axis_config in self.axes.items():
            # Compute axis score from signals
            axis_score = self.compute_axis_score(signals, axis_config['signals'])

            # EMA update for stability
            current_score = self.profiles.get(model_id, {}).get(axis_name, 0.5)
            new_score = (1 - self.ema_alpha) * current_score + self.ema_alpha * axis_score

            # Update profile
            if model_id not in self.profiles:
                self.profiles[model_id] = {}
            self.profiles[model_id][axis_name] = new_score

            # Track confidence (based on number of observations)
            confidence_key = f"{axis_name}_confidence"
            obs_count = self.observation_counts.get(model_id, {}).get(axis_name, 0) + 1
            confidence = min(1.0, obs_count / 50)  # Full confidence after 50 observations
            self.profiles[model_id][confidence_key] = confidence
```

**Personality Visualization:**
```python
def generate_radar_card(self, model_id):
    """Generate radar chart for model personality"""
    profile = self.profiles.get(model_id, {})

    # Only show axes with sufficient confidence
    display_axes = {
        axis: score for axis, score in profile.items()
        if not axis.endswith('_confidence') and
           profile.get(f"{axis}_confidence", 0) >= 0.6
    }

    return {
        'model': model_id,
        'personality': display_axes,
        'summary': self.generate_personality_summary(display_axes),
        'confidence': np.mean([profile.get(f"{axis}_confidence", 0) for axis in display_axes.keys()])
    }

def generate_personality_summary(self, axes):
    """Generate human-readable personality summary"""
    traits = []

    if axes.get('risk_tolerance', 0.5) < 0.3:
        traits.append("conservative")
    elif axes.get('risk_tolerance', 0.5) > 0.7:
        traits.append("risk-taking")

    if axes.get('evidence_style', 0.5) > 0.7:
        traits.append("data-driven")
    elif axes.get('evidence_style', 0.5) < 0.3:
        traits.append("intuitive")

    if axes.get('complexity_preference', 0.5) > 0.7:
        traits.append("architecture-heavy")
    elif axes.get('complexity_preference', 0.5) < 0.3:
        traits.append("simplicity-focused")

    return ", ".join(traits) if traits else "balanced approach"
```

---

## 4. Hierarchical Preference Calibration

### Cross-Type Score Normalization

Ensure model ratings remain comparable across decision types while capturing specialization.

**Bradley-Terry Hierarchical Model:**
```python
class HierarchicalPreferenceModel:
    def __init__(self):
        self.global_skills = {}     # μᵢ - global model skill
        self.type_adjustments = {}  # αᵢ,ₖ - type-specific adjustments
        self.judge_biases = {}      # bⱼ - per-judge bias correction

    def preference_probability(self, model_i, model_j, decision_type, judge):
        """P(i ≻ j | decision_type, judge)"""
        global_diff = self.global_skills[model_i] - self.global_skills[model_j]
        type_diff = (self.type_adjustments[(model_i, decision_type)] -
                    self.type_adjustments[(model_j, decision_type)])
        judge_bias = self.judge_biases.get(judge, 0)

        logit = global_diff + type_diff + judge_bias
        return 1 / (1 + np.exp(-logit))

    def update_from_judgment(self, winner, loser, decision_type, judge, draw=False):
        """Update parameters from pairwise judgment"""
        if draw:
            # Handle draws with margin
            draw_margin = 0.1
            outcome = 0.5  # Equal weight to both models
        else:
            outcome = 1.0 if winner else 0.0

        # Gradient-based updates
        predicted = self.preference_probability(winner, loser, decision_type, judge)
        error = outcome - predicted

        # Update global skills
        learning_rate = 0.01
        self.global_skills[winner] += learning_rate * error
        self.global_skills[loser] -= learning_rate * error

        # Update type-specific adjustments (with L2 regularization)
        regularization = 0.001
        type_update = learning_rate * error - regularization * self.type_adjustments.get((winner, decision_type), 0)
        self.type_adjustments[(winner, decision_type)] = self.type_adjustments.get((winner, decision_type), 0) + type_update
        self.type_adjustments[(loser, decision_type)] = self.type_adjustments.get((loser, decision_type), 0) - type_update
```

**Anchor Set Management:**
```python
class AnchorSetManager:
    def __init__(self):
        self.gold_sets = {}         # Per-type anchors
        self.cross_type_anchors = []  # Universal anchors
        self.refresh_interval = 30    # Days

    def validate_anchor_stability(self):
        """Ensure anchor scores remain stable over time"""
        for decision_type, anchors in self.gold_sets.items():
            stability_scores = []
            for anchor in anchors:
                recent_scores = self.get_recent_anchor_scores(anchor, days=7)
                historical_scores = self.get_historical_anchor_scores(anchor, days=30)
                stability = 1 - np.std(recent_scores) / np.std(historical_scores)
                stability_scores.append(stability)

            avg_stability = np.mean(stability_scores)
            if avg_stability < 0.8:  # 80% stability threshold
                self.refresh_anchor_set(decision_type)

    def calibrate_judges(self):
        """Normalize judge biases via z-scoring"""
        for judge in self.judge_biases.keys():
            judge_scores = self.get_judge_score_history(judge, days=30)
            if len(judge_scores) >= 10:  # Minimum observations
                z_scored = (judge_scores - np.mean(judge_scores)) / np.std(judge_scores)
                self.judge_biases[judge] = np.mean(z_scored)
```

---

## 5. Cross-Domain Bridge Analysis

### Multi-Criteria Decision Analysis for Spanning Decisions

Handle architectural decisions that span multiple domains with systematic conflict resolution.

**Bridge Analysis Algorithm:**
```python
class CrossDomainBridge:
    def __init__(self):
        self.criteria_weights = {
            'development_speed': 0.25,
            'operational_complexity': 0.20,
            'scalability': 0.20,
            'security': 0.15,
            'cost': 0.10,
            'team_expertise': 0.10
        }

    def analyze_cross_impacts(self, sub_decisions):
        """Identify conflicts between sub-decision recommendations"""
        conflicts = []
        synergies = []

        for i, decision_a in enumerate(sub_decisions):
            for j, decision_b in enumerate(sub_decisions[i+1:], i+1):
                impact = self.compute_cross_impact(decision_a, decision_b)

                if impact['conflict_score'] > 0.5:
                    conflicts.append({
                        'domains': (decision_a['type'], decision_b['type']),
                        'description': impact['description'],
                        'severity': impact['conflict_score'],
                        'mitigation': impact['suggested_mitigation']
                    })
                elif impact['synergy_score'] > 0.7:
                    synergies.append({
                        'domains': (decision_a['type'], decision_b['type']),
                        'description': impact['description'],
                        'benefit': impact['synergy_score']
                    })

        return {'conflicts': conflicts, 'synergies': synergies}

    def compute_cross_impact(self, decision_a, decision_b):
        """Analyze interaction between two sub-decisions"""
        # Pattern matching for known conflict/synergy patterns
        patterns = {
            ('authentication', 'api_design'): {
                'jwt + graphql': {'conflict': 0.3, 'description': 'Complex token management'},
                'oauth + rest': {'synergy': 0.8, 'description': 'Standard integration pattern'}
            },
            ('database', 'deployment'): {
                'mongodb + kubernetes': {'synergy': 0.7, 'description': 'Cloud-native stack'},
                'postgresql + serverless': {'conflict': 0.6, 'description': 'Connection pooling challenges'}
            }
        }

        # Extract technology combinations
        tech_a = decision_a['recommended_technologies']
        tech_b = decision_b['recommended_technologies']

        # Look up known patterns
        domain_pair = (decision_a['type'], decision_b['type'])
        if domain_pair in patterns:
            for pattern, impact in patterns[domain_pair].items():
                if self.pattern_matches(pattern, tech_a + tech_b):
                    return {
                        'conflict_score': impact.get('conflict', 0),
                        'synergy_score': impact.get('synergy', 0),
                        'description': impact['description'],
                        'suggested_mitigation': self.suggest_mitigation(pattern)
                    }

        # Default neutral impact
        return {'conflict_score': 0, 'synergy_score': 0, 'description': 'No significant interaction'}

    def reconcile_with_mcda(self, sub_decisions, cross_impacts):
        """Apply multi-criteria decision analysis for final reconciliation"""
        options = self.generate_solution_combinations(sub_decisions)

        # Score each option across criteria
        option_scores = {}
        for option_id, option in options.items():
            scores = {}

            # Base scores from individual decisions
            scores['development_speed'] = np.mean([d['speed_score'] for d in option['decisions']])
            scores['operational_complexity'] = 1 - np.mean([d['complexity_score'] for d in option['decisions']])
            scores['scalability'] = np.mean([d['scalability_score'] for d in option['decisions']])

            # Adjust for cross-impacts
            conflict_penalty = sum(c['severity'] for c in cross_impacts['conflicts'] if self.affects_option(c, option))
            synergy_bonus = sum(s['benefit'] for s in cross_impacts['synergies'] if self.affects_option(s, option))

            # Apply MCDA weighted scoring
            weighted_score = sum(
                self.criteria_weights[criterion] * (score - conflict_penalty + synergy_bonus)
                for criterion, score in scores.items()
            )

            option_scores[option_id] = {
                'total_score': weighted_score,
                'detailed_scores': scores,
                'conflicts': [c for c in cross_impacts['conflicts'] if self.affects_option(c, option)],
                'synergies': [s for s in cross_impacts['synergies'] if self.affects_option(s, option)]
            }

        # Return ranked options
        return sorted(option_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
```

---

## 6. Implementation Integration

### Algorithm Coordination

Integrate all advanced evaluation components within the existing ADR-001 infrastructure.

**Unified Evaluation Pipeline:**
```python
class AdvancedEvaluationOrchestrator:
    def __init__(self):
        self.trueskill = HierarchicalTrueSkill()
        self.bandit = ContextualBanditSelector()
        self.profiler = PersonalityProfiler()
        self.bridge = CrossDomainBridge()
        self.calibrator = HierarchicalPreferenceModel()

    def process_decision_outcome(self, outcome):
        """Comprehensive outcome processing"""
        # Update TrueSkill ratings
        self.trueskill.update_skills(
            outcome.arbiter_selection,
            outcome.decision_type
        )

        # Update contextual bandit
        self.bandit.update_model_performance(
            outcome,
            outcome.context,
            outcome.selected_models
        )

        # Update personality profiles
        for model_result in outcome.model_results:
            self.profiler.update_personality(
                model_result.model_id,
                model_result
            )

        # Update preference calibration
        for judgment in outcome.pairwise_judgments:
            self.calibrator.update_from_judgment(
                judgment.winner,
                judgment.loser,
                judgment.decision_type,
                judgment.judge
            )

    def select_models_for_decision(self, context):
        """Intelligent model selection using all algorithms"""
        # Get base selection from contextual bandit
        base_models = self.bandit.select_models(context, n_models=2)

        # Check if escalation needed (from ADR-001 triggers)
        if self.should_escalate(context, base_models):
            # Use TrueSkill rankings for additional models
            top_specialists = self.trueskill.get_top_models_for_type(
                context.decision_type,
                exclude=base_models,
                n=2
            )
            return base_models + top_specialists

        return base_models

    def generate_evaluation_report(self, model_id):
        """Comprehensive model evaluation report"""
        return {
            'trueskill_ratings': self.trueskill.get_model_ratings(model_id),
            'specialization_areas': self.trueskill.get_specializations(model_id),
            'personality_profile': self.profiler.generate_radar_card(model_id),
            'selection_frequency': self.bandit.get_selection_stats(model_id),
            'performance_trends': self.get_performance_trends(model_id)
        }
```

**Weekly Maintenance Tasks:**
```python
def weekly_evaluation_maintenance():
    """Monday 00:00 UTC maintenance tasks"""
    # TrueSkill re-ranking (from ADR-001)
    orchestrator.trueskill.weekly_rerank()

    # Anchor set validation
    orchestrator.calibrator.anchor_manager.validate_anchor_stability()

    # Judge bias recalibration
    orchestrator.calibrator.calibrate_judges()

    # Personality profile confidence updates
    orchestrator.profiler.update_confidence_scores()

    # Bandit exploration parameter tuning
    orchestrator.bandit.tune_exploration_parameters()

    # Generate weekly analytics report
    generate_weekly_evaluation_report()
```

---

## Algorithm Parameters

### Production Configuration
```yaml
# config/advanced_evaluation.yaml
trueskill:
  mu_0: 25.0
  sigma_0: 8.33
  beta: 4.17
  tau: 0.08
  draw_probability: 0.10
  decay_factor: 0.98
  hierarchical_weight_global: 0.3
  hierarchical_weight_type: 0.7

contextual_bandit:
  exploration_floor: 0.1
  confidence_multiplier: 1.96
  max_exploitation_streak: 5
  model_starvation_days: 7
  feature_dimensions:
    decision_type_onehot: 8
    tech_stack_embedding: 16
    cache_similarity: 3
    complexity_features: 4

personality_profiling:
  ema_alpha: 0.1
  confidence_threshold: 0.6
  min_observations: 50
  update_frequency: "weekly"
  axes:
    risk_tolerance: 0.25
    complexity_preference: 0.25
    vendor_neutrality: 0.20
    evidence_style: 0.20
    innovation_bias: 0.10

preference_calibration:
  learning_rate: 0.01
  regularization: 0.001
  draw_margin: 0.1
  judge_rotation_size: 3
  anchor_refresh_days: 30
  stability_threshold: 0.8

cross_domain_bridge:
  criteria_weights:
    development_speed: 0.25
    operational_complexity: 0.20
    scalability: 0.20
    security: 0.15
    cost: 0.10
    team_expertise: 0.10
  conflict_threshold: 0.5
  synergy_threshold: 0.7
```

### Extended Database Schema
```sql
-- Additional tables for advanced algorithms
CREATE TABLE contextual_features (
    id INTEGER PRIMARY KEY,
    decision_id TEXT,
    feature_vector TEXT, -- JSON blob
    selected_models TEXT, -- JSON array
    outcome_reward REAL,
    created_at TIMESTAMP
);

CREATE TABLE personality_signals (
    id INTEGER PRIMARY KEY,
    model_id TEXT,
    decision_id TEXT,
    signal_name TEXT,
    signal_value REAL,
    axis_contribution REAL,
    created_at TIMESTAMP
);

CREATE TABLE anchor_performance (
    id INTEGER PRIMARY KEY,
    anchor_id TEXT,
    decision_type TEXT,
    model_id TEXT,
    score REAL,
    stability_score REAL,
    created_at TIMESTAMP
);

CREATE TABLE cross_domain_analysis (
    id INTEGER PRIMARY KEY,
    decision_id TEXT,
    sub_decisions TEXT, -- JSON array
    conflicts TEXT,     -- JSON array
    synergies TEXT,     -- JSON array
    mcda_scores TEXT,   -- JSON object
    created_at TIMESTAMP
);
```

---

## Success Metrics

### Algorithm Performance
- **TrueSkill prediction accuracy**: Rankings predict arbiter selection >70% of time
- **Contextual bandit regret**: <5% suboptimal selections over 1000 decisions
- **Personality profile stability**: Axes change <0.05 per week after convergence
- **Cross-type calibration**: Pearson r > 0.75 between global and type-specific rankings

### System Efficiency
- **Intelligent selection impact**: 20-30% cost reduction vs random model selection
- **Exploration balance**: 10-15% exploration rate with <2% model starvation
- **Cross-domain bridge accuracy**: >80% conflict detection for known problematic combinations
- **Anchor stability**: Weekly score variance < 0.03 for established anchors

### Model Discovery
- **Specialization identification**: Detect 3+ distinct specializations per decision type
- **Personality differentiation**: Clear axis separation (>0.3 difference) between models
- **Selection optimization**: Reduce time-to-optimal-model by 40% vs naive rotation
- **Judge calibration**: Inter-judge agreement (Krippendorff's α) > 0.65

## Implementation Priority

1. **TrueSkill hierarchical rating system** (builds on ADR-001 analytics tables)
2. **Contextual bandit model selection** (integrates with existing escalation triggers)
3. **Personality profiling infrastructure** (leverages structured rationale schema)
4. **Preference calibration framework** (uses existing pairwise judgment tables)
5. **Cross-domain bridge analysis** (optional extension for spanning decisions)
6. **Algorithm integration and orchestration** (unified evaluation pipeline)

## Open Implementation Questions

- How to balance global vs type-specific skill components in TrueSkill hierarchy?
- What's the optimal contextual feature engineering for tech stack embeddings?
- Should personality profiles influence model presets or remain purely analytical?
- How to handle concept drift in model behavior over time?
- What's the minimum observation count for reliable personality axis estimation?
- How to detect and prevent gaming of evaluation metrics by model providers?