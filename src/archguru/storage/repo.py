"""
ArchGuru v0.3 - Persistence Repository
Single persistence hook for writing run results to SQLite
"""

import sqlite3
import uuid
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from .elo import update_elo_ratings_for_run, get_elo_stats_by_decision_type

@dataclass
class RunResult:
    """Result from a complete ArchGuru run"""
    decision_type: str
    language: Optional[str]
    framework: Optional[str]
    requirements: Optional[str]
    model_responses: List[Dict[str, Any]]
    arbiter_model: str
    consensus_recommendation: Optional[str]
    debate_summary: Optional[str]
    total_time_sec: float
    winning_model: Optional[str] = None  # v0.4: Track winner for Elo updates
    winner_source: Optional[str] = None  # v0.4: Track selection method (arbiter/fallback)
    error: Optional[str] = None


class ArchGuruRepo:
    """SQLite repository for ArchGuru persistence"""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = Path.home() / ".archguru" / "archguru.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database with schema"""
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path, 'r') as f:
            schema = f.read()

        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(schema)

    def _get_or_create_model(self, conn: sqlite3.Connection, model_name: str) -> int:
        """Get or create model record, return model_id"""
        provider = model_name.split('/')[0] if '/' in model_name else None

        # Try to get existing
        cursor = conn.execute("SELECT id FROM model WHERE name = ?", (model_name,))
        result = cursor.fetchone()
        if result:
            return result[0]

        # Create new
        cursor = conn.execute(
            "INSERT INTO model (name, provider) VALUES (?, ?)",
            (model_name, provider)
        )
        return cursor.lastrowid

    def _get_decision_type_id(self, conn: sqlite3.Connection, decision_type: str) -> int:
        """Get decision_type_id, create if doesn't exist"""
        cursor = conn.execute("SELECT id FROM decision_type WHERE key = ?", (decision_type,))
        result = cursor.fetchone()
        if result:
            return result[0]

        # Create new decision type
        cursor = conn.execute(
            "INSERT INTO decision_type (key, label) VALUES (?, ?)",
            (decision_type, decision_type.replace('-', ' ').title())
        )
        return cursor.lastrowid

    def persist_run_result(
        self,
        result: RunResult,
        prompt_version: str = "1.0"
    ) -> str:
        """
        Persist a complete run result to SQLite
        Returns the run_id
        """
        run_id = str(uuid.uuid4())

        with sqlite3.connect(self.db_path) as conn:
            # Get/create related records
            decision_type_id = self._get_decision_type_id(conn, result.decision_type)
            arbiter_model_id = self._get_or_create_model(conn, result.arbiter_model)

            # Insert run record
            conn.execute("""
                INSERT INTO run (
                    id, decision_type_id, language, framework, requirements,
                    prompt_version, arbiter_model_id, consensus_reco, debate_summary,
                    total_time_sec, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, decision_type_id, result.language, result.framework,
                result.requirements, prompt_version, arbiter_model_id,
                result.consensus_recommendation, result.debate_summary,
                result.total_time_sec, result.error
            ))

            # Insert model responses
            for response in result.model_responses:
                model_id = self._get_or_create_model(conn, response['model'])
                response_id = str(uuid.uuid4())

                conn.execute("""
                    INSERT INTO model_response (
                        id, run_id, model_id, team, recommendation, reasoning,
                        trade_offs, confidence_score, response_time_sec, success, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    response_id, run_id, model_id, response.get('team'),
                    response.get('recommendation'), response.get('reasoning'),
                    json.dumps(response.get('trade_offs', [])),
                    response.get('confidence_score'), response.get('response_time_sec'),
                    response.get('success', True), response.get('error')
                ))

                # Insert tool calls if present
                for tool_call in response.get('tool_calls', []):
                    conn.execute("""
                        INSERT INTO tool_call (response_id, function, arguments, result_excerpt)
                        VALUES (?, ?, ?, ?)
                    """, (
                        response_id, tool_call.get('function'),
                        json.dumps(tool_call.get('arguments', {})),
                        tool_call.get('result_excerpt')
                    ))

            # v0.4: Update Elo ratings if we have a valid winner
            if result.winning_model:
                # Ensure winner is one of the successful participating models
                successful_models = [r.get('model') for r in result.model_responses
                                   if r.get('success', True) and r.get('model')]
                if result.winning_model in successful_models:
                    try:
                        self._update_elo_ratings_for_winner(
                            conn, run_id, result.winning_model,
                            result.model_responses, decision_type_id, arbiter_model_id,
                            result.winner_source
                        )
                    except Exception as e:
                        print(f"Warning: Failed to update Elo ratings: {str(e)}")
                else:
                    print("ℹ️  Skipping Elo: no valid winner in this run")

        return run_id

    def _update_elo_ratings_for_winner(
        self,
        conn: sqlite3.Connection,
        run_id: str,
        winning_model_name: str,
        model_responses: List[Dict[str, Any]],
        decision_type_id: int,
        judge_model_id: int,
        winner_source: Optional[str] = None
    ):
        """Update Elo ratings when arbiter selects a winner"""
        # Get winner model ID
        winner_model_id = self._get_or_create_model(conn, winning_model_name)

        # Get all other model IDs that participated successfully
        loser_model_ids = []
        for response in model_responses:
            model_name = response.get('model')
            success = response.get('success', True)
            if model_name and success and model_name != winning_model_name:
                loser_id = self._get_or_create_model(conn, model_name)
                loser_model_ids.append(loser_id)

        if loser_model_ids:
            # Set reason based on winner source
            reason = "Arbiter selection" if winner_source == "arbiter" else "Fallback scoring"

            # Update Elo ratings for all pairwise comparisons
            updates = update_elo_ratings_for_run(
                conn, run_id, winner_model_id, loser_model_ids,
                decision_type_id, judge_model_id, reason
            )
            print(f"  📊 Updated Elo ratings: {len(updates)} pairwise comparisons ({reason})")

    def get_stats(self) -> Dict[str, Any]:
        """Get basic statistics for --stats command"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Basic counts
            total_runs = conn.execute("SELECT COUNT(*) FROM run").fetchone()[0]

            # Average latency
            avg_latency = conn.execute(
                "SELECT AVG(total_time_sec) FROM run WHERE total_time_sec IS NOT NULL"
            ).fetchone()[0] or 0

            # Decision type breakdown
            decision_types = conn.execute("""
                SELECT dt.label, COUNT(*) as count
                FROM run r
                JOIN decision_type dt ON r.decision_type_id = dt.id
                GROUP BY dt.id, dt.label
                ORDER BY count DESC
            """).fetchall()

            # Model usage
            model_usage = conn.execute("""
                SELECT m.name, COUNT(*) as responses
                FROM model_response mr
                JOIN model m ON mr.model_id = m.id
                GROUP BY m.id, m.name
                ORDER BY responses DESC
            """).fetchall()

            # Recent activity (last 7 days)
            recent_runs = conn.execute("""
                SELECT COUNT(*) FROM run
                WHERE created_at >= datetime('now', '-7 days')
            """).fetchone()[0]

            # v0.4: Get Elo rankings by decision type
            elo_rankings = get_elo_stats_by_decision_type(conn)

            return {
                'total_runs': total_runs,
                'avg_latency_sec': round(avg_latency, 2),
                'recent_runs_7d': recent_runs,
                'decision_types': [dict(row) for row in decision_types],
                'model_usage': [dict(row) for row in model_usage],
                'elo_rankings': elo_rankings  # v0.4: Top 5 by Elo per decision type
            }


# Simple integration function for pipeline
def persist_pipeline_result(
    decision_type: str,
    language: Optional[str],
    framework: Optional[str],
    requirements: Optional[str],
    model_responses: List[Dict[str, Any]],
    arbiter_model: str,
    consensus_recommendation: Optional[str],
    debate_summary: Optional[str],
    total_time_sec: float,
    winning_model: Optional[str] = None,  # v0.4: Track winner for Elo
    winner_source: Optional[str] = None,  # v0.4: Track selection method
    prompt_version: str = "1.0",
    db_path: Optional[str] = None
) -> str:
    """Simple persistence function for pipeline integration"""

    run_result = RunResult(
        decision_type=decision_type,
        language=language,
        framework=framework,
        requirements=requirements,
        model_responses=model_responses,
        arbiter_model=arbiter_model,
        consensus_recommendation=consensus_recommendation,
        debate_summary=debate_summary,
        total_time_sec=total_time_sec,
        winning_model=winning_model,  # v0.4: Include winner for Elo
        winner_source=winner_source   # v0.4: Include selection method
    )

    repo = ArchGuruRepo(db_path)
    return repo.persist_run_result(run_result, prompt_version)