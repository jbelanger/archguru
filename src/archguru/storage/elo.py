"""
ArchGuru v0.4 - Elo Rating System
Online Elo rating updates for model performance tracking per decision type
"""

import math
from typing import List, Tuple, Dict, Any, Optional, Union
import sqlite3

def calculate_elo_update(
    winner_rating: float,
    loser_rating: float,
    k_factor: float = 32.0
) -> Tuple[float, float]:
    """
    Calculate new Elo ratings after a match

    Args:
        winner_rating: Current Elo rating of the winning model
        loser_rating: Current Elo rating of the losing model
        k_factor: K-factor for rating sensitivity (higher = more volatile)

    Returns:
        Tuple of (new_winner_rating, new_loser_rating)
    """
    # Expected scores based on current ratings
    expected_winner = 1 / (1 + 10**((loser_rating - winner_rating) / 400))
    expected_loser = 1 / (1 + 10**((winner_rating - loser_rating) / 400))

    # Actual scores (winner = 1, loser = 0)
    actual_winner = 1.0
    actual_loser = 0.0

    # Update ratings
    new_winner_rating = winner_rating + k_factor * (actual_winner - expected_winner)
    new_loser_rating = loser_rating + k_factor * (actual_loser - expected_loser)

    return new_winner_rating, new_loser_rating


def get_or_create_elo_rating(
    conn: sqlite3.Connection,
    model_id: int,
    decision_type_id: int,
    initial_rating: float = 1200.0,
    k_factor: float = 32.0
) -> Dict[str, Any]:
    """
    Get existing Elo rating or create new one with defaults

    Returns:
        Dict with rating info: {'rating': float, 'k_factor': float, 'matches': int}
    """
    cursor = conn.execute("""
        SELECT rating, k_factor, matches
        FROM model_rating
        WHERE model_id = ? AND decision_type_id = ? AND algo = 'elo'
    """, (model_id, decision_type_id))

    result = cursor.fetchone()
    if result:
        return {
            'rating': result[0],
            'k_factor': result[1],
            'matches': result[2]
        }

    # Create new rating record
    conn.execute("""
        INSERT INTO model_rating (model_id, decision_type_id, algo, rating, k_factor, matches)
        VALUES (?, ?, 'elo', ?, ?, 0)
    """, (model_id, decision_type_id, initial_rating, k_factor))

    return {
        'rating': initial_rating,
        'k_factor': k_factor,
        'matches': 0
    }


def update_elo_ratings_for_run(
    conn: sqlite3.Connection,
    run_id: str,
    winner_model_id: int,
    loser_model_ids: List[int],
    decision_type_id: int,
    judge_model_id: int,
    reason: str = "Arbiter selection"
) -> List[Dict[str, Any]]:
    """
    Update Elo ratings for all model pairs after a run
    Winner beats all other models in pairwise comparisons

    Returns:
        List of rating updates applied
    """
    updates = []

    for loser_model_id in loser_model_ids:
        if winner_model_id == loser_model_id:
            continue

        # Get current ratings
        winner_rating_info = get_or_create_elo_rating(conn, winner_model_id, decision_type_id)
        loser_rating_info = get_or_create_elo_rating(conn, loser_model_id, decision_type_id)

        # Calculate new ratings with consistent K-factor
        K = 32.0  # Standard Elo K-factor for both players
        new_winner_rating, new_loser_rating = calculate_elo_update(
            winner_rating_info['rating'],
            loser_rating_info['rating'],
            K
        )

        # Update winner rating
        conn.execute("""
            UPDATE model_rating
            SET rating = ?, matches = matches + 1, last_updated = CURRENT_TIMESTAMP
            WHERE model_id = ? AND decision_type_id = ? AND algo = 'elo'
        """, (new_winner_rating, winner_model_id, decision_type_id))

        # Update loser rating
        conn.execute("""
            UPDATE model_rating
            SET rating = ?, matches = matches + 1, last_updated = CURRENT_TIMESTAMP
            WHERE model_id = ? AND decision_type_id = ? AND algo = 'elo'
        """, (new_loser_rating, loser_model_id, decision_type_id))

        # Record pairwise judgment
        conn.execute("""
            INSERT INTO pairwise_judgment
            (run_id, decision_type_id, judge_model_id, winner_model_id, loser_model_id, reason)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (run_id, decision_type_id, judge_model_id, winner_model_id, loser_model_id, reason))

        updates.append({
            'winner_model_id': winner_model_id,
            'loser_model_id': loser_model_id,
            'winner_old_rating': winner_rating_info['rating'],
            'winner_new_rating': new_winner_rating,
            'loser_old_rating': loser_rating_info['rating'],
            'loser_new_rating': new_loser_rating,
            'rating_change_winner': new_winner_rating - winner_rating_info['rating'],
            'rating_change_loser': new_loser_rating - loser_rating_info['rating']
        })

    return updates


def get_top_models_by_elo(
    conn: sqlite3.Connection,
    decision_type_id: Optional[int] = None,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Get top models ranked by Elo rating for a decision type

    Args:
        decision_type_id: Filter by decision type (None = all types)
        limit: Number of top models to return

    Returns:
        List of model rankings with rating info
    """
    if decision_type_id is not None:
        query = """
            SELECT
                m.name as model_name,
                dt.label as decision_type,
                mr.rating,
                mr.matches,
                mr.last_updated
            FROM model_rating mr
            JOIN model m ON mr.model_id = m.id
            JOIN decision_type dt ON mr.decision_type_id = dt.id
            WHERE mr.algo = 'elo' AND mr.decision_type_id = ?
            ORDER BY mr.rating DESC
            LIMIT ?
        """
        params: Union[Tuple[int, int], Tuple[int]] = (decision_type_id, limit)
    else:
        query = """
            SELECT
                m.name as model_name,
                dt.label as decision_type,
                mr.rating,
                mr.matches,
                mr.last_updated
            FROM model_rating mr
            JOIN model m ON mr.model_id = m.id
            JOIN decision_type dt ON mr.decision_type_id = dt.id
            WHERE mr.algo = 'elo'
            ORDER BY mr.rating DESC
            LIMIT ?
        """
        params = (limit,)

    conn.row_factory = sqlite3.Row
    cursor = conn.execute(query, params)
    return [dict(row) for row in cursor.fetchall()]


def get_elo_stats_by_decision_type(
    conn: sqlite3.Connection
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get top 5 models by Elo rating for each decision type

    Returns:
        Dict mapping decision type labels to ranked model lists
    """
    conn.row_factory = sqlite3.Row

    # Get all decision types that have ratings
    decision_types = conn.execute("""
        SELECT DISTINCT dt.id, dt.label
        FROM decision_type dt
        JOIN model_rating mr ON dt.id = mr.decision_type_id
        WHERE mr.algo = 'elo'
    """).fetchall()

    results = {}
    for dt in decision_types:
        top_models = get_top_models_by_elo(conn, dt['id'], 5)
        results[dt['label']] = top_models

    return results