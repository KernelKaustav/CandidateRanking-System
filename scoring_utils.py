"""
scoring_utils.py
----------------
Custom-weight scoring variant used by the Streamlit app (app.py).
Allows the user to override default WEIGHTS via the sidebar sliders.
"""

from __future__ import annotations
from scorer import behavioral_composite, PENALTIES


def score_candidate_custom(feat: dict, semantic_sim: float, custom_weights: dict) -> float:
    """
    Compute composite score using caller-supplied weights (from Streamlit sliders).

    Parameters
    ----------
    feat           : feature dict from features.extract_features()
    semantic_sim   : cosine similarity between candidate and JD embedding
    custom_weights : dict with keys matching WEIGHTS in scorer.py

    Returns
    -------
    float in [0, 1] (approximately)
    """
    raw = (
        custom_weights.get("semantic_sim", 0.30)    * float(semantic_sim)
      + custom_weights.get("required_skill", 0.20)  * feat.get("required_skill_score", 0)
      + custom_weights.get("experience_fit", 0.15)  * feat.get("exp_fit", 0)
      + custom_weights.get("behavioral", 0.20)      * behavioral_composite(feat)
      + custom_weights.get("nice_skill", 0.05)      * feat.get("nice_skill_score", 0)
      + custom_weights.get("github_activity", 0.05) * feat.get("github_activity", 0)
      + custom_weights.get("title_match", 0.05)     * feat.get("title_ml_match", 0)
    )

    # Penalty multipliers (same as scorer.py)
    multiplier = 1.0

    if feat.get("all_consulting_career"):
        multiplier *= PENALTIES["consulting_only"]

    if feat.get("location_match", 1.0) < 0.5:
        multiplier *= PENALTIES["location_mismatch"]

    if feat.get("honeypot_score", 0) > 0.5:
        multiplier *= PENALTIES["honeypot"]

    disq_hits = int(feat.get("disqualifier_hit_count", 0))
    if disq_hits > 0:
        multiplier *= (PENALTIES["disqualifier"] ** disq_hits)

    return float(raw * multiplier)
