"""
scorer.py  —  Real schema version
----------------------------------
Weighted scoring formula + hard penalty multipliers.

Weights tuned to JD priorities:
  - NDCG@10 is 50% of the score → top-10 quality is everything
  - Behavioral signals explicitly called out in JD and signals doc
  - Honeypots must not appear in top 100
"""

from __future__ import annotations

# ── Weights ───────────────────────────────────────────────────────────────────

WEIGHTS = {
    "semantic_sim":        0.25,   # BGE cosine sim vs JD (or TF-IDF in lite mode)
    "required_skill":      0.18,   # fraction of required skills matched
    "req_skill_depth":     0.07,   # total months in required skills (real usage)
    "assessment_score":    0.05,   # Redrob-verified skill assessments
    "experience_fit":      0.09,   # years in ideal 5-9 range
    "behavioral":          0.13,   # composite of all 23 signals
    "career_relevance":    0.12,   # career descriptions mention ML/retrieval work
    "nice_skill":          0.04,   # nice-to-have skills
    "github_activity":     0.04,   # open-source signal
    "title_match":         0.02,   # current title relevance
    "edu_tier":            0.01,   # education institution tier
}

# ── Penalty multipliers ───────────────────────────────────────────────────────

PENALTIES = {
    "honeypot":          0.0,   # zero-out completely
    "consulting_only":   0.50,  # entire career at IT services firms
    "location_mismatch": 0.80,  # outside India + unwilling to relocate
    "disqualifier":      0.65,  # per disqualifier hit (applied exponentially)
    "cv_speech_only":    0.70,  # CV/speech without NLP/IR
}

# ── Behavioral composite ──────────────────────────────────────────────────────

def behavioral_composite(feat: dict) -> float:
    """
    Weighted composite of all 23 redrob behavioral signals.
    Returns float in [0, 1].
    """
    score = (
        0.20 * feat.get("recency_score", 0)
      + 0.15 * feat.get("open_to_work", 0)
      + 0.18 * feat.get("recruiter_response_rate", 0)
      + 0.12 * feat.get("interview_completion_rate", 0)
      + 0.08 * feat.get("response_speed", 0)
      + 0.07 * feat.get("notice_score", 0)
      + 0.05 * feat.get("profile_completeness", 0)
      + 0.05 * feat.get("market_demand", 0)
      + 0.03 * feat.get("verified_email", 0)
      + 0.03 * feat.get("verified_phone", 0)
      + 0.02 * feat.get("linkedin_connected", 0)
      + 0.02 * min(feat.get("applications_30d", 0) / 5.0, 1.0)
    )
    return min(float(score), 1.0)


# ── Main scorer ───────────────────────────────────────────────────────────────

def score_candidate(feat: dict, semantic_sim: float) -> float:
    """
    Compute composite score. Returns float in [0, 1] approximately.
    Hard penalties applied as multipliers after weighted sum.
    """

    # Hard disqualify: honeypot
    if feat.get("honeypot_score", 0) > 0.5:
        return 0.0

    beh = behavioral_composite(feat)

    raw = (
        WEIGHTS["semantic_sim"]    * float(semantic_sim)
      + WEIGHTS["required_skill"]  * feat.get("required_skill_score", 0)
      + WEIGHTS["req_skill_depth"] * feat.get("req_skill_depth", 0)
      + WEIGHTS["assessment_score"]* feat.get("assessment_score", 0)
      + WEIGHTS["experience_fit"]  * feat.get("exp_fit", 0)
      + WEIGHTS["behavioral"]      * beh
      + WEIGHTS["nice_skill"]      * feat.get("nice_skill_score", 0)
      + WEIGHTS["github_activity"] * feat.get("github_activity", 0)
      + WEIGHTS["title_match"]     * feat.get("title_ml_match", 0)
      + WEIGHTS["edu_tier"]        * feat.get("edu_tier_score", 0)
      + WEIGHTS["career_relevance"]  * feat.get("career_relevance", 0)
    )

    # Multiplicative penalties
    multiplier = 1.0

    if feat.get("all_consulting_career"):
        multiplier *= PENALTIES["consulting_only"]

    loc = feat.get("location_match", 1.0)
    if loc < 0.4:
        multiplier *= PENALTIES["location_mismatch"]

    disq = int(feat.get("disqualifier_hit_count", 0))
    for _ in range(disq):
        multiplier *= PENALTIES["disqualifier"]

    return float(raw * multiplier)
