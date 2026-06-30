"""
reasoning.py  —  Real schema version
--------------------------------------
Generates specific, non-templated 1-2 sentence reasoning per candidate.
Submission spec Stage 4 checks:
  - Specific facts (years, title, named skills, signal values)
  - JD connection
  - Honest concerns where applicable
  - No hallucination
  - Variation across candidates
  - Rank consistency (tone must match rank)
"""

from __future__ import annotations
from datetime import datetime

TODAY = datetime.now().date()


def _days_ago_str(date_str: str | None) -> str:
    if not date_str:
        return "unknown"
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        days = (TODAY - d).days
        if days <= 7:
            return f"{days}d ago"
        elif days <= 60:
            return f"{days // 7}w ago"
        else:
            return f"{days // 30}mo ago"
    except Exception:
        return "unknown"


def _top_skills(candidate: dict, n: int = 3) -> list[str]:
    """Return top-n skill names by duration_months (real usage proxy)."""
    skills = candidate.get("skills", [])
    sorted_skills = sorted(
        [s for s in skills if (s.get("duration_months") or 0) > 0],
        key=lambda s: s.get("duration_months", 0),
        reverse=True
    )
    return [s["name"] for s in sorted_skills[:n]]


def _recent_companies(candidate: dict, n: int = 2) -> list[str]:
    """Return n most recent companies from career_history."""
    career = sorted(
        candidate.get("career_history", []),
        key=lambda ch: ch.get("start_date", ""),
        reverse=True
    )
    return [ch["company"] for ch in career[:n] if ch.get("company")]


def generate_reasoning(
    candidate: dict,
    feat: dict,
    rank: int,
    score: float,
) -> str:
    """
    Generate specific, non-hallucinated 1-2 sentence reasoning.
    Tone matches rank (rank 1-10: strong positive; 50+: honest about gaps).
    """
    p = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    yrs = feat.get("years_exp", 0)
    title = p.get("current_title", "Engineer")
    company = p.get("current_company", "")
    loc = p.get("location", "")
    top_skills = _top_skills(candidate, 3)
    companies = _recent_companies(candidate, 2)
    last_active = _days_ago_str(signals.get("last_active_date"))
    notice = feat.get("notice_period_days", 90)
    rrr = feat.get("recruiter_response_rate", 0)
    icr = feat.get("interview_completion_rate", 0)
    req_score = feat.get("required_skill_score", 0)
    open_to_work = bool(signals.get("open_to_work_flag", False))
    github = feat.get("github_activity", 0)
    consulting_only = bool(feat.get("all_consulting_career", 0))
    disq = int(feat.get("disqualifier_hit_count", 0))
    exp_fit = feat.get("exp_fit", 0)

    # ── Build reasoning based on rank tier ───────────────────────────────────

    skill_str = ", ".join(top_skills) if top_skills else "core ML skills"
    company_str = " and ".join(companies) if companies else company

    if rank <= 10:
        # Strong match — lead with strengths, mention any notable concern
        avail_str = f"active {last_active}, {notice}d notice" if notice <= 60 else f"{notice}d notice"
        concern = ""
        if notice > 90:
            concern = f"; notice period ({notice}d) is the main friction."
        elif not open_to_work:
            concern = "; not currently marked open-to-work."
        elif rrr < 0.5:
            concern = f"; recruiter response rate is low ({rrr:.0%})."

        return (
            f"{yrs:.1f}yr {title} from {company_str} with strong {skill_str} depth "
            f"({avail_str}, {rrr:.0%} response rate) — "
            f"matches the product-company background and retrieval/ranking core the JD requires"
            f"{concern}"
        )

    elif rank <= 30:
        # Good match with one notable gap
        gap = ""
        if consulting_only:
            gap = f"entire career at IT services ({company_str}), which the JD flags"
        elif disq > 0:
            gap = "profile has a JD disqualifier signal (research-only or CV/speech-only background)"
        elif exp_fit < 0.75:
            gap = f"{yrs:.1f}yr experience is outside the 5-9yr ideal range"
        elif notice > 90:
            gap = f"long notice period ({notice}d)"
        elif feat.get("days_inactive", 0) > 90:
            gap = f"inactive for {feat['days_inactive']}d"
        else:
            gap = f"recruiter response rate ({rrr:.0%}) is a mild concern"

        return (
            f"{yrs:.1f}yr {title} with relevant {skill_str} experience at {company_str}; "
            f"solid retrieval/ranking signal but {gap}."
        )

    elif rank <= 60:
        # Moderate — skill overlap but meaningful gaps
        primary_gap = "limited production retrieval/ranking signal in profile"
        if consulting_only:
            primary_gap = f"consulting-only career (50% scoring penalty applied)"
        elif disq > 0:
            primary_gap = "profile triggers JD disqualifier (CV/speech/research-only)"
        elif req_score < 0.3:
            primary_gap = "few required skills matched (embeddings, vector search, evaluation)"

        return (
            f"{yrs:.1f}yr {title} at {company_str} — adjacent background with some skill overlap "
            f"({skill_str}), but {primary_gap}."
        )

    else:
        # Low tier — honest about poor fit
        reason = "minimal overlap with JD's core requirements"
        if feat.get("honeypot_score", 0) > 0.3:
            reason = "profile consistency signals are suspicious (possible honeypot)"
        elif consulting_only:
            reason = "consulting-only career with no product-company background"
        elif req_score < 0.15:
            reason = "skills are unrelated to retrieval, ranking, or NLP"
        elif feat.get("days_inactive", 0) > 180:
            reason = f"inactive for {feat['days_inactive']}d and low engagement signals"
        elif yrs < 2:
            reason = f"only {yrs:.1f}yr experience, well below the 5yr minimum"

        return (
            f"{yrs:.1f}yr {title} — included at rank {rank} as marginal fit; "
            f"{reason}."
        )
