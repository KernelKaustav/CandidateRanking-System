"""
features.py  —  Real schema version
------------------------------------
Extracts all scoring signals from a candidate record matching the actual
candidates.jsonl schema from the Redrob hackathon bundle.
"""

from __future__ import annotations
import re
from datetime import datetime, date

# ── Constants ─────────────────────────────────────────────────────────────────

TODAY = datetime.now().date()

CONSULTING_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree",
    "l&t infotech", "ltimindtree", "niit technologies", "mastech", "patni",
    "igate", "syntel", "kpit", "cyient", "zensar", "birlasoft", "persistent",
    "nagarro", "coforge", "sonata", "xoriant", "happiest minds"
}

CONSULTING_INDUSTRIES = {"IT Services", "Consulting", "Outsourcing"}

# JD: required skills (exact/fuzzy match against candidate skill names)
REQUIRED_SKILLS = [
    "embeddings", "embedding", "sentence-transformers", "sentence transformers",
    "bge", "e5", "openai embeddings",
    "vector database", "vector search", "vector db",
    "pinecone", "weaviate", "qdrant", "milvus", "faiss",
    "opensearch", "elasticsearch",
    "hybrid search", "hybrid retrieval",
    "python",
    "ranking", "retrieval", "information retrieval",
    "ndcg", "mrr", "map", "evaluation framework", "a/b testing",
]

NICE_SKILLS = [
    "lora", "qlora", "peft", "fine-tuning", "fine tuning",
    "learning to rank", "xgboost", "lightgbm",
    "distributed systems", "large scale inference",
    "open source", "github",
    "recommendation", "recommender",
    "nlp", "natural language processing",
]

# JD disqualifiers — explicit red flags
DISQUALIFIER_PATTERNS = [
    r"\bpure research\b", r"\bacademic lab\b", r"\bphd researcher\b",
    r"\bcomputer vision\b", r"\bimage (classification|segmentation|detection)\b",
    r"\bspeech recognition\b", r"\brobotic\b",
    r"\blangchain\b",  # only bad if that's ALL they have
    r"\bmarketing\b",
    r"\bonly.*openai\b", r"\bwrapper\b",
]

# Location targets from JD
TARGET_LOCATIONS = {"pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore", "bengaluru", "ncr"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _days_since(date_str: str | None) -> int:
    """Days since a date string. Returns 999 if missing/invalid."""
    if not date_str:
        return 999
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        return (TODAY - d).days
    except Exception:
        return 999


def _skill_name_matches(skill_name: str, targets: list[str]) -> bool:
    s = skill_name.lower().strip()
    for t in targets:
        if t in s or s in t:
            return True
    return False


def _is_consulting(company: str, industry: str) -> bool:
    co = company.lower()
    for c in CONSULTING_COMPANIES:
        if c in co:
            return True
    if industry in CONSULTING_INDUSTRIES:
        return True
    return False


def _text_for_embedding(c: dict) -> str:
    """Build a single text blob representing the candidate for embedding."""
    p = c.get("profile", {})
    parts = [
        p.get("headline", ""),
        p.get("summary", ""),
        p.get("current_title", ""),
    ]
    for s in c.get("skills", []):
        parts.append(s.get("name", ""))
    for ch in c.get("career_history", []):
        parts.append(ch.get("title", ""))
        parts.append(ch.get("description", ""))
    return " ".join(p for p in parts if p)


# ── Main feature extractor ────────────────────────────────────────────────────

def extract_features(c: dict, jd: dict | None = None) -> dict:
    """
    Extract all scoring features from a real candidate record.
    Returns a flat dict ready for scorer.py.
    """
    feat: dict = {}
    p = c.get("profile", {})
    signals = c.get("redrob_signals", {})
    career = c.get("career_history", [])
    skills = c.get("skills", [])

    feat["candidate_id"] = c.get("candidate_id", "")
    feat["text"] = _text_for_embedding(c)

    # ── Profile basics ────────────────────────────────────────────────────────
    feat["years_exp"] = float(p.get("years_of_experience", 0) or 0)
    feat["current_title"] = (p.get("current_title") or "").lower()
    feat["current_industry"] = p.get("current_industry", "")
    feat["location"] = (p.get("location") or "").lower()
    feat["country"] = (p.get("country") or "").lower()

    # ── Experience fit (ideal: 5-9 yrs per JD) ────────────────────────────────
    yrs = feat["years_exp"]
    if 5 <= yrs <= 9:
        feat["exp_fit"] = 1.0
    elif 4 <= yrs < 5 or 9 < yrs <= 11:
        feat["exp_fit"] = 0.75
    elif 3 <= yrs < 4 or 11 < yrs <= 13:
        feat["exp_fit"] = 0.45
    else:
        feat["exp_fit"] = 0.15

    # ── Skill matching ────────────────────────────────────────────────────────
    skill_names = [s.get("name", "") for s in skills]
    skill_text = " ".join(skill_names).lower()

    req_hits = sum(
        1 for t in REQUIRED_SKILLS if t in skill_text
    )
    feat["required_skill_score"] = min(req_hits / max(len(REQUIRED_SKILLS) * 0.4, 1), 1.0)

    nice_hits = sum(1 for t in NICE_SKILLS if t in skill_text)
    feat["nice_skill_score"] = min(nice_hits / max(len(NICE_SKILLS) * 0.4, 1), 1.0)

    # Also check skill names individually (duration as proxy for real usage)
    req_skill_duration = 0
    for s in skills:
        if _skill_name_matches(s.get("name", ""), REQUIRED_SKILLS):
            months = s.get("duration_months") or 0
            if months > 0:
                req_skill_duration += months
    feat["req_skill_total_months"] = req_skill_duration
    feat["req_skill_depth"] = min(req_skill_duration / 60.0, 1.0)  # normalise to 5 yrs

    # Skill assessment scores (Redrob-verified)
    assessment = signals.get("skill_assessment_scores", {}) or {}
    if assessment:
        relevant_scores = []
        for skill_name, score in assessment.items():
            if _skill_name_matches(skill_name, REQUIRED_SKILLS + NICE_SKILLS):
                relevant_scores.append(float(score) / 100.0)
        feat["assessment_score"] = sum(relevant_scores) / len(relevant_scores) if relevant_scores else 0.0
    else:
        feat["assessment_score"] = 0.0

    # ── Title relevance ───────────────────────────────────────────────────────
    title = feat["current_title"]
    ml_title_keywords = ["ml", "machine learning", "ai ", "artificial intelligence",
                          "nlp", "search", "ranking", "retrieval", "data scientist",
                          "applied scientist", "research scientist", "engineer"]
    feat["title_ml_match"] = 1.0 if any(k in title for k in ml_title_keywords) else 0.3

    # ── Career analysis ───────────────────────────────────────────────────────
    total_roles = len(career)
    consulting_roles = sum(
        1 for ch in career
        if _is_consulting(ch.get("company", ""), ch.get("industry", ""))
    )
    feat["all_consulting_career"] = int(total_roles > 0 and consulting_roles == total_roles)
    feat["consulting_fraction"] = consulting_roles / max(total_roles, 1)

    # Has product company experience (non-consulting)
    feat["has_product_exp"] = int(consulting_roles < total_roles)

    # Career tenure stability (avg months per role — penalise title-chasers)
    if career:
        durations = [ch.get("duration_months", 0) or 0 for ch in career]
        avg_tenure = sum(durations) / len(durations) if durations else 0
        feat["avg_tenure_months"] = avg_tenure
        # JD wants 3+ year tenure; penalise < 18 months average
        feat["tenure_stability"] = min(avg_tenure / 36.0, 1.0)
    else:
        feat["avg_tenure_months"] = 0
        feat["tenure_stability"] = 0.0

    # ── Honeypot detection ────────────────────────────────────────────────────
    honeypot_flags = 0

    # Flag 1: "expert" proficiency + 0 duration months on many skills
    zero_duration_experts = sum(
        1 for s in skills
        if (s.get("proficiency", "").lower() == "expert"
            and (s.get("duration_months") or 0) == 0)
    )
    if zero_duration_experts >= 3:
        honeypot_flags += 1
    if zero_duration_experts >= 6:
        honeypot_flags += 1

    # Flag 2: Career duration_months vs stated years_of_experience inconsistency
    if career:
        stated_months = feat["years_exp"] * 12
        actual_career_months = sum(ch.get("duration_months", 0) or 0 for ch in career)
        # If actual career months is wildly less than stated (>3yr gap) → suspicious
        if stated_months > 12 and actual_career_months < stated_months * 0.3:
            honeypot_flags += 1

    # Flag 3: Skills with zero endorsements AND zero duration on many skills
    zero_everything = sum(
        1 for s in skills
        if (s.get("endorsements", 0) == 0
            and (s.get("duration_months") or 0) == 0
            and s.get("proficiency", "").lower() in ("expert", "advanced"))
    )
    if zero_everything >= 4:
        honeypot_flags += 1

    # Flag 4: Profile completeness low but skill list huge
    completeness = signals.get("profile_completeness_score", 100)
    if completeness < 50 and len(skills) > 12:
        honeypot_flags += 1

    # Flag 5: interview_completion_rate very low with high recruiter_response_rate
    icr = signals.get("interview_completion_rate", 1.0)
    rrr = signals.get("recruiter_response_rate", 0.0)
    if icr is not None and rrr is not None:
        if float(icr) < 0.2 and float(rrr) > 0.8:
            honeypot_flags += 1

    feat["honeypot_flags"] = honeypot_flags
    feat["honeypot_score"] = min(honeypot_flags / 4.0, 1.0)

    # ── Disqualifiers from JD ─────────────────────────────────────────────────
    full_text = feat["text"].lower()
    summary = (p.get("summary") or "").lower()
    headline = (p.get("headline") or "").lower()

    disq_count = 0

    # Pure research / academic (explicit JD disqualifier)
    if any(w in summary for w in ["pure research", "academic lab", "phd researcher", "no production"]):
        disq_count += 1

    # CV/speech/robotics without NLP/IR
    cv_only = (
        any(w in skill_text for w in ["computer vision", "image classification", "speech recognition", "robotics"])
        and not any(w in skill_text for w in ["nlp", "retrieval", "ranking", "embedding", "search"])
    )
    if cv_only:
        disq_count += 1

    # LangChain-only with very low experience
    langchain_only = (
        "langchain" in skill_text
        and not any(w in skill_text for w in ["faiss", "elasticsearch", "pinecone", "qdrant", "milvus", "weaviate"])
        and feat["years_exp"] < 3
    )
    if langchain_only:
        disq_count += 1

    # No production code in career descriptions (architecture/tech lead drift)
    arch_drift = all(
        any(w in (ch.get("description", "") or "").lower()
            for w in ["architecture", "strategy", "roadmap", "no code", "management"])
        for ch in career[-2:] if career
    )
    # Only flag if recent 2 roles show this AND title suggests it
    if arch_drift and "architect" in title and "engineer" not in title:
        disq_count += 1

    feat["disqualifier_hit_count"] = disq_count

    # ── Location ──────────────────────────────────────────────────────────────
    loc = feat["location"]
    country = feat["country"]
    in_india = country in ("india", "in", "") or any(
        city in loc for city in TARGET_LOCATIONS
    )
    in_target = any(city in loc for city in TARGET_LOCATIONS)
    relocate = bool(signals.get("willing_to_relocate", False))
    feat["location_match"] = 1.0 if in_target else (0.7 if (in_india and relocate) else (0.5 if in_india else 0.2))

    # ── Behavioral signals (all 23) ───────────────────────────────────────────
    feat["profile_completeness"] = float(signals.get("profile_completeness_score", 0) or 0) / 100.0
    feat["open_to_work"] = float(bool(signals.get("open_to_work_flag", False)))
    feat["recruiter_response_rate"] = float(signals.get("recruiter_response_rate", 0) or 0)
    feat["interview_completion_rate"] = float(signals.get("interview_completion_rate", 0) or 0)
    feat["offer_acceptance_rate"] = float(signals.get("offer_acceptance_rate", -1) or -1)

    # Recency of activity
    days_inactive = _days_since(signals.get("last_active_date"))
    if days_inactive <= 7:
        feat["recency_score"] = 1.0
    elif days_inactive <= 30:
        feat["recency_score"] = 0.85
    elif days_inactive <= 90:
        feat["recency_score"] = 0.60
    elif days_inactive <= 180:
        feat["recency_score"] = 0.35
    else:
        feat["recency_score"] = 0.10
    feat["days_inactive"] = days_inactive

    # Response speed (lower = better)
    avg_resp_hours = float(signals.get("avg_response_time_hours", 999) or 999)
    if avg_resp_hours <= 4:
        feat["response_speed"] = 1.0
    elif avg_resp_hours <= 24:
        feat["response_speed"] = 0.8
    elif avg_resp_hours <= 72:
        feat["response_speed"] = 0.5
    else:
        feat["response_speed"] = 0.2

    # Notice period (JD wants sub-30 days preferred)
    notice = int(signals.get("notice_period_days", 90) or 90)
    if notice <= 30:
        feat["notice_score"] = 1.0
    elif notice <= 60:
        feat["notice_score"] = 0.7
    elif notice <= 90:
        feat["notice_score"] = 0.5
    else:
        feat["notice_score"] = 0.2
    feat["notice_period_days"] = notice

    feat["github_activity"] = float(signals.get("github_activity_score", 0) or 0)
    if feat["github_activity"] < 0:  # -1 means no GitHub linked
        feat["github_activity"] = 0.0
    feat["github_activity"] /= 100.0  # normalise

    feat["profile_views_30d"] = int(signals.get("profile_views_received_30d", 0) or 0)
    feat["applications_30d"] = int(signals.get("applications_submitted_30d", 0) or 0)
    feat["saved_by_recruiters_30d"] = int(signals.get("saved_by_recruiters_30d", 0) or 0)
    feat["search_appearance_30d"] = int(signals.get("search_appearance_30d", 0) or 0)
    feat["connection_count"] = int(signals.get("connection_count", 0) or 0)
    feat["endorsements_received"] = int(signals.get("endorsements_received", 0) or 0)
    feat["verified_email"] = float(bool(signals.get("verified_email", False)))
    feat["verified_phone"] = float(bool(signals.get("verified_phone", False)))
    feat["linkedin_connected"] = float(bool(signals.get("linkedin_connected", False)))

    # Market demand signal (recruiters actively interested)
    feat["market_demand"] = min(
        (feat["saved_by_recruiters_30d"] / 10.0)
        + (feat["profile_views_30d"] / 100.0), 1.0
    )

    # Education tier bonus
    edu = c.get("education", [])
    best_tier = "tier_3"
    for e in edu:
        t = e.get("tier", "tier_3")
        if t == "tier_1":
            best_tier = "tier_1"
            break
        elif t == "tier_2" and best_tier != "tier_1":
            best_tier = "tier_2"
    feat["edu_tier_score"] = {"tier_1": 1.0, "tier_2": 0.7, "tier_3": 0.4}.get(best_tier, 0.4)

    return feat


# ── Career description relevance (added after initial testing) ────────────────

def add_career_relevance(feat: dict, candidate: dict) -> dict:
    """
    Score how much career descriptions (not just skill tags) mention ML/retrieval work.
    This catches keyword-stuffers: skills list has 'Embeddings' but career is .NET dev.
    """
    career = candidate.get("career_history", [])
    CAREER_KEYWORDS = [
        "embedding", "retrieval", "ranking", "search", "recommendation",
        "vector", "faiss", "pinecone", "elasticsearch", "opensearch",
        "nlp", "language model", "information retrieval", "semantic",
        "ndcg", "mrr", "a/b test", "fine-tun", "rag", "dense retrieval",
    ]
    total_months = 0
    relevant_months = 0
    for ch in career:
        desc = (ch.get("description") or "").lower()
        months = ch.get("duration_months", 0) or 0
        total_months += months
        if any(kw in desc for kw in CAREER_KEYWORDS):
            relevant_months += months
    feat["career_relevance"] = relevant_months / max(total_months, 1)
    add_career_relevance(feat, c)
    return feat
