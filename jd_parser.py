"""
jd_parser.py
------------
Uses Claude (Anthropic API) to extract structured requirements from the Job Description.
Run ONCE offline to produce jd_parsed.json — do NOT call during the ranking step.

Usage:
    python jd_parser.py --jd job_description.md --out jd_parsed.json
"""

import argparse
import json
import os
import re

# ── Anthropic client (API key injected by environment) ────────────────────────
try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


JD_TEXT = """
Job Description: Senior AI Engineer — Founding Team
Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India (Hybrid)
Experience Required: 5-9 years

MUST HAVE:
- Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5)
- Production experience with vector databases or hybrid search (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS)
- Strong Python
- Hands-on experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP, A/B testing)

NICE TO HAVE:
- LLM fine-tuning (LoRA, QLoRA, PEFT)
- Learning-to-rank models (XGBoost or neural)
- HR-tech / recruiting tech / marketplace experience
- Distributed systems or large-scale inference optimization
- Open-source contributions in AI/ML

DISQUALIFIERS (hard):
- Pure research / academic only (no production deployment)
- AI experience only from recent LangChain/OpenAI wrappers (< 12 months, no prior ML production)
- Not written production code in last 18 months (pure architect/tech-lead)
- Consulting-firm only career (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, Capgemini) without product-company experience
- Primary expertise in computer vision / speech / robotics without NLP/IR
- Entirely closed-source work for 5+ years, no external validation

IDEAL PROFILE:
- 6-8 years total, 4-5 years in applied ML at product companies
- Shipped at least one end-to-end ranking/search/recommendation system at scale
- Tilts toward "shipper" not "researcher"
- Located in or willing to relocate to Noida/Pune
- Active on job market
"""

SYSTEM_PROMPT = """You are a technical recruiter assistant. Extract structured hiring requirements from job descriptions.
Return ONLY valid JSON, no markdown fences, no preamble."""

USER_PROMPT = f"""Extract structured requirements from this job description and return a JSON object with these exact keys:

{{
  "required_skills": ["list of must-have technical skills as lowercase strings"],
  "nice_to_have_skills": ["list of good-to-have skills as lowercase strings"],
  "disqualifier_keywords": ["words/phrases that, if dominant in profile, indicate disqualification"],
  "ideal_experience_years_min": 5,
  "ideal_experience_years_max": 9,
  "preferred_locations": ["list of city names"],
  "role_type": "applied_ml_engineer",
  "key_signals": ["embedding retrieval", "vector search", "ranking systems", "python", "production ml"],
  "anti_signals": ["pure research", "consulting only", "computer vision only", "langchain wrapper only"]
}}

Job Description:
{JD_TEXT}
"""


def parse_with_llm() -> dict:
    """Call Claude API to parse JD. Returns structured dict."""
    if not _HAS_ANTHROPIC:
        raise ImportError("anthropic package not installed. pip install anthropic")

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_PROMPT}]
    )
    raw = response.content[0].text.strip()
    # Strip markdown fences if model adds them despite instructions
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def parse_fallback() -> dict:
    """Hardcoded fallback if API unavailable."""
    return {
        "required_skills": [
            "embeddings", "sentence-transformers", "vector database", "faiss",
            "pinecone", "weaviate", "qdrant", "elasticsearch", "opensearch",
            "python", "retrieval", "ranking", "ndcg", "mrr", "a/b testing",
            "hybrid search", "information retrieval"
        ],
        "nice_to_have_skills": [
            "lora", "qlora", "peft", "fine-tuning", "xgboost", "learning to rank",
            "hr-tech", "marketplace", "distributed systems", "inference optimization",
            "open source"
        ],
        "disqualifier_keywords": [
            "computer vision", "speech recognition", "robotics",
            "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
            "research scientist", "phd researcher"
        ],
        "ideal_experience_years_min": 5,
        "ideal_experience_years_max": 9,
        "preferred_locations": ["pune", "noida", "delhi", "ncr", "hyderabad", "mumbai", "bangalore"],
        "role_type": "applied_ml_engineer",
        "key_signals": [
            "embedding retrieval", "vector search", "ranking systems",
            "python", "production ml", "search", "recommendation"
        ],
        "anti_signals": [
            "pure research", "consulting only", "computer vision only",
            "langchain wrapper only", "no production"
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd", default=None, help="Path to job description text file (optional)")
    parser.add_argument("--out", default="jd_parsed.json", help="Output JSON path")
    parser.add_argument("--fallback", action="store_true", help="Use hardcoded fallback (no API call)")
    args = parser.parse_args()

    if args.fallback or not _HAS_ANTHROPIC:
        print("[jd_parser] Using hardcoded fallback (no API call)")
        result = parse_fallback()
    else:
        print("[jd_parser] Calling Claude API to parse JD...")
        try:
            result = parse_with_llm()
            print("[jd_parser] LLM parse successful")
        except Exception as e:
            print(f"[jd_parser] LLM parse failed ({e}), using fallback")
            result = parse_fallback()

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[jd_parser] Saved to {args.out}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
