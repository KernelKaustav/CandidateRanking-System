"""
Uses TF-IDF cosine similarity as the semantic layer.
Identical scoring logic and penalties as rank.py.

Produces a valid submission CSV.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
from pathlib import Path

from features import extract_features
from scorer import score_candidate, behavioral_composite
from reasoning import generate_reasoning

# ── JD text (full, for TF-IDF) ───────────────────────────────────────────────

JD_TEXT = """
senior ai engineer founding team product company embeddings retrieval ranking
sentence transformers bge e5 openai embeddings vector database pinecone weaviate
qdrant milvus faiss opensearch elasticsearch hybrid search python production
evaluation framework ndcg mrr map ab testing lora qlora peft fine tuning
learning to rank xgboost distributed systems inference optimization open source
nlp natural language processing recommendation retrieval ranking production
shipped product company real users 5 9 years experience
"""

# ── TF-IDF ────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_idf(docs: list[list[str]]) -> dict[str, float]:
    N = len(docs)
    df: dict[str, int] = {}
    for toks in docs:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    return {t: math.log((N + 1) / (c + 1)) + 1 for t, c in df.items()}


def tfidf_vec(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    from collections import Counter
    tf = Counter(tokens)
    total = max(len(tokens), 1)
    return {t: (c / total) * idf.get(t, 1.0) for t, c in tf.items()}


def cosine(a: dict, b: dict) -> float:
    shared = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in shared)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_candidates(path: str) -> list[dict]:
    p = Path(path)
    candidates = []
    if p.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
    elif p.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        candidates = data if isinstance(data, list) else [data]
    return candidates


def validate_and_save(results: list[dict], out_path: str) -> None:
    """Validate format then save CSV."""
    assert len(results) == 100, f"Need exactly 100 rows, got {len(results)}"
    ranks = [r["rank"] for r in results]
    assert sorted(ranks) == list(range(1, 101)), "Ranks must be 1-100 each exactly once"
    scores = [r["score"] for r in results]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], f"Score not descending at rank {i + 2}"
    ids = [r["candidate_id"] for r in results]
    assert len(ids) == len(set(ids)), "Duplicate candidate_ids"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(results)

    print(f"[validate] ✅  All checks passed — {len(results)} rows → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Redrob ranker (lite, CPU-only)")
    parser.add_argument("--candidates", required=True, help="candidates.jsonl / .jsonl.gz / .json")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument("--top", type=int, default=100, help="Keep top-N (must be 100 for submission)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"[rank_lite] Loading candidates from {args.candidates} ...")
    candidates = load_candidates(args.candidates)
    print(f"[rank_lite] Loaded {len(candidates):,} candidates")

    print("[rank_lite] Extracting features ...")
    all_features = []
    for i, c in enumerate(candidates):
        all_features.append(extract_features(c))
        if (i + 1) % 10000 == 0:
            print(f"  ... {i+1:,}/{len(candidates):,}")

    print("[rank_lite] Building TF-IDF index ...")
    jd_tokens = tokenize(JD_TEXT)
    cand_tokens = [tokenize(f["text"]) for f in all_features]
    all_docs = [jd_tokens] + cand_tokens
    idf = build_idf(all_docs)
    jd_vec = tfidf_vec(jd_tokens, idf)

    print("[rank_lite] Computing similarities ...")
    sims = [cosine(jd_vec, tfidf_vec(toks, idf)) for toks in cand_tokens]

    print("[rank_lite] Scoring ...")
    scored = []
    for i, (feat, sim) in enumerate(zip(all_features, sims)):
        s = score_candidate(feat, sim)
        scored.append({"idx": i, "candidate": candidates[i], "features": feat, "score": s, "sim": sim})

    # Sort by score desc, break ties by candidate_id asc (per spec)
    scored.sort(key=lambda x: (-x["score"], x["features"]["candidate_id"]))
    top_n = scored[:args.top]

    print(f"[rank_lite] Generating reasoning for top {len(top_n)} ...")
    results = []
    for rank_pos, item in enumerate(top_n, start=1):
        if args.verbose and rank_pos <= 20:
            beh = behavioral_composite(item["features"])
            f = item["features"]
            print(
                f"  #{rank_pos:>3}  {f['candidate_id']}  "
                f"score={item['score']:.4f}  sim={item['sim']:.3f}  "
                f"req={f.get('required_skill_score', 0):.2f}  "
                f"exp={f.get('exp_fit', 0):.2f}  beh={beh:.2f}  "
                f"honeypot={f.get('honeypot_score', 0):.2f}  "
                f"consulting={f.get('all_consulting_career', 0)}"
            )
        reasoning = generate_reasoning(
            item["candidate"], item["features"], rank=rank_pos, score=item["score"]
        )
        results.append({
            "candidate_id": item["features"]["candidate_id"],
            "rank":         rank_pos,
            "score":        round(item["score"], 6),
            "reasoning":    reasoning,
        })

    print(f"\n{'─'*70}")
    print(f"  TOP 10")
    print(f"{'─'*70}")
    for r in results[:10]:
        print(f"  #{r['rank']:>3}  {r['candidate_id']}  score={r['score']:.4f}")
        print(f"       {r['reasoning'][:100]}...")
    print(f"{'─'*70}\n")

    validate_and_save(results, args.out)


if __name__ == "__main__":
    main()
