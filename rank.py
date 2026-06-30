"""
rank.py
-------
MAIN ENTRY POINT — produces the submission CSV.

Two-phase design:
  Phase 1 (precompute, can be slow, run once):
    - Load all candidates
    - Extract features
    - Generate embeddings with BGE-small (CPU)
    - Build FAISS index
    - Save to disk

  Phase 2 (ranking step, must finish in < 5 min on CPU, no network):
    - Load precomputed index + features
    - Score all candidates
    - Pick top 100
    - Generate reasoning
    - Write CSV

Usage:
    # Full pipeline (precompute + rank):
    python rank.py --candidates candidates.jsonl --out submission.csv

    # Ranking only (after precompute already done):
    python rank.py --candidates candidates.jsonl --out submission.csv --skip-precompute

    # Small sample test:
    python rank.py --candidates sample_candidates.json --out submission.csv --sample
"""

import argparse
import csv
import gzip
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

# ── optional tqdm progress bar ────────────────────────────────────────────────
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

# ── local modules ─────────────────────────────────────────────────────────────
from features import extract_features
from scorer import score_candidate
from reasoning import generate_reasoning


# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_JD_PATH   = "jd_parsed.json"
DEFAULT_INDEX_DIR = "precomputed"
MODEL_NAME        = "BAAI/bge-small-en-v1.5"   # ~90 MB, fast on CPU


# ── helpers ───────────────────────────────────────────────────────────────────

def load_candidates(path: str) -> list[dict]:
    """Load candidates from .jsonl, .jsonl.gz, or .json (sample list)."""
    p = Path(path)
    if p.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if p.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    # .json — assume list
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def load_jd(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[rank] JD file not found at {path}. Run jd_parser.py first or use --fallback flag.")
        print("[rank] Using built-in fallback JD.")
        from jd_parser import parse_fallback
        return parse_fallback()
    with open(path) as f:
        return json.load(f)


def build_jd_text(jd: dict) -> str:
    """Build a text blob from the parsed JD for embedding."""
    parts = jd.get("required_skills", []) + jd.get("nice_to_have_skills", []) + jd.get("key_signals", [])
    return " ".join(parts) + " Senior AI Engineer embeddings vector search ranking production python"


# ── phase 1: precompute ───────────────────────────────────────────────────────

def precompute(candidates: list[dict], jd: dict, index_dir: str):
    """
    Extract features + embeddings for all candidates. Save to disk.
    This step can use as much time as needed — only Phase 2 must be < 5 min.
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    os.makedirs(index_dir, exist_ok=True)
    print(f"[precompute] Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"[precompute] Extracting features for {len(candidates)} candidates...")
    all_features = []
    texts = []
    for c in tqdm(candidates, desc="features"):
        f = extract_features(c, jd)
        all_features.append(f)
        texts.append(f["text"] or "no profile")

    print("[precompute] Generating embeddings (CPU, batch=128)...")
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,   # needed for cosine similarity via inner product
        convert_to_numpy=True,
    )

    print("[precompute] Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product == cosine when normalized
    index.add(embeddings.astype("float32"))

    # JD embedding
    jd_text = build_jd_text(jd)
    jd_emb = model.encode([jd_text], normalize_embeddings=True, convert_to_numpy=True)

    # Save everything
    faiss.write_index(index, f"{index_dir}/candidates.faiss")
    np.save(f"{index_dir}/embeddings.npy", embeddings)
    np.save(f"{index_dir}/jd_embedding.npy", jd_emb)
    with open(f"{index_dir}/features.pkl", "wb") as f:
        pickle.dump(all_features, f)

    print(f"[precompute] Done. Saved to {index_dir}/")


# ── phase 2: ranking (must run < 5 min, no network) ──────────────────────────

def rank_candidates(candidates: list[dict], jd: dict, index_dir: str) -> list[dict]:
    """
    Load precomputed data, score all candidates, return top 100.
    Must complete within 5 minutes on CPU with no network access.
    """
    import faiss

    print("[rank] Loading precomputed data...")
    index      = faiss.read_index(f"{index_dir}/candidates.faiss")
    embeddings = np.load(f"{index_dir}/embeddings.npy")
    jd_emb     = np.load(f"{index_dir}/jd_embedding.npy")
    with open(f"{index_dir}/features.pkl", "rb") as f:
        all_features = pickle.load(f)

    print("[rank] Computing similarities...")
    # Batch cosine sim: jd_emb (1 x dim) @ embeddings.T (dim x N) = (1 x N)
    sims = (jd_emb @ embeddings.T).squeeze()   # shape: (N,)

    print("[rank] Scoring all candidates...")
    scored = []
    for i, (feat, sim) in enumerate(zip(all_features, sims)):
        s = score_candidate(feat, float(sim))
        scored.append({
            "idx":          i,
            "candidate":    candidates[i],
            "features":     feat,
            "score":        s,
            "semantic_sim": float(sim),
        })

    # Sort descending by score
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Take top 100
    top100 = scored[:100]

    print("[rank] Generating reasoning for top 100...")
    results = []
    for rank_pos, item in enumerate(top100, start=1):
        reasoning = generate_reasoning(
            item["candidate"],
            item["features"],
            rank=rank_pos,
            score=item["score"],
        )
        results.append({
            "candidate_id": item["features"]["candidate_id"],
            "rank":         rank_pos,
            "score":        round(item["score"], 6),
            "reasoning":    reasoning,
        })

    return results


# ── write CSV ─────────────────────────────────────────────────────────────────

def write_csv(results: list[dict], out_path: str):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(results)
    print(f"[rank] Submission saved → {out_path}")


# ── validate output ───────────────────────────────────────────────────────────

def validate(out_path: str):
    import pandas as pd
    df = pd.read_csv(out_path)
    assert len(df) == 100, f"Expected 100 rows, got {len(df)}"
    assert list(df.columns) == ["candidate_id", "rank", "score", "reasoning"]
    assert sorted(df["rank"].tolist()) == list(range(1, 101)), "Ranks must be 1-100 unique"
    assert df["candidate_id"].nunique() == 100, "Duplicate candidate_ids found"
    scores = df["score"].tolist()
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i+1], f"Score not monotonically decreasing at rank {i+2}"
    print("[validate] ✅ All checks passed!")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Redrob candidate ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--jd",         default=DEFAULT_JD_PATH, help="Path to jd_parsed.json")
    parser.add_argument("--out",        default="submission.csv", help="Output CSV path")
    parser.add_argument("--index-dir",  default=DEFAULT_INDEX_DIR, help="Precompute cache directory")
    parser.add_argument("--skip-precompute", action="store_true", help="Skip precompute (use existing cache)")
    parser.add_argument("--sample",     action="store_true", help="Cap at 200 candidates for quick testing")
    args = parser.parse_args()

    t0 = time.time()

    print("[rank] Loading candidates...")
    candidates = load_candidates(args.candidates)
    if args.sample:
        candidates = candidates[:200]
        print(f"[rank] Sample mode: using {len(candidates)} candidates")
    else:
        print(f"[rank] Loaded {len(candidates)} candidates")

    jd = load_jd(args.jd)
    print(f"[rank] JD loaded: {len(jd.get('required_skills', []))} required skills")

    if not args.skip_precompute:
        precompute(candidates, jd, args.index_dir)

    # ── ranking step: must be < 5 min ─────────────────────────────────────────
    t_rank = time.time()
    results = rank_candidates(candidates, jd, args.index_dir)
    print(f"[rank] Ranking completed in {time.time() - t_rank:.1f}s")

    write_csv(results, args.out)
    validate(args.out)

    print(f"[rank] Total time: {time.time() - t0:.1f}s")
    print(f"\nTop 5 candidates:")
    for r in results[:5]:
        print(f"  #{r['rank']} {r['candidate_id']} score={r['score']:.4f}")
        print(f"     {r['reasoning'][:100]}...")


if __name__ == "__main__":
    main()
