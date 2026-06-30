"""
test_pipeline.py
----------------
End-to-end test: runs rank_lite on sample_candidates.json,
validates output CSV, prints a summary report.

No external dependencies beyond pandas + numpy.

Usage:
    python test_pipeline.py
"""

import json
import sys
import subprocess
import os
import csv
from pathlib import Path

ROOT = Path(__file__).parent

def run_tests():
    print("=" * 65)
    print("  REDROB CANDIDATE RANKER — PIPELINE TEST")
    print("=" * 65)

    # 1. Feature extraction unit test
    print("\n[1/4] Testing feature extraction...")
    sys.path.insert(0, str(ROOT))
    from features import extract_features
    from jd_parser import parse_fallback

    jd = parse_fallback()
    with open(ROOT / "sample_candidates.json") as f:
        candidates = json.load(f)

    features = [extract_features(c, jd) for c in candidates]
    assert len(features) == len(candidates), "Feature count mismatch"

    # Spot-check best candidate (cand_010)
    best = next(f for f in features if f["candidate_id"] == "cand_010")
    assert best["required_skill_score"] > 0.7, f"cand_010 should have high skill score, got {best['required_skill_score']:.2f}"
    assert best["all_consulting_career"] == 0, "cand_010 should not be consulting-only"

    # Spot-check honeypot (cand_009)
    honeypot = next(f for f in features if f["candidate_id"] == "cand_009")
    assert honeypot["honeypot_score"] > 0.4, f"cand_009 should have high honeypot score, got {honeypot['honeypot_score']:.2f}"

    # Spot-check consulting-only (cand_004)
    consulting = next(f for f in features if f["candidate_id"] == "cand_004")
    assert consulting["all_consulting_career"] == 1, "cand_004 should be consulting-only"

    print("    ✅  Feature extraction OK")
    print(f"    cand_010 required_skill_score = {best['required_skill_score']:.2f}")
    print(f"    cand_009 honeypot_score        = {honeypot['honeypot_score']:.2f}")
    print(f"    cand_004 all_consulting_career = {consulting['all_consulting_career']}")

    # 2. Scorer unit test
    print("\n[2/4] Testing scorer...")
    from scorer import score_candidate, behavioral_composite, PENALTIES

    score_best = score_candidate(best, semantic_sim=0.85)
    score_honeypot = score_candidate(honeypot, semantic_sim=0.70)
    score_consulting = score_candidate(consulting, semantic_sim=0.65)

    assert score_honeypot == 0.0, f"Honeypot should score 0.0, got {score_honeypot:.4f}"
    assert score_consulting < score_best, "Consulting-only should score below product-company candidate"
    print(f"    ✅  Scorer OK")
    print(f"    cand_010 score = {score_best:.4f}")
    print(f"    cand_004 score = {score_consulting:.4f}  (consulting penalty applied)")
    print(f"    cand_009 score = {score_honeypot:.4f}  (honeypot → zeroed out)")

    # 3. Reasoning unit test
    print("\n[3/4] Testing reasoning generator...")
    from reasoning import generate_reasoning
    r = generate_reasoning(candidates[9], best, rank=1, score=score_best)
    assert isinstance(r, str) and len(r) > 20, "Reasoning should be non-empty string"
    assert "cand_010" not in r or "yrs" in r, "Reasoning should contain profile facts"
    print(f"    ✅  Reasoning OK")
    print(f"    Sample: {r[:120]}...")

    # 4. Full pipeline via rank_lite
    print("\n[4/4] Running full pipeline (rank_lite.py)...")
    out_csv = ROOT / "test_submission.csv"
    result = subprocess.run(
        [sys.executable, str(ROOT / "rank_lite.py"),
         "--candidates", str(ROOT / "sample_candidates.json"),
         "--out", str(out_csv),
         "--verbose"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        print("STDOUT:", result.stdout)
        raise RuntimeError("rank_lite.py failed")

    print(result.stdout)

    # Validate CSV
    with open(out_csv) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == len(candidates), f"Expected {len(candidates)} rows, got {len(rows)}"
    ranks = [int(r["rank"]) for r in rows]
    assert ranks == list(range(1, len(rows) + 1)), "Ranks must be 1-N sequential"

    scores = [float(r["score"]) for r in rows]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], f"Score not descending at rank {i+2}"

    # Honeypot must be last or near-last
    hp_rank = next(int(r["rank"]) for r in rows if r["candidate_id"] == "cand_009")
    assert hp_rank >= len(rows) - 1, f"Honeypot should rank last, got rank {hp_rank}"

    # Consulting-only should be penalised
    consulting_rank = next(int(r["rank"]) for r in rows if r["candidate_id"] == "cand_004")
    best_rank = next(int(r["rank"]) for r in rows if r["candidate_id"] == "cand_010")
    assert best_rank < consulting_rank, "Best candidate should outrank consulting-only"

    print("    ✅  CSV validation passed")
    print(f"    cand_010 (best)       → rank #{best_rank}")
    print(f"    cand_004 (consulting) → rank #{consulting_rank}")
    print(f"    cand_009 (honeypot)   → rank #{hp_rank}")

    print("\n" + "=" * 65)
    print("  ALL TESTS PASSED ✅")
    print("=" * 65)
    print(f"\nSubmission preview saved at: {out_csv}")
    print("\nFirst 5 rows:")
    for row in rows[:5]:
        print(f"  #{row['rank']:>2}  {row['candidate_id']:15s}  score={float(row['score']):.4f}")
        print(f"       {row['reasoning'][:90]}...")


if __name__ == "__main__":
    run_tests()
