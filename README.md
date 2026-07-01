# 🎯 Redrob Intelligent Candidate Ranker

> Hackathon submission for the **Intelligent Candidate Discovery & Ranking Challenge**

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-orange?style=for-the-badge)](https://huggingface.co/spaces/Kaustav27/Candidate_ranker)
[![GitHub](https://img.shields.io/badge/GitHub-Source%20Code-black?style=for-the-badge&logo=github)](https://github.com/KernelKaustav/CandidateRanking-System)
[![Demo Video](https://img.shields.io/badge/▶%20Demo-Google%20Drive-blue?style=for-the-badge)](https://drive.google.com/drive/folders/1ecwfLtnZ3ddXszg52ubKnLTEJvNQKN43?usp=sharing)

---

## 🚀 Live Demo

**HuggingFace Space:** https://huggingface.co/spaces/Kaustav27/Candidate_ranker

**Demo Video:** https://drive.google.com/drive/folders/1ecwfLtnZ3ddXszg52ubKnLTEJvNQKN43?usp=sharing

---

## 📸 Screenshots

### Upload & Run
Upload a JSON/JSONL candidates file, configure scoring weights (Optimal or Manual mode), and hit **Run Ranker**.

![Upload Screen](screenshots/upload.png)

---

### Score Distribution & Component Breakdown
Visual bar chart of score distribution across all ranked candidates, plus a donut chart showing average contribution of each weighted signal.

![Score Distribution](screenshots/score_distribution.png)

---

### Ranked Candidates Table
Full ranked table with Candidate ID, Score, Semantic Similarity, Experience Years, Skills %, Behavioral score, and Active status.

![Ranked Candidates](screenshots/ranked_candidates.png)

---

### JD Understanding & Candidate Evaluation
Parsed job description signals (required skills, nice-to-haves, disqualifiers, ideal profile, locations) alongside all candidate signals used for scoring.

![JD Understanding](screenshots/jd_understanding.png)

---

### Technologies Used
Architecture choices explained — BGE-small, FAISS, Claude API, Python stack, Streamlit, GitHub.

![Technologies](screenshots/technologies.png)

---

## 🏗️ Architecture

```
Job Description (text)
        │
        ▼
┌─────────────────┐     ┌──────────────────────────────────┐
│   jd_parser.py  │────▶│  jd_parsed.json                  │
│  (Claude API,   │     │  (required skills, disqualifiers, │
│   run offline)  │     │   locations, key signals)         │
└─────────────────┘     └──────────────┬───────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────────┐
                        │         rank.py (Phase 1)        │
                        │  • Load 100K candidates           │
                        │  • extract_features() per cand   │
                        │  • BGE-small-en-v1.5 embeddings  │
                        │  • FAISS IndexFlatIP              │
                        │  • Save to precomputed/           │
                        └──────────────┬───────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────────┐
                        │         rank.py (Phase 2)        │  ← must run < 5 min
                        │  • Load FAISS index              │    CPU only, no network
                        │  • Cosine sim: JD vs all cands   │
                        │  • Weighted scoring formula       │
                        │  • Hard penalty multipliers       │
                        │  • Top 100 selected               │
                        │  • Template reasoning generated   │
                        └──────────────┬───────────────────┘
                                       │
                                       ▼
                              submission.csv
```

---

## 📐 Scoring Formula

```
score = (
    0.30 × semantic_similarity      # BGE cosine sim vs JD embedding
  + 0.20 × required_skill_match     # fraction of JD required skills found
  + 0.15 × experience_fit           # years in ideal 5-9 range
  + 0.20 × behavioral_composite     # recency + open_to_work + response_rate + interview_completion
  + 0.05 × nice_skill_match         # nice-to-have skills bonus
  + 0.05 × github_activity          # open source signal
  + 0.05 × title_match              # current title relevance
) × penalty_multiplier
```

**Penalties (multiplicative):**

| Multiplier | Condition |
|---|---|
| `0.50×` | Entire career at consulting firms (TCS, Infosys, Wipro, etc.) |
| `0.80×` | Location mismatch + unwilling to relocate |
| `0.00×` | Honeypot profile detected (timeline impossibilities) |
| `0.60^n×` | n disqualifier keyword hits |

---

## ⚡ Quick Start

### Option A — Full Pipeline (BGE + FAISS, production-grade)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Parse JD (offline, one time) — optional, fallback used if no API key
python jd_parser.py --fallback --out jd_parsed.json

# 3. Run full pipeline (precompute + rank)
python rank.py --candidates candidates.jsonl --out submission.csv

# 4. Rank only (skip precompute if cache exists, must be < 5 min)
python rank.py --candidates candidates.jsonl --out submission.csv --skip-precompute

# 5. Quick sample test
python rank.py --candidates sample_candidates.json --out submission.csv --sample
```

### Option B — Lightweight (no GPU deps, TF-IDF fallback)

```bash
# Only needs: pip install pandas numpy
python rank_lite.py --candidates sample_candidates.json --out submission.csv
python rank_lite.py --candidates sample_candidates.json --out submission.csv --verbose
```

### Run Tests

```bash
python test_pipeline.py
```

Validates: feature extraction, scoring logic, honeypot detection, consulting penalty, CSV format.

### Streamlit Demo (local)

```bash
streamlit run app.py
```

---

## 📁 File Structure

```
├── rank.py                  # Main entry point (BGE + FAISS, production)
├── rank_lite.py             # Lightweight version (TF-IDF, no GPU deps)
├── jd_parser.py             # Claude API JD parsing (run offline)
├── features.py              # Feature extraction from candidate records
├── scorer.py                # Weighted scoring formula + penalties
├── scoring_utils.py         # Custom-weight scorer variant (used by app.py)
├── reasoning.py             # Template-based reasoning generator
├── app.py                   # Streamlit sandbox demo
├── test_pipeline.py         # End-to-end pipeline tests
├── sample_candidates.json   # 10 test candidates (covers all edge cases)
├── jd_parsed.json           # Pre-parsed JD (committed)
├── requirements.txt
├── Candidate_ranker/        # HuggingFace Space (git submodule)
├── precomputed/             # FAISS index cache (created by rank.py Phase 1)
│   ├── candidates.faiss
│   ├── embeddings.npy
│   ├── jd_embedding.npy
│   └── features.pkl
└── submission.csv           # Final output
```

---

## 🧪 Sample Candidates Coverage

| ID | Profile Type | Expected Ranking Behavior |
|---|---|---|
| cand_010 | 9yr Senior AI Eng, full skill match | Rank #1 |
| cand_001 | 7yr ML Eng, Flipkart (product co) | Top 3 |
| cand_002 | 6yr Applied Scientist, Meesho | Top 3 |
| cand_006 | 8yr ML Platform, Nykaa | Top 5 |
| cand_003 | 5.5yr NLP Eng, Qdrant | Mid-tier |
| cand_005 | 5yr Research Scientist, no prod | Penalised (disqualifier hit) |
| cand_004 | 8yr Consultant, TCS+Infosys+Wipro | Penalised 0.50× (consulting) |
| cand_007 | 4yr CV Engineer (vision/speech) | Penalised (CV-only mismatch) |
| cand_008 | 2yr LangChain wrapper only | Low (insufficient exp + skill) |
| cand_009 | Honeypot — impossible timelines | Score = 0.00, rank last |

---

## 📊 Compute Constraints Compliance

| Constraint | Limit | Approach |
|---|---|---|
| Runtime (ranking step) | ≤ 5 min | FAISS load + cosine sim + scoring ≈ 30–60s for 100K |
| Memory | ≤ 16 GB | BGE-small (100K × 384 × 4B) ≈ 150 MB |
| Compute | CPU only | FAISS IndexFlatIP, no CUDA required |
| Network | Off during ranking | All models preloaded; no API calls in Phase 2 |

---

## 🧠 Key Design Decisions

**Why BGE-small over larger models?**
BGE-small-en-v1.5 (22M params, 384 dim) encodes 100K candidates in ~8 min on CPU. Larger models exceed the 5-min ranking budget. Quality is sufficient for candidate–JD matching.

**Why weighted scoring vs pure semantic search?**
Pure embedding similarity is fooled by keyword stuffers. Behavioral signals (`last_active`, `response_rate`) filter "on-paper-perfect but unreachable" candidates — a real-world constraint the JD explicitly highlights.

**Why template reasoning vs LLM per candidate?**
100K × LLM call is impossible within a 5-min CPU budget. Template reasoning is factual (zero hallucination risk), rank-consistent, and specific to each candidate's actual profile data.

**How are honeypots detected?**
Timeline impossibilities (`start_date` before `company_founded_year`) and implausible skill inflation (Expert in many skills with 0 years used) trigger a `0.0×` penalty — zeroing the candidate's score.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Embeddings | BGE-small-en-v1.5 (22M params, 384 dim) |
| Vector Search | FAISS IndexFlatIP (CPU) |
| JD Parsing | Claude API — `claude-sonnet-4-6` (offline) |
| Scoring | Custom weighted formula + penalty multipliers |
| Reasoning | Template-based (zero hallucination risk) |
| Demo UI | Streamlit |
| Hosting | HuggingFace Spaces |

---

*Redrob Hackathon 2026 · BGE-small + FAISS + Weighted Scoring + Template Reasoning*
