"""
app.py — Redrob Candidate Ranker (v4)
"""

import json
import math
import re
import os
import sys
import time
from collections import Counter

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="🎯", layout="wide")

st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-label { font-size: 12px; color: #6c757d; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-value { font-size: 28px; font-weight: 700; color: #1a1a2e; }
.metric-sub   { font-size: 11px; color: #adb5bd; margin-top: 2px; }
.score-bar-wrap { background:#f0f0f0; border-radius:4px; height:6px; margin-top:4px; }
.score-bar    { background: linear-gradient(90deg,#667eea,#764ba2); border-radius:4px; height:6px; }
.tag          { display:inline-block; background:#e3f2fd; color:#1565c0; border-radius:4px; padding:1px 7px; font-size:11px; margin:2px; }
.tag-warn     { background:#fff3e0; color:#e65100; }
.tag-ok       { background:#e8f5e9; color:#2e7d32; }
.honeypot-alert { background:#fff3cd; border:1px solid #ffc107; border-radius:8px; padding:8px 12px; font-size:13px; color:#856404; }
.explain-row  { display:flex; align-items:center; gap:8px; margin:6px 0; }
.explain-label { width:90px; font-size:11px; color:#6c757d; flex-shrink:0; }
.explain-track { flex:1; background:#e9ecef; border-radius:4px; height:8px; position:relative; }
.explain-fill  { border-radius:4px; height:8px; }
.explain-val   { font-size:11px; font-weight:600; color:#1a1a2e; min-width:34px; text-align:right; }
</style>
""", unsafe_allow_html=True)

# ── TF-IDF helpers ────────────────────────────────────────────────────────────

JD_TEXT = """senior ai engineer embeddings retrieval ranking sentence transformers
bge vector database pinecone weaviate qdrant milvus faiss opensearch elasticsearch
hybrid search python production ndcg mrr evaluation framework ab testing lora peft
fine tuning nlp recommendation information retrieval"""

OPTIMAL_WEIGHTS = {
    "semantic_sim":     0.25,
    "required_skill":   0.18,
    "career_relevance": 0.12,
    "experience_fit":   0.09,
    "behavioral":       0.13,
    "nice_skill":       0.04,
    "github_activity":  0.04,
    "title_match":      0.02,
}

WEIGHT_LABELS = {
    "semantic_sim":     "Semantic similarity",
    "required_skill":   "Required skills match",
    "career_relevance": "Career relevance",
    "experience_fit":   "Experience fit",
    "behavioral":       "Behavioral signals",
    "nice_skill":       "Nice-to-have skills",
    "github_activity":  "GitHub activity",
    "title_match":      "Title match",
}

# Tidepool-style fixed categorical colors for the donut/breakdown
WEIGHT_COLORS = {
    "semantic_sim":     "#2a78d6",
    "required_skill":   "#1baf7a",
    "career_relevance": "#eda100",
    "experience_fit":   "#008300",
    "behavioral":       "#4a3aa7",
    "nice_skill":       "#e34948",
    "github_activity":  "#e87ba4",
    "title_match":      "#eb6834",
}

def tokenize(t):
    return re.findall(r"[a-z0-9]+", t.lower())

def build_idf(docs):
    N = len(docs)
    df = {}
    for toks in docs:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    return {t: math.log((N + 1) / (c + 1)) + 1 for t, c in df.items()}

def tfidf_vec(toks, idf):
    tf = Counter(toks)
    total = max(len(toks), 1)
    return {t: (c / total) * idf.get(t, 1.0) for t, c in tf.items()}

def cosine(a, b):
    shared = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in shared)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0

def explain_bar(label, value, max_value=1.0, color="#667eea", fmt="{:.2f}"):
    """Render a single explainability progress bar (HTML)."""
    pct = max(0, min(100, (value / max_value) * 100)) if max_value else 0
    val_str = fmt.format(value)
    return f"""
<div class="explain-row">
  <div class="explain-label">{label}</div>
  <div class="explain-track"><div class="explain-fill" style="width:{pct}%;background:{color}"></div></div>
  <div class="explain-val">{val_str}</div>
</div>"""

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Upload candidates")
    uploaded = st.file_uploader(
        "JSON or JSONL · max 100 candidates",
        type=["json", "jsonl"],
    )

    candidates = []
    if uploaded:
        raw = uploaded.read().decode("utf-8")
        try:
            data = json.loads(raw)
            candidates = data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            candidates = [json.loads(l) for l in raw.splitlines() if l.strip()]
        candidates = candidates[:100]
        st.success(f"{len(candidates)} candidates loaded")

        st.markdown("---")
        st.markdown("## Scoring weights")

        weight_mode = st.radio(
            "Mode",
            ["Optimal (recommended)", "Manual"],
            index=0,
        )

        if weight_mode == "Manual":
            st.caption("Sliders are independent — no need to sum to 1.")
            w = {}
            for key, label in WEIGHT_LABELS.items():
                w[key] = st.slider(label, 0.0, 1.0, OPTIMAL_WEIGHTS[key], 0.01, key=f"w_{key}")
            total = sum(w.values())
            if 0.95 <= total <= 1.05:
                st.success(f"Weight sum: {total:.2f}")
            else:
                st.warning(f"Weight sum: {total:.2f} (ideally ~1.0)")
        else:
            w = OPTIMAL_WEIGHTS.copy()
            st.caption("Tuned for this JD's priorities.")
            for key, label in WEIGHT_LABELS.items():
                pct = int(OPTIMAL_WEIGHTS[key] * 100)
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0'>"
                    f"<div style='width:90px;font-size:11px;color:#6c757d'>{label}</div>"
                    f"<div style='flex:1;background:#e9ecef;border-radius:4px;height:6px'>"
                    f"<div style='width:{pct*4}px;max-width:100%;background:{WEIGHT_COLORS[key]};border-radius:4px;height:6px'></div></div>"
                    f"<div style='font-size:11px;font-weight:600;color:#1a1a2e;min-width:26px'>{pct}%</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

# ── Main ──────────────────────────────────────────────────────────────────────

if not candidates:
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2.5, 1])
    with c2:
        st.markdown("""
<div style="text-align:center; margin-bottom: 2rem;">
  <div style="font-size:4.5rem; margin-bottom:0.5rem;">🎯</div>
  <h1 style="font-size:2rem; font-weight:800; margin:0 0 0.4rem;">Redrob Candidate Ranker</h1>
  <p style="font-size:1rem; color:#6c757d; margin:0">
    Upload a candidate file on the left to get started.
  </p>
</div>
""", unsafe_allow_html=True)

        f1, f2, f3 = st.columns(3)
        for col, icon, title, body in [
            (f1, "🧠", "Smart scoring", "23 behavioral signals + semantic similarity + career depth"),
            (f2, "⚡", "CPU-only", "TF-IDF similarity — no GPU, no sentence-transformers"),
            (f3, "🛡️", "Honeypot safe", "Fake profiles automatically detected and zeroed out"),
        ]:
            with col:
                st.markdown(f"""
<div style="background:#f8f9fa;border:1px solid #e9ecef;border-radius:12px;padding:1rem;text-align:center;height:140px;display:flex;flex-direction:column;justify-content:center">
  <div style="font-size:1.8rem;margin-bottom:0.4rem">{icon}</div>
  <div style="font-weight:700;font-size:0.85rem;margin-bottom:0.3rem">{title}</div>
  <div style="font-size:0.75rem;color:#6c757d;line-height:1.4">{body}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### How it works")
        steps = [
            ("1", "Upload", "Drop your candidates JSON on the left"),
            ("2", "Extract", "Features pulled: skills depth, career history, 23 redrob signals"),
            ("3", "Score",  "Weighted formula + honeypot/consulting penalties applied"),
            ("4", "Rank",   "Top 100 sorted, reasoning generated, CSV ready to submit"),
        ]
        cols = st.columns(4)
        for col, (num, title, body) in zip(cols, steps):
            with col:
                st.markdown(f"""
<div style="text-align:center;padding:0.75rem 0.5rem">
  <div style="width:32px;height:32px;border-radius:50%;background:#667eea;color:#fff;font-weight:700;font-size:14px;display:flex;align-items:center;justify-content:center;margin:0 auto 0.5rem">{num}</div>
  <div style="font-weight:600;font-size:0.8rem;margin-bottom:0.25rem">{title}</div>
  <div style="font-size:0.7rem;color:#6c757d;line-height:1.4">{body}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
<div style="text-align:center;color:#adb5bd;font-size:0.78rem">
  Redrob Intelligent Candidate Discovery Challenge · Team <strong>elabs</strong>
</div>""", unsafe_allow_html=True)

else:
    st.markdown(f"""
<div style="background:#e8f4fd;border:1px solid #90caf9;border-radius:8px;padding:10px 16px;display:flex;align-items:center;gap:10px;margin-bottom:1rem">
  <span style="font-size:1.2rem">📂</span>
  <span style="font-size:14px;color:#1565c0;font-weight:500">{len(candidates)} candidate profiles loaded and ready to rank.</span>
</div>
""", unsafe_allow_html=True)

    run = st.button("Run ranker", type="primary", use_container_width=True)

    if run:
        progress = st.progress(0, text="Extracting features...")
        with st.spinner(""):
            t0 = time.time()
            try:
                sys.path.insert(0, os.path.dirname(__file__))
                from features import extract_features
                from reasoning import generate_reasoning
                from scorer import behavioral_composite, PENALTIES

                jd = {"required_skills": [], "nice_to_have_skills": [], "disqualifiers": []}

                all_features = []
                for i, c in enumerate(candidates):
                    all_features.append(extract_features(c, jd))
                    progress.progress((i + 1) / len(candidates), text=f"Extracting features… {i+1}/{len(candidates)}")

                progress.progress(0.7, text="Computing similarity…")
                texts = [f["text"] or "empty" for f in all_features]
                jd_toks = tokenize(JD_TEXT)
                cand_toks = [tokenize(t) for t in texts]
                idf = build_idf([jd_toks] + cand_toks)
                jd_vec = tfidf_vec(jd_toks, idf)
                sims = [cosine(jd_vec, tfidf_vec(toks, idf)) for toks in cand_toks]

                progress.progress(0.85, text="Scoring candidates…")
                scored = []
                honeypot_count = 0
                consulting_count = 0
                rejected_count = 0
                for i, (feat, sim) in enumerate(zip(all_features, sims)):
                    component_raw = {
                        "semantic_sim":     float(sim),
                        "required_skill":   feat.get("required_skill_score", 0),
                        "career_relevance": feat.get("career_relevance", 0),
                        "experience_fit":   feat.get("exp_fit", 0),
                        "behavioral":       behavioral_composite(feat),
                        "nice_skill":       feat.get("nice_skill_score", 0),
                        "github_activity":  feat.get("github_activity", 0),
                        "title_match":      feat.get("title_ml_match", 0),
                    }

                    if feat.get("honeypot_score", 0) > 0.5:
                        score = 0.0
                        honeypot_count += 1
                        rejected_count += 1
                    else:
                        raw = sum(w[k] * v for k, v in component_raw.items())
                        mult = 1.0
                        if feat.get("all_consulting_career"):
                            mult *= PENALTIES["consulting_only"]
                            consulting_count += 1
                        disq = int(feat.get("disqualifier_hit_count", 0))
                        for _ in range(disq):
                            mult *= PENALTIES["disqualifier"]
                        if mult < 1.0:
                            rejected_count += 1
                        score = raw * mult

                    scored.append({
                        "candidate": candidates[i],
                        "features": feat,
                        "score": score,
                        "sim": float(sim),
                        "components": component_raw,
                    })

                scored.sort(key=lambda x: (-x["score"], x["features"]["candidate_id"]))
                top_n = scored[:min(100, len(scored))]

                progress.progress(0.95, text="Generating reasoning…")
                rows = []
                for rank_pos, item in enumerate(top_n, 1):
                    r = generate_reasoning(item["candidate"], item["features"], rank_pos, item["score"])
                    beh = behavioral_composite(item["features"])
                    rows.append({
                        "Rank":        rank_pos,
                        "Candidate ID": item["features"]["candidate_id"],
                        "Score":       round(item["score"], 4),
                        "Sim":         round(item["sim"], 3),
                        "Exp yrs":     round(item["features"].get("years_exp", 0), 1),
                        "Skills %":    int(item["features"].get("required_skill_score", 0) * 100),
                        "Behavioral":  round(beh, 2),
                        "Active":      "✅" if item["features"].get("recency_score", 0) > 0.5 else "⚠️",
                        "Reasoning":   r,
                        "_consulting": item["features"].get("all_consulting_career", 0),
                        "_honeypot":   item["features"].get("honeypot_score", 0),
                        "_components": item["components"],
                    })

                elapsed = time.time() - t0
                progress.progress(1.0, text="Done!")
                time.sleep(0.3)
                progress.empty()

                # ── KPI cards (highest priority) ───────────────────────────────
                k1, k2, k3, k4 = st.columns(4)
                top_score = rows[0]["Score"] if rows else 0
                with k1:
                    st.metric("👥 Candidates", len(candidates))
                with k2:
                    st.metric("🏆 Top score", f"{top_score:.4f}")
                with k3:
                    st.metric("⏱ Runtime", f"{elapsed:.2f}s")
                with k4:
                    st.metric("🚫 Rejected", rejected_count)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Alerts ────────────────────────────────────────────────────
                if honeypot_count > 0:
                    st.markdown(f"""
<div class="honeypot-alert">
  ⚠️ <strong>{honeypot_count} honeypot profile(s) detected</strong> — scored 0.0 and excluded from top 100.
</div><br>""", unsafe_allow_html=True)

                if consulting_count > 0:
                    st.info(f"ℹ️ {consulting_count} candidate(s) penalised for consulting-only career history.")

                # ── Score distribution ────────────────────────────────────────
                st.markdown("### Score distribution")
                df = pd.DataFrame(rows)
                hist_df = df[["Rank", "Score"]].set_index("Rank")
                st.bar_chart(hist_df, height=220, color="#667eea")

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Score breakdown donut (average across top 100) ─────────────
                st.markdown("### Score component breakdown")
                st.caption("Average contribution of each weighted component across the ranked set.")

                avg_components = {}
                for key in WEIGHT_LABELS:
                    avg_components[key] = sum(r["_components"][key] * w[key] for r in rows) / max(len(rows), 1)
                total_contrib = sum(avg_components.values()) or 1.0

                dc1, dc2 = st.columns([1, 1])

                with dc1:
                    # Donut via Chart.js
                    labels_js = json.dumps([WEIGHT_LABELS[k] for k in WEIGHT_LABELS])
                    data_js = json.dumps([round(avg_components[k], 4) for k in WEIGHT_LABELS])
                    colors_js = json.dumps([WEIGHT_COLORS[k] for k in WEIGHT_LABELS])

                    chart_html = f"""
<div style="position:relative;width:100%;height:260px">
<canvas id="donutChart" role="img" aria-label="Donut chart of score component contributions">Score breakdown by component</canvas>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
new Chart(document.getElementById('donutChart'), {{
  type: 'doughnut',
  data: {{
    labels: {labels_js},
    datasets: [{{ data: {data_js}, backgroundColor: {colors_js}, borderWidth: 2, borderColor: '#fff' }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }}
  }}
}});
</script>
"""
                    st.components.v1.html(chart_html, height=270)

                with dc2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    for key in WEIGHT_LABELS:
                        pct_of_total = (avg_components[key] / total_contrib) * 100
                        st.markdown(
                            f"<div style='display:flex;align-items:center;gap:8px;margin:5px 0;font-size:13px'>"
                            f"<span style='width:10px;height:10px;border-radius:2px;background:{WEIGHT_COLORS[key]};display:inline-block'></span>"
                            f"<span style='flex:1;color:#444'>{WEIGHT_LABELS[key]}</span>"
                            f"<span style='font-weight:600;color:#1a1a2e'>{pct_of_total:.0f}%</span>"
                            f"</div>", unsafe_allow_html=True
                        )

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Results table ─────────────────────────────────────────────
                st.markdown("### Ranked candidates")
                display_df = df.drop(columns=["Reasoning", "_consulting", "_honeypot", "_components"])
                st.dataframe(display_df, use_container_width=True, height=420)

                # ── Top 10 cards with explainability bars ───────────────────────
                st.markdown("### Top 10 candidates")
                for row in rows[:10]:
                    score_pct = min(int(row["Score"] * 200), 100)
                    active_tag = '<span class="tag tag-ok">active</span>' if row["Active"] == "✅" else '<span class="tag tag-warn">inactive</span>'
                    consulting_tag = '<span class="tag tag-warn">consulting</span>' if row["_consulting"] else ""

                    comp = row["_components"]
                    explain_html = "".join([
                        explain_bar("Skills", comp["required_skill"] * 100, 100, WEIGHT_COLORS["required_skill"], "{:.0f}"),
                        explain_bar("Behavior", comp["behavioral"], 1.0, WEIGHT_COLORS["behavioral"], "{:.2f}"),
                        explain_bar("Career fit", comp["career_relevance"], 1.0, WEIGHT_COLORS["career_relevance"], "{:.2f}"),
                        explain_bar("Semantic sim", comp["semantic_sim"], 1.0, WEIGHT_COLORS["semantic_sim"], "{:.2f}"),
                        explain_bar("Experience", comp["experience_fit"], 1.0, WEIGHT_COLORS["experience_fit"], "{:.2f}"),
                    ])

                    with st.expander(f"#{row['Rank']}  {row['Candidate ID']}  —  score {row['Score']}"):
                        st.markdown(f"""
<div style="display:flex;gap:1.5rem;align-items:flex-start">
  <div style="flex:1">
    <div style="margin-bottom:8px">
      {active_tag}
      <span class="tag">{row['Exp yrs']}yr exp</span>
      <span class="tag">skills {row['Skills %']}%</span>
      {consulting_tag}
    </div>
    <div style="font-size:13px;color:#444;line-height:1.6;margin-bottom:10px">{row['Reasoning']}</div>
    {explain_html}
  </div>
  <div style="min-width:110px;text-align:center">
    <div style="font-size:22px;font-weight:700;color:#1a1a2e">{row['Score']}</div>
    <div class="score-bar-wrap"><div class="score-bar" style="width:{score_pct}%"></div></div>
    <div style="font-size:10px;color:#adb5bd;margin-top:4px">match score</div>
  </div>
</div>""", unsafe_allow_html=True)

                # ── Download ──────────────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                csv_data = (
                    df[["Candidate ID", "Rank", "Score", "Reasoning"]]
                    .rename(columns={
                        "Candidate ID": "candidate_id",
                        "Rank": "rank",
                        "Score": "score",
                        "Reasoning": "reasoning",
                    })
                    .to_csv(index=False)
                )
                st.download_button(
                    "Download submission CSV",
                    csv_data,
                    "elabs_submission.csv",
                    "text/csv",
                    use_container_width=True,
                )

            except Exception as e:
                progress.empty()
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

st.markdown("---")
st.caption("Redrob Hackathon 2026 · Team elabs · TF-IDF + Weighted Scoring + Behavioral Signals")