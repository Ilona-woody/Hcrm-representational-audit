"""
Sensitivity Analysis for Representational Audit
================================================
Tests whether the main finding (O*NET > RPDC > ESCO) holds under:
  1. Mean vs Max aggregation across segments/anchors
  2. Alternative embedding model (all-mpnet-base-v2 vs all-MiniLM-L6-v2)

USAGE:
    py sensitivity_analysis.py Electrician_data.xlsx Dog_Groomer_data.xlsx Massage_Therapist_data.xlsx

OUTPUT FILES:
    sensitivity_mean_vs_max.xlsx      <- Analysis 1 results
    sensitivity_model_comparison.xlsx <- Analysis 2 results
    sensitivity_ranking_table.xlsx    <- Summary: does ranking hold? (for paper)
    sensitivity_ranking_table.csv     <- Same as CSV
"""

import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Sheet / column config (same as original script) ──────────────────────────
EXPECTED_SHEETS = {
    "RPDC":    "RPDC",
    "ONET":    "ONET_Activities",
    "ESCO":    "ESCO_Skills",
    "ANCHORS": "Pillar_Anchors",
}

COL_RPDC       = "EN translation"
COL_ONET_DESC  = "Work Activity Description"
COL_ONET_FALLBACK = "Work Activity"
COL_ESCO       = "Description"

MIN_WORDS = 2

# ── Models to compare ─────────────────────────────────────────────────────────
MODEL_ORIGINAL    = "sentence-transformers/all-MiniLM-L6-v2"   # original
MODEL_ALTERNATIVE = "sentence-transformers/all-mpnet-base-v2"   # sensitivity check

SYSTEM_ORDER = ["RPDC", "O*NET", "ESCO"]

_sentence_split = re.compile(r"(?<=[.!?])\s+|\n+")
_spaces         = re.compile(r"\s+")


# ── Text utilities ────────────────────────────────────────────────────────────
def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.replace("\u00a0", " ")
    return _spaces.sub(" ", s.strip())


def is_valid(s):
    return len(clean_text(s).split()) >= MIN_WORDS


def split_sentences(text):
    text = clean_text(text)
    parts = [p.strip() for p in _sentence_split.split(text) if p.strip()]
    return [p for p in parts if len(p.split()) >= MIN_WORDS]


def embed_norm(model, texts):
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()))
    return np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False))


def safe_sheet(xls, wanted):
    names = xls.sheet_names
    if wanted in names:
        return wanted
    low = {n.lower(): n for n in names}
    if wanted.lower() in low:
        return low[wanted.lower()]
    raise ValueError(f"Worksheet '{wanted}' not found. Available: {names}")


# ── Data loading (identical to original) ─────────────────────────────────────
def descriptor_dataframe(df, col, system):
    out = df[[col]].copy().rename(columns={col: "descriptor"})
    out["descriptor"] = out["descriptor"].apply(clean_text)
    out = out[out["descriptor"].apply(is_valid)].copy()
    out["System"] = system
    out = out.reset_index(drop=True)
    out["descriptor_id"] = range(1, len(out) + 1)
    return out


def onet_dataframe(df):
    desc     = df[COL_ONET_DESC].astype(str).tolist()     if COL_ONET_DESC     in df.columns else [""] * len(df)
    fallback = df[COL_ONET_FALLBACK].astype(str).tolist() if COL_ONET_FALLBACK in df.columns else [""] * len(df)
    chosen   = [clean_text(d) if clean_text(d) and clean_text(d).lower() != "nan"
                else clean_text(a) for d, a in zip(desc, fallback)]
    out = pd.DataFrame({"descriptor": chosen})
    out["descriptor"] = out["descriptor"].apply(clean_text)
    out = out[out["descriptor"].apply(is_valid)]
    out["System"] = "ONET_Activities"
    out = out.reset_index(drop=True)
    out["descriptor_id"] = range(1, len(out) + 1)
    return out


def load_occupation_data(file_path):
    """Load all sheets from one occupation Excel file."""
    xls = pd.ExcelFile(file_path)
    rpdc    = pd.read_excel(file_path, sheet_name=safe_sheet(xls, EXPECTED_SHEETS["RPDC"]))
    onet    = pd.read_excel(file_path, sheet_name=safe_sheet(xls, EXPECTED_SHEETS["ONET"]))
    esco    = pd.read_excel(file_path, sheet_name=safe_sheet(xls, EXPECTED_SHEETS["ESCO"]))
    anchors = pd.read_excel(file_path, sheet_name=safe_sheet(xls, EXPECTED_SHEETS["ANCHORS"]))

    rpdc_df = descriptor_dataframe(rpdc, COL_RPDC, "RPDC")
    onet_df = onet_dataframe(onet)
    esco_df = descriptor_dataframe(esco, COL_ESCO, "ESCO_Skills")

    anchors["Pillar"]       = anchors["Pillar"].astype(str).str.strip()
    anchors["Anchor_Text"]  = anchors["Anchor_Text"].apply(clean_text)
    anchors = anchors[anchors["Anchor_Text"].apply(is_valid)]

    return rpdc_df, onet_df, esco_df, anchors


# ── Core similarity computation ───────────────────────────────────────────────
def compute_similarity_scores(model, descriptor_df, anchor_emb, aggregation="max"):
    """
    Compute descriptor-level similarity scores.

    aggregation="max"  : original method — max over segments AND anchors
    aggregation="mean" : sensitivity check — mean over all segment-anchor pairs
    """
    rows = []
    for _, row in descriptor_df.iterrows():
        descriptor = row["descriptor"]
        segments   = split_sentences(descriptor) or [descriptor]
        seg_emb    = embed_norm(model, segments)

        # seg_emb: (n_segments, dim)   anchor_emb: (n_anchors, dim)
        sims_matrix = seg_emb @ anchor_emb.T   # shape: (n_segments, n_anchors)

        if aggregation == "max":
            score = float(sims_matrix.max())
        else:  # mean
            score = float(sims_matrix.mean())

        rows.append({
            "System":        row["System"],
            "descriptor_id": row["descriptor_id"],
            "descriptor":    descriptor,
            "similarity":    score,
        })
    return pd.DataFrame(rows)


def run_summary(model, systems, anchors_df, occupation_name, aggregation="max"):
    """Run full pillar × system comparison and return summary rows."""
    summary_rows = []

    for pillar in sorted(anchors_df["Pillar"].unique()):
        anchor_texts = anchors_df.loc[anchors_df["Pillar"] == pillar, "Anchor_Text"].tolist()
        anchor_emb   = embed_norm(model, anchor_texts)

        for df in systems:
            scored = compute_similarity_scores(model, df, anchor_emb, aggregation)
            scored["Pillar"] = pillar
            sims = scored["similarity"].values

            sys_label = df["System"].iloc[0]
            if sys_label == "ONET_Activities":
                sys_label = "O*NET"
            elif sys_label == "ESCO_Skills":
                sys_label = "ESCO"

            summary_rows.append({
                "Occupation":      occupation_name,
                "Pillar":          pillar,
                "System":          sys_label,
                "n_descriptors":   len(sims),
                "mean_similarity": round(float(np.mean(sims)), 4),
            })

    return pd.DataFrame(summary_rows)


def ranking_holds(df_summary):
    """
    For each Occupation × Pillar, check whether RPDC < O*NET and ESCO < O*NET.
    Returns a summary string.
    """
    results = []
    for (occ, pillar), grp in df_summary.groupby(["Occupation", "Pillar"]):
        row = {s: grp.loc[grp["System"] == s, "mean_similarity"].values[0]
               for s in SYSTEM_ORDER if s in grp["System"].values}
        onet_highest = (row.get("O*NET", 0) >= row.get("RPDC", 0) and
                        row.get("O*NET", 0) >= row.get("ESCO", 0))
        rpdc_mid     = (row.get("RPDC", 0) >= row.get("ESCO", 0))
        results.append({
            "Occupation":    occ,
            "Pillar":        pillar,
            "RPDC":          round(row.get("RPDC", np.nan), 4),
            "O*NET":         round(row.get("O*NET", np.nan), 4),
            "ESCO":          round(row.get("ESCO", np.nan), 4),
            "O*NET_highest": onet_highest,
            "RPDC>ESCO":     rpdc_mid,
            "Ranking_holds": onet_highest and rpdc_mid,
        })
    return pd.DataFrame(results)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: py sensitivity_analysis.py file1.xlsx [file2.xlsx ...]")
        sys.exit(1)

    input_files = [Path(a) for a in sys.argv[1:]]
    missing = [str(f) for f in input_files if not f.exists()]
    if missing:
        raise FileNotFoundError("Files not found:\n" + "\n".join(missing))

    out_dir = input_files[0].parent

    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS")
    print("="*60)

    # ── Load all occupation data ───────────────────────────────────
    occupation_data = {}
    for f in input_files:
        occ_name = f.stem.replace("_", " ").strip()
        print(f"\nLoading: {f.name}  →  '{occ_name}'")
        occupation_data[occ_name] = load_occupation_data(f)

    # ── ANALYSIS 1: Mean vs Max (original model) ───────────────────
    print("\n" + "-"*60)
    print("ANALYSIS 1: Mean vs Max aggregation  |  Model: all-MiniLM-L6-v2")
    print("-"*60)

    model_orig = SentenceTransformer(MODEL_ORIGINAL)

    max_frames  = []
    mean_frames = []

    for occ_name, (rpdc_df, onet_df, esco_df, anchors_df) in occupation_data.items():
        systems = [rpdc_df, onet_df, esco_df]
        print(f"  Processing (max):  {occ_name}")
        max_frames.append(run_summary(model_orig, systems, anchors_df, occ_name, aggregation="max"))
        print(f"  Processing (mean): {occ_name}")
        mean_frames.append(run_summary(model_orig, systems, anchors_df, occ_name, aggregation="mean"))

    df_max  = pd.concat(max_frames,  ignore_index=True)
    df_mean = pd.concat(mean_frames, ignore_index=True)

    df_max["Aggregation"]  = "max (original)"
    df_mean["Aggregation"] = "mean (sensitivity)"

    df_analysis1 = pd.concat([df_max, df_mean], ignore_index=True)

    rank_max  = ranking_holds(df_max)
    rank_mean = ranking_holds(df_mean)
    rank_max["Aggregation"]  = "max (original)"
    rank_mean["Aggregation"] = "mean (sensitivity)"

    # ── ANALYSIS 2: Alternative model (max aggregation) ───────────
    print("\n" + "-"*60)
    print("ANALYSIS 2: Alternative model  |  all-mpnet-base-v2  |  max aggregation")
    print("-"*60)

    model_alt = SentenceTransformer(MODEL_ALTERNATIVE)
    alt_frames = []

    for occ_name, (rpdc_df, onet_df, esco_df, anchors_df) in occupation_data.items():
        systems = [rpdc_df, onet_df, esco_df]
        print(f"  Processing: {occ_name}")
        alt_frames.append(run_summary(model_alt, systems, anchors_df, occ_name, aggregation="max"))

    df_alt = pd.concat(alt_frames, ignore_index=True)
    df_alt["Aggregation"] = "max + mpnet (sensitivity)"

    rank_alt = ranking_holds(df_alt)
    rank_alt["Aggregation"] = "max + mpnet (sensitivity)"

    # ── Build master ranking table ─────────────────────────────────
    rank_all = pd.concat([rank_max, rank_mean, rank_alt], ignore_index=True)

    # Summary counts
    summary_counts = rank_all.groupby("Aggregation")["Ranking_holds"].agg(
        Holds="sum", Total="count"
    ).reset_index()
    summary_counts["Holds_pct"] = (summary_counts["Holds"] / summary_counts["Total"] * 100).round(1)
    summary_counts.columns = ["Condition", "Ranking holds (n)", "Total combinations", "% holds"]

    # ── Save outputs ───────────────────────────────────────────────
    out1 = out_dir / "sensitivity_mean_vs_max.xlsx"
    out2 = out_dir / "sensitivity_model_comparison.xlsx"
    out3 = out_dir / "sensitivity_ranking_table.xlsx"
    out4 = out_dir / "sensitivity_ranking_table.csv"

    with pd.ExcelWriter(out1, engine="openpyxl") as w:
        df_analysis1.to_excel(w, sheet_name="all_scores",      index=False)
        rank_max.to_excel(    w, sheet_name="ranking_max",     index=False)
        rank_mean.to_excel(   w, sheet_name="ranking_mean",    index=False)

    df_model_compare = pd.concat([df_max.assign(Aggregation="max_miniLM"),
                                   df_alt.assign(Aggregation="max_mpnet")], ignore_index=True)
    with pd.ExcelWriter(out2, engine="openpyxl") as w:
        df_model_compare.to_excel(w, sheet_name="all_scores",     index=False)
        rank_max.to_excel(         w, sheet_name="ranking_miniLM", index=False)
        rank_alt.to_excel(         w, sheet_name="ranking_mpnet",  index=False)

    with pd.ExcelWriter(out3, engine="openpyxl") as w:
        rank_all.to_excel(      w, sheet_name="per_combination", index=False)
        summary_counts.to_excel(w, sheet_name="summary",         index=False)

    summary_counts.to_csv(out4, index=False, encoding="utf-8-sig")

    # ── Print results to console ───────────────────────────────────
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(summary_counts.to_string(index=False))

    print("\nDetailed ranking check (all conditions):")
    cols_show = ["Occupation", "Pillar", "RPDC", "O*NET", "ESCO",
                 "O*NET_highest", "RPDC>ESCO", "Ranking_holds", "Aggregation"]
    print(rank_all[cols_show].to_string(index=False))

    print(f"\nSaved:\n - {out1.name}\n - {out2.name}\n - {out3.name}\n - {out4.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
