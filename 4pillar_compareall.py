import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sentence_transformers import SentenceTransformer

EXPECTED_SHEETS = {
    "RPDC": "RPDC",
    "ONET": "ONET_Activities",
    "ESCO": "ESCO_Skills",
    "ANCHORS": "Pillar_Anchors",
}

COL_RPDC = "EN translation"
COL_ONET_DESC = "Work Activity Description"
COL_ONET_FALLBACK = "Work Activity"
COL_ESCO = "Description"

TOP_N = 15
MIN_WORDS = 2
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_sentence_split = re.compile(r"(?<=[.!?])\s+|\n+")
_spaces = re.compile(r"\s+")


def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.replace("\u00a0", " ")
    return _spaces.sub(" ", s.strip())


def is_valid(s):
    s = clean_text(s)
    return len(s.split()) >= MIN_WORDS


def split_sentences(text):
    text = clean_text(text)
    parts = [p.strip() for p in _sentence_split.split(text) if p.strip()]
    return [p for p in parts if len(p.split()) >= MIN_WORDS]


def embed_norm(model, texts):
    if not texts:
        return np.zeros((0, 384))
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb)


def cosine_max(sent_emb, anchor_emb):
    sims = sent_emb @ anchor_emb.T
    return sims.max(axis=1)


def safe_sheet(xls, wanted):
    names = xls.sheet_names
    if wanted in names:
        return wanted
    low = {n.lower(): n for n in names}
    if wanted.lower() in low:
        return low[wanted.lower()]
    raise ValueError(f"Worksheet '{wanted}' not found. Available: {names}")


def descriptor_dataframe(df, col, system):
    out = df[[col]].copy()
    out = out.rename(columns={col: "descriptor"})
    out["descriptor"] = out["descriptor"].apply(clean_text)
    out = out[out["descriptor"].apply(is_valid)].copy()
    out["System"] = system
    out = out.reset_index(drop=True)
    out["descriptor_id"] = range(1, len(out) + 1)
    return out


def onet_dataframe(df):
    desc = df[COL_ONET_DESC].astype(str).tolist() if COL_ONET_DESC in df.columns else [""] * len(df)
    fallback = df[COL_ONET_FALLBACK].astype(str).tolist() if COL_ONET_FALLBACK in df.columns else [""] * len(df)

    chosen = []
    for d, a in zip(desc, fallback):
        d = clean_text(d)
        a = clean_text(a)
        if d and d.lower() != "nan":
            chosen.append(d)
        else:
            chosen.append(a)

    out = pd.DataFrame({"descriptor": chosen})
    out["descriptor"] = out["descriptor"].apply(clean_text)
    out = out[out["descriptor"].apply(is_valid)]
    out["System"] = "ONET_Activities"
    out = out.reset_index(drop=True)
    out["descriptor_id"] = range(1, len(out) + 1)

    return out


def descriptor_similarity(model, descriptor_df, anchor_emb):

    rows = []

    for _, row in descriptor_df.iterrows():

        descriptor = row["descriptor"]
        segments = split_sentences(descriptor)

        if not segments:
            segments = [descriptor]

        seg_emb = embed_norm(model, segments)
        sims = cosine_max(seg_emb, anchor_emb)

        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]

        rows.append({
            "System": row["System"],
            "descriptor_id": row["descriptor_id"],
            "descriptor": descriptor,
            "best_segment": segments[best_idx],
            "similarity": float(best_sim),
            "n_segments": len(segments)
        })

    return pd.DataFrame(rows)


def main():

    if len(sys.argv) < 2:
        print("Provide Excel filename")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    xls = pd.ExcelFile(file_path)

    sheet_rpdc = safe_sheet(xls, EXPECTED_SHEETS["RPDC"])
    sheet_onet = safe_sheet(xls, EXPECTED_SHEETS["ONET"])
    sheet_esco = safe_sheet(xls, EXPECTED_SHEETS["ESCO"])
    sheet_anchors = safe_sheet(xls, EXPECTED_SHEETS["ANCHORS"])

    rpdc = pd.read_excel(file_path, sheet_name=sheet_rpdc)
    onet = pd.read_excel(file_path, sheet_name=sheet_onet)
    esco = pd.read_excel(file_path, sheet_name=sheet_esco)
    anchors = pd.read_excel(file_path, sheet_name=sheet_anchors)

    rpdc_df = descriptor_dataframe(rpdc, COL_RPDC, "RPDC")
    onet_df = onet_dataframe(onet)
    esco_df = descriptor_dataframe(esco, COL_ESCO, "ESCO_Skills")

    print("Descriptor counts:")
    print("RPDC:", len(rpdc_df))
    print("ONET:", len(onet_df))
    print("ESCO:", len(esco_df))

    model = SentenceTransformer(MODEL_NAME)

    anchors["Pillar"] = anchors["Pillar"].astype(str).str.strip()
    anchors["Anchor_Text"] = anchors["Anchor_Text"].apply(clean_text)
    anchors = anchors[anchors["Anchor_Text"].apply(is_valid)]

    summary_rows = []
    all_rows = []
    top_rows = []

    systems = [rpdc_df, onet_df, esco_df]

    for pillar in sorted(anchors["Pillar"].unique()):

        anchor_texts = anchors.loc[anchors["Pillar"] == pillar, "Anchor_Text"].tolist()
        anchor_emb = embed_norm(model, anchor_texts)

        for df in systems:

            scored = descriptor_similarity(model, df, anchor_emb)
            scored["Pillar"] = pillar

            sims = scored["similarity"].values

            summary_rows.append({
                "Pillar": pillar,
                "System": scored["System"].iloc[0],
                "n_descriptors": len(sims),
                "mean_similarity": np.mean(sims),
                "median_similarity": np.median(sims),
                "std_similarity": np.std(sims, ddof=1),
                "min_similarity": np.min(sims),
                "max_similarity": np.max(sims),
                "p10_similarity": np.percentile(sims, 10),
                "p90_similarity": np.percentile(sims, 90)
            })

            scored_sorted = scored.sort_values("similarity", ascending=False)

            for rank, row in enumerate(scored_sorted.head(TOP_N).itertuples(), 1):
                top_rows.append({
                    "Pillar": pillar,
                    "System": row.System,
                    "rank": rank,
                    "similarity": row.similarity,
                    "descriptor": row.descriptor,
                    "best_segment": row.best_segment
                })

            all_rows.append(scored)

    summary_df = pd.DataFrame(summary_rows)
    top_df = pd.DataFrame(top_rows)
    all_scores_df = pd.concat(all_rows)

    stem = file_path.stem

    summary_df.to_excel(f"{stem}_summary_allstats.xlsx", index=False)
    top_df.to_excel(f"{stem}_top15_matches.xlsx", index=False)
    all_scores_df.to_excel(f"{stem}_all_descriptor_scores.xlsx", index=False)

    print("Files saved")

    pivot = summary_df.pivot(index="Pillar", columns="System", values="mean_similarity")
    pivot.plot(kind="bar", figsize=(10,5))
    plt.title("Mean similarity by pillar and system")
    plt.tight_layout()
    plt.savefig(f"{stem}_mean_similarity_plot.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()