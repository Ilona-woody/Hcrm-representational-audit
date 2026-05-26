import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── same column names as your main script ──────────────────────
COL_RPDC      = "EN translation"
COL_ONET_DESC = "Work Activity Description"
COL_ONET_FALL = "Work Activity"
COL_ESCO      = "Description"
MIN_WORDS     = 2

_spaces = re.compile(r"\s+")

def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.replace("\u00a0", " ")
    return _spaces.sub(" ", s.strip())

def word_count(s):
    return len(clean_text(s).split())

def is_valid(s):
    return word_count(s) >= MIN_WORDS

def load_descriptors(file_path):
    xls   = pd.ExcelFile(file_path)
    names = {n.lower(): n for n in xls.sheet_names}

    rpdc  = pd.read_excel(file_path,
                sheet_name=names.get("rpdc", "RPDC"))
    onet  = pd.read_excel(file_path,
                sheet_name=names.get("onet_activities", "ONET_Activities"))
    esco  = pd.read_excel(file_path,
                sheet_name=names.get("esco_skills", "ESCO_Skills"))

    # RPDC
    rpdc_texts = rpdc[COL_RPDC].apply(clean_text)
    rpdc_texts = rpdc_texts[rpdc_texts.apply(is_valid)]

    # O*NET
    if COL_ONET_DESC in onet.columns:
        onet_texts = onet[COL_ONET_DESC].apply(clean_text)
    else:
        onet_texts = onet[COL_ONET_FALL].apply(clean_text)
    onet_texts = onet_texts[onet_texts.apply(is_valid)]

    # ESCO
    esco_texts = esco[COL_ESCO].apply(clean_text)
    esco_texts = esco_texts[esco_texts.apply(is_valid)]

    return rpdc_texts, onet_texts, esco_texts


def compute_stats(name, texts):
    wc = texts.apply(word_count)
    return {
        "System":          name,
        "n_descriptors":   len(wc),
        "median_words":    round(float(np.median(wc)), 1),
        "mean_words":      round(float(np.mean(wc)), 1),
        "min_words":       int(wc.min()),
        "max_words":       int(wc.max()),
        "p10_words":       round(float(np.percentile(wc, 10)), 1),
        "p90_words":       round(float(np.percentile(wc, 90)), 1),
    }

def print_stats(stats):
    print(f"\n{stats['System']}")
    print(f"  n descriptors : {stats['n_descriptors']}")
    print(f"  median words  : {stats['median_words']}")
    print(f"  mean words    : {stats['mean_words']}")
    print(f"  min / max     : {stats['min_words']} / {stats['max_words']}")
    print(f"  p10 / p90     : {stats['p10_words']} / {stats['p90_words']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python descriptor_length.py <your_excel_file.xlsx>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    rpdc, onet, esco = load_descriptors(file_path)

    print("=" * 45)
    print("DESCRIPTOR LENGTH STATISTICS (word count)")
    print("=" * 45)

    rows = [
        compute_stats("RPDC (vocational standards)", rpdc),
        compute_stats("O*NET (Work Activities)",     onet),
        compute_stats("ESCO (skill labels)",         esco),
    ]

    for r in rows:
        print_stats(r)

    # ── Save to xlsx ───────────────────────────────────────────
    out_path = file_path.stem + "_descriptor_length_stats.xlsx"
    df = pd.DataFrame(rows)
    df.to_excel(out_path, index=False)
    print(f"\nResults saved to: {out_path}")