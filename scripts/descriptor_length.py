import sys
import re
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


def report(name, texts):
    wc = texts.apply(word_count)
    print(f"\n{name}")
    print(f"  n descriptors : {len(wc)}")
    print(f"  median words  : {np.median(wc):.1f}")
    print(f"  mean words    : {np.mean(wc):.1f}")
    print(f"  min / max     : {wc.min()} / {wc.max()}")
    print(f"  p10 / p90     : "
          f"{np.percentile(wc,10):.1f} / {np.percentile(wc,90):.1f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python descriptor_lengths.py <your_excel_file.xlsx>")
        sys.exit(1)

    rpdc, onet, esco = load_descriptors(Path(sys.argv[1]))

    print("=" * 45)
    print("DESCRIPTOR LENGTH STATISTICS (word count)")
    print("=" * 45)
    report("RPDC  (vocational standards)", rpdc)
    report("O*NET (Work Activities)",       onet)
    report("ESCO  (skill labels)",          esco)