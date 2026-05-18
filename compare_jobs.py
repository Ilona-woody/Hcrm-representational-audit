import re
import sys
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# USAGE
# --------------------------------------------------
# Option 1: run on all matching files in a folder
#   py compare_jobs.py "C:\path\to\folder"
#
# Option 2: run on specific files
#   py compare_jobs.py "Electrician_Representation_Comparison_summary_allstats.xlsx" ^
#                      "Dog_Groomer_Representation_Comparison_summary_allstats.xlsx" ^
#                      "Massage_Therapist_Representation_Comparison_summary_allstats.xlsx"
#
# OUTPUTS:
#   combined_jobs_systems_long.xlsx
#   combined_jobs_systems_wide.xlsx
#   combined_jobs_systems_report.xlsx
#   paper_mean_table.xlsx
#   paper_mean_table.csv
# --------------------------------------------------

EXPECTED_COLUMNS = [
    "Pillar",
    "System",
    "n_descriptors",
    "mean_similarity",
    "median_similarity",
    "std_similarity",
    "min_similarity",
    "max_similarity",
    "p10_similarity",
    "p90_similarity",
]

FILE_SUFFIX = "_summary_allstats.xlsx"

SYSTEM_RENAME = {
    "RPDC": "RPDC",
    "ONET_Activities": "O*NET",
    "ESCO_Skills": "ESCO",
}

PREFERRED_SYSTEM_ORDER = ["RPDC", "O*NET", "ESCO"]


def infer_occupation_from_filename(path: Path) -> str:
    """
    Extract occupation name from filenames like:
    Electrician_Representation_Comparison_summary_allstats.xlsx
    Dog_Groomer_Representation_Comparison_summary_allstats.xlsx
    """
    stem = path.stem
    stem = re.sub(r"_Representation_Comparison_summary_allstats$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_summary_allstats$", "", stem, flags=re.IGNORECASE)
    return stem.replace("_", " ").strip()


def load_one_summary(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"File '{path.name}' is missing expected columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    df = df[EXPECTED_COLUMNS].copy()
    df["Occupation"] = infer_occupation_from_filename(path)

    df["Occupation"] = df["Occupation"].astype(str)
    df["Pillar"] = df["Pillar"].astype(str)
    df["System"] = df["System"].astype(str).replace(SYSTEM_RENAME)

    return df


def collect_input_files(args: list[str]) -> list[Path]:
    if not args:
        raise ValueError("Please provide either a folder path or one or more summary_allstats.xlsx files.")

    if len(args) == 1:
        p = Path(args[0]).expanduser()
        if p.is_dir():
            files = sorted(p.glob(f"*{FILE_SUFFIX}"))
            if not files:
                raise FileNotFoundError(f"No files matching '*{FILE_SUFFIX}' found in folder: {p}")
            return files
        if p.is_file():
            return [p]

    files = [Path(a).expanduser() for a in args]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError("These input files were not found:\n" + "\n".join(missing))

    return files


def pillar_sort_key(p: str):
    m = re.match(r"[Pp](\d+)", str(p).strip())
    return int(m.group(1)) if m else 999


def build_wide_table(combined: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "n_descriptors",
        "mean_similarity",
        "median_similarity",
        "std_similarity",
        "min_similarity",
        "max_similarity",
        "p10_similarity",
        "p90_similarity",
    ]

    wide_parts = []
    for metric in metrics:
        pivot = combined.pivot(
            index=["Occupation", "Pillar"],
            columns="System",
            values=metric
        )
        pivot.columns = [f"{metric}__{c}" for c in pivot.columns]
        wide_parts.append(pivot)

    wide = pd.concat(wide_parts, axis=1).reset_index()

    ordered_cols = ["Occupation", "Pillar"]
    for metric in metrics:
        for system in PREFERRED_SYSTEM_ORDER:
            col = f"{metric}__{system}"
            if col in wide.columns:
                ordered_cols.append(col)

    remaining = [c for c in wide.columns if c not in ordered_cols]
    wide = wide[ordered_cols + remaining]

    return wide


def build_mean_only_table(combined: pd.DataFrame) -> pd.DataFrame:
    mean_table = combined.pivot(
        index=["Occupation", "Pillar"],
        columns="System",
        values="mean_similarity"
    ).reset_index()

    ordered_cols = ["Occupation", "Pillar"]
    for system in PREFERRED_SYSTEM_ORDER:
        if system in mean_table.columns:
            ordered_cols.append(system)

    remaining = [c for c in mean_table.columns if c not in ordered_cols]
    mean_table = mean_table[ordered_cols + remaining]

    return mean_table


def build_paper_table(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Publication-ready table:
    Occupation | Pillar | RPDC | O*NET | ESCO
    with means rounded to 3 decimals
    """
    paper = build_mean_only_table(combined).copy()

    for col in PREFERRED_SYSTEM_ORDER:
        if col in paper.columns:
            paper[col] = paper[col].round(3)

    return paper


def main():
    input_files = collect_input_files(sys.argv[1:])

    print("Reading files:")
    for f in input_files:
        print(f" - {f.name}")

    frames = [load_one_summary(f) for f in input_files]
    combined = pd.concat(frames, ignore_index=True)

    system_order = {name: i for i, name in enumerate(PREFERRED_SYSTEM_ORDER, start=1)}
    combined["_system_order"] = combined["System"].map(system_order).fillna(99)
    combined["_pillar_order"] = combined["Pillar"].map(pillar_sort_key)
    combined = combined.sort_values(["Occupation", "_pillar_order", "_system_order"]).drop(
        columns=["_system_order"]
    )

    wide = build_wide_table(combined)
    mean_only = build_mean_only_table(combined)
    paper_table = build_paper_table(combined)

    # sort output tables cleanly
    wide["_pillar_order"] = wide["Pillar"].map(pillar_sort_key)
    wide = wide.sort_values(["Occupation", "_pillar_order"]).drop(columns=["_pillar_order"])

    mean_only["_pillar_order"] = mean_only["Pillar"].map(pillar_sort_key)
    mean_only = mean_only.sort_values(["Occupation", "_pillar_order"]).drop(columns=["_pillar_order"])

    paper_table["_pillar_order"] = paper_table["Pillar"].map(pillar_sort_key)
    paper_table = paper_table.sort_values(["Occupation", "_pillar_order"]).drop(columns=["_pillar_order"])

    out_dir = input_files[0].parent
    out_long = out_dir / "combined_jobs_systems_long.xlsx"
    out_wide = out_dir / "combined_jobs_systems_wide.xlsx"
    out_report = out_dir / "combined_jobs_systems_report.xlsx"
    out_paper_xlsx = out_dir / "paper_mean_table.xlsx"
    out_paper_csv = out_dir / "paper_mean_table.csv"

    combined.to_excel(out_long, index=False)
    wide.to_excel(out_wide, index=False)
    paper_table.to_excel(out_paper_xlsx, index=False)
    paper_table.to_csv(out_paper_csv, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(out_report, engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="long_all_results", index=False)
        wide.to_excel(writer, sheet_name="wide_all_results", index=False)
        mean_only.to_excel(writer, sheet_name="mean_only", index=False)
        paper_table.to_excel(writer, sheet_name="paper_mean_table", index=False)

    print("\nSaved:")
    print(f" - {out_long.name}")
    print(f" - {out_wide.name}")
    print(f" - {out_report.name}")
    print(f" - {out_paper_xlsx.name}")
    print(f" - {out_paper_csv.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()