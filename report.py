import pandas as pd
from datetime import datetime


def generate_report(before_profile: dict, after_df: pd.DataFrame, output_path: str = "output/cleaning_report.md"):
    after_missing = after_df.isnull().sum().sum()
    before_missing = before_profile["missing_values"]["missing_count"].sum()
    before_rows = before_profile["shape"][0]
    after_rows = after_df.shape[0]

    lines = []
    lines.append(f"# AI Data Cleaning Engine - Report")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")

    lines.append("## Summary")
    lines.append(f"- Rows before cleaning: **{before_rows}**")
    lines.append(f"- Rows after cleaning: **{after_rows}**")
    lines.append(f"- Rows removed (duplicates/etc.): **{before_rows - after_rows}**")
    lines.append(f"- Missing values before: **{int(before_missing)}**")
    lines.append(f"- Missing values after: **{int(after_missing)}**\n")

    lines.append("## Missing Values (Before Cleaning)")
    lines.append(before_profile["missing_values"].to_markdown(index=False))
    lines.append("")

    lines.append("## Duplicates (Before Cleaning)")
    dupes = before_profile["duplicates"]
    lines.append(f"- Exact duplicate rows found: {dupes['exact_duplicate_rows']} "
                 f"({dupes['exact_duplicate_pct']}%)\n")

    lines.append("## Outliers (Before Cleaning, Z-score method)")
    lines.append(before_profile["outliers"].to_markdown(index=False))
    lines.append("")

    if "__is_outlier__" in after_df.columns:
        ml_outliers = int(after_df["__is_outlier__"].sum())
        lines.append("## ML Outlier Detection (Isolation Forest)")
        lines.append(f"- Rows flagged as anomalies: {ml_outliers}\n")

    after_missing_by_col = after_df.isnull().sum()
    remaining = after_missing_by_col[after_missing_by_col > 0]
    lines.append("## Missing Values Left As-Is (After Cleaning)")
    if len(remaining) > 0:
        lines.append("These columns still contain missing values **on purpose** -- "
                     "there's no reliable way to guess a missing email, phone number, "
                     "or similar identifying detail, so the engine leaves them missing "
                     "rather than fabricating a plausible-looking fake value:\n")
        for col, count in remaining.items():
            pct = round(count / len(after_df) * 100, 2) if len(after_df) else 0
            lines.append(f"- **{col}**: {int(count)} missing ({pct}%)")
    else:
        lines.append("None -- every column was fully filled in.")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[report] Cleaning report saved to: {output_path}")
    return output_path