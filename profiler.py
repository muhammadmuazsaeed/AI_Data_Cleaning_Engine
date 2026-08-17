import pandas as pd
import numpy as np


def profile_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    return pd.DataFrame({
        "column": df.columns,
        "missing_count": missing.values,
        "missing_pct": missing_pct.values,
        "dtype": df.dtypes.astype(str).values,
    })


def profile_duplicates(df: pd.DataFrame) -> dict:
    exact_dupes = df.duplicated().sum()
    return {
        "exact_duplicate_rows": int(exact_dupes),
        "exact_duplicate_pct": round(exact_dupes / len(df) * 100, 2) if len(df) else 0,
    }


def profile_outliers(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    """Uses simple Z-score method on numeric columns to flag potential outliers."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    results = []
    for col in numeric_cols:
        col_data = df[col].dropna()
        if col_data.std(ddof=0) == 0 or len(col_data) == 0:
            outlier_count = 0
        else:
            z_scores = np.abs((col_data - col_data.mean()) / col_data.std(ddof=0))
            outlier_count = int((z_scores > z_thresh).sum())
        results.append({
            "column": col,
            "outlier_count": outlier_count,
            "outlier_pct": round(outlier_count / len(df) * 100, 2) if len(df) else 0,
        })
    return pd.DataFrame(results)


def profile_dataset(df: pd.DataFrame) -> dict:
    """Runs a full profile and returns a summary dict used for reporting."""
    report = {
        "shape": df.shape,
        "missing_values": profile_missing_values(df),
        "duplicates": profile_duplicates(df),
        "outliers": profile_outliers(df),
    }
    return report


def print_profile_summary(report: dict):
    print("\n===== DATA PROFILE SUMMARY =====")
    print(f"Rows: {report['shape'][0]}, Columns: {report['shape'][1]}")
    print("\n-- Missing Values --")
    print(report["missing_values"].to_string(index=False))
    print("\n-- Duplicates --")
    print(report["duplicates"])
    print("\n-- Outliers (numeric columns, z-score>3) --")
    print(report["outliers"].to_string(index=False))
    print("=================================\n")
