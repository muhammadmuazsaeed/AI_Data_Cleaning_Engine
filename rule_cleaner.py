import pandas as pd
import numpy as np

from type_cleaner import run_type_cleaning, drop_id_like_columns


def trim_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing spaces AND collapse internal double/multiple
    spaces (e.g. "Iqra  Chaudhry" -> "Iqra Chaudhry") in all text columns."""
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = (
            df[col].astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .replace("nan", np.nan)
        )
    return df


def standardize_casing(df: pd.DataFrame, columns=None, mode="title") -> pd.DataFrame:
   
    cols = columns if columns else df.select_dtypes(include=["object"]).columns
    for col in cols:
        if col not in df.columns:
            continue
        if mode == "title":
            df[col] = df[col].astype(str).str.title()
        elif mode == "lower":
            df[col] = df[col].astype(str).str.lower()
        elif mode == "upper":
            df[col] = df[col].astype(str).str.upper()
    return df


def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[rule_cleaner] Removed {before - len(df)} exact duplicate rows.")
    return df


def standardize_dates(df: pd.DataFrame, date_columns: list) -> pd.DataFrame:
    """Convert given columns to a consistent YYYY-MM-DD date format."""
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def fix_data_types(df: pd.DataFrame, type_map: dict) -> pd.DataFrame:
    """
    Force specific columns into specific types.
    type_map example: {"age": "int", "salary": "float", "id": "str"}
    """
    for col, dtype in type_map.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (ValueError, TypeError):
                print(f"[rule_cleaner] Could not convert column '{col}' to {dtype}, skipping.")
    return df


def cap_outliers_iqr(df: pd.DataFrame, columns=None, protect_columns=None) -> pd.DataFrame:
    
    numeric_cols = columns if columns else list(df.select_dtypes(include=[np.number]).columns)
    numeric_cols = [c for c in numeric_cols if c != "__is_outlier__"]
    numeric_cols = drop_id_like_columns(df, numeric_cols)
    protect_columns = set(protect_columns or [])
    if protect_columns:
        skipped = [c for c in numeric_cols if c in protect_columns]
        if skipped:
            print(f"[rule_cleaner] Skipping IQR capping for {skipped} (currency-converted "
                  f"columns -- capping would destroy legitimately large converted values).")
        numeric_cols = [c for c in numeric_cols if c not in protect_columns]
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        capped = df[col].clip(lower=lower, upper=upper)
        n_capped = int(((capped != df[col]) & df[col].notna()).sum())
        if n_capped:
            print(f"[rule_cleaner] Capped {n_capped} outlier value(s) in '{col}'.")
        df[col] = capped
    return df


def run_rule_based_cleaning(df: pd.DataFrame, date_columns=None, type_map=None, convert_currency: bool = False) -> pd.DataFrame:
    
    df = trim_whitespace(df)
    df, multi_currency_cols = run_type_cleaning(df, convert_currency=convert_currency)
    df = remove_exact_duplicates(df)
    if date_columns:
        df = standardize_dates(df, date_columns)
    if type_map:
        df = fix_data_types(df, type_map)
    df = cap_outliers_iqr(df, protect_columns=multi_currency_cols)
    return df