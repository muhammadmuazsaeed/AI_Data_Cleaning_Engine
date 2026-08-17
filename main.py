import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from ingestion import load_data
from profiler import profile_dataset, print_profile_summary
from rule_cleaner import run_rule_based_cleaning
from ml_cleaner import run_ml_cleaning
from llm_cleaner import run_llm_cleaning
from report import generate_report


def main():
    parser = argparse.ArgumentParser(description="AI Data Cleaning Engine")
    parser.add_argument("--source", choices=["csv", "excel", "database"], required=True)
    parser.add_argument("--filepath", help="Path to CSV/Excel file")
    parser.add_argument("--sheet_name", default=0, help="Excel sheet name/index")
    parser.add_argument("--conn", help="Database connection string")
    parser.add_argument("--table", help="Database table name")
    parser.add_argument("--query", help="Custom SQL query instead of table")
    parser.add_argument("--output", default="output/cleaned_data.csv", help="Output cleaned file path")
    parser.add_argument("--fuzzy_columns", nargs="*", default=None,
                        help="Categorical columns to fuzzy-clean. If omitted, columns are auto-detected.")
    parser.add_argument("--llm_columns", nargs="*", default=[], help="Text columns to clean via Gemini LLM")
    parser.add_argument("--use_llm", action="store_true", help="Enable Gemini LLM cleaning layer")
    parser.add_argument("--convert_currency", action="store_true",
                        help="Convert $/€/£ amounts to PKR (OFF by default -- only use if your "
                             "data genuinely mixes currencies, not just stray formatting symbols)")
    args = parser.parse_args()

    # ---------- 1. INGESTION ----------
    if args.source == "csv":
        df = load_data("csv", filepath=args.filepath)
    elif args.source == "excel":
        df = load_data("excel", filepath=args.filepath, sheet_name=args.sheet_name)
    else:
        df = load_data("database", connection_string=args.conn, table_name=args.table, query=args.query)

    # ---------- 2. PROFILE (before cleaning) ----------
    before_profile = profile_dataset(df)
    print_profile_summary(before_profile)

    # ---------- 3. RULE-BASED CLEANING ----------
    df = run_rule_based_cleaning(df, convert_currency=args.convert_currency)

    # ---------- 4. ML-BASED CLEANING ----------
    df = run_ml_cleaning(df, fuzzy_columns=args.fuzzy_columns, auto_detect_fuzzy=(args.fuzzy_columns is None))

    # ---------- 5. LLM-BASED CLEANING (optional) ----------
    if args.use_llm:
        df, llm_messages = run_llm_cleaning(df, text_columns=args.llm_columns)
        for msg in llm_messages:
            print(f"[main] Gemini: {msg}")

    # ---------- 6. SAVE OUTPUT + REPORT ----------
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[main] Cleaned data saved to: {args.output}")

    generate_report(before_profile, df, output_path="output/cleaning_report.md")


if __name__ == "__main__":
    main()