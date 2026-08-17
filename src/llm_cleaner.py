import os
import json
import pandas as pd

try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(dotenv_path=_env_path, override=True)  # reads .env in the project root, always fresh
except ImportError:
    pass  # dotenv not installed -- GEMINI_API_KEY can still be set manually as an env var

try:
    from google import genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

_MODEL_NAME = "gemini-2.5-flash-lite"  # current free-tier model (gemini-2.0-flash was shut down June 2026)
_PLACEHOLDER_KEY = "your-gemini-api-key-here"


def _get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == _PLACEHOLDER_KEY or not _GEMINI_AVAILABLE:
        return None
    return genai.Client(api_key=api_key)


def _clean_json_text(text: str) -> str:
    return text.strip().replace("```json", "").replace("```", "").strip()


def llm_standardize_text_column(df: pd.DataFrame, column: str, batch_size: int = 50) -> tuple:
    
    client = _get_client()
    if client is None:
        msg = ("Skipped: no valid GEMINI_API_KEY found. Check your .env file "
               "or paste a valid key in the sidebar.")
        print(f"[llm_cleaner] {msg}")
        return df, msg

    if column not in df.columns:
        return df, f"Column '{column}' not found in data."

    unique_vals = df[column].dropna().astype(str).unique().tolist()
    mapping = {}
    errors = []

    for i in range(0, len(unique_vals), batch_size):
        batch = unique_vals[i:i + batch_size]
        prompt = (
            "You are a data cleaning assistant. Below is a JSON list of raw values "
            f"from a column called '{column}'. Some values are different spellings, "
            "abbreviations, or formats of the same real-world entity.\n\n"
            f"Values: {json.dumps(batch)}\n\n"
            "Return ONLY a JSON object mapping each original value to one standardized, "
            "clean version. No explanation, no markdown fences, just raw JSON."
        )
        try:
            response = client.models.generate_content(model=_MODEL_NAME, contents=prompt)
            text = _clean_json_text(response.text)
            batch_mapping = json.loads(text)
            mapping.update(batch_mapping)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            errors.append(error_msg)
            print(f"[llm_cleaner] Batch failed for '{column}', keeping original values: {error_msg}")

    if mapping:
        df[column] = df[column].astype(str).map(mapping).fillna(df[column])
        print(f"[llm_cleaner] Gemini-standardized '{column}' ({len(mapping)} values mapped).")

    if errors and not mapping:
        return df, f"Failed for '{column}': {errors[0]}"
    elif errors and mapping:
        return df, f"'{column}' partially cleaned ({len(mapping)} values mapped), {len(errors)} batch(es) failed: {errors[0]}"
    elif mapping:
        return df, f"'{column}' cleaned successfully ({len(mapping)} values mapped)."
    else:
        return df, f"No mapping returned for '{column}' (empty result from Gemini)."


def llm_suggest_missing_values(df: pd.DataFrame, column: str, context_columns: list, max_rows: int = 30) -> pd.DataFrame:
  
    client = _get_client()
    if client is None:
        print("[llm_cleaner] Skipped (no valid GEMINI_API_KEY set).")
        return df

    missing_idx = df[df[column].isnull()].index.tolist()[:max_rows]
    if not missing_idx:
        return df

    for idx in missing_idx:
        row_context = df.loc[idx, context_columns].to_dict()
        prompt = (
            f"Given this row of data: {json.dumps(row_context, default=str)}\n"
            f"Suggest the most plausible value for the missing field '{column}'. "
            "Return ONLY the value itself, nothing else."
        )
        try:
            response = client.models.generate_content(model=_MODEL_NAME, contents=prompt)
            suggestion = response.text.strip()
            df.at[idx, column] = suggestion
        except Exception as e:
            print(f"[llm_cleaner] Could not suggest value for row {idx}: {e}")

    print(f"[llm_cleaner] Gemini-suggested values for {len(missing_idx)} missing '{column}' entries.")
    return df


def run_llm_cleaning(df: pd.DataFrame, text_columns=None) -> tuple:
    
    messages = []
    if text_columns:
        for col in text_columns:
            df, msg = llm_standardize_text_column(df, col)
            messages.append(msg)
    return df, messages
