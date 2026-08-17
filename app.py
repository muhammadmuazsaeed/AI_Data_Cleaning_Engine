import os
import io
import sys
import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(dotenv_path=_env_path, override=True)  # loads GEMINI_API_KEY from .env file if present
except ImportError:
    pass

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from profiler import profile_dataset
from ingestion import _build_forced_text_dtype_map
from rule_cleaner import run_rule_based_cleaning
from ml_cleaner import run_ml_cleaning, auto_detect_categorical_columns
from llm_cleaner import run_llm_cleaning
from report import generate_report

st.set_page_config(page_title="AI Data Cleaning Engine", page_icon="🧹", layout="wide")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🧹 AI Data Cleaning Engine")
st.caption("Upload any messy CSV or Excel file — get it cleaned automatically using Rules + ML + optional Gemini AI.")

# ---------------------------------------------------------------------------
# Sidebar - options
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Cleaning Options")

    auto_fuzzy = st.checkbox(
        "Auto-detect category columns for fuzzy matching",
        value=True,
        help="Automatically finds columns like city/country/status and merges similar "
             "spellings (e.g. 'NYC' and 'New York' become one value) — no manual input needed."
    )
    fuzzy_input = ""
    if not auto_fuzzy:
        fuzzy_input = st.text_input(
            "Or manually specify columns (comma-separated)",
            placeholder="e.g. city, country"
        )

    st.markdown("---")
    convert_currency = st.checkbox(
        "Convert foreign currency ($/€/£) to PKR",
        value=False,
        help="OFF by default. A '$' or '€' in your data doesn't always mean a real "
             "foreign-currency amount — sometimes it's just a stray formatting symbol "
             "and the number was always in PKR. Only turn this on if you've checked "
             "your data and know it genuinely mixes currencies. When on, uses approximate "
             "rates: 1 USD=280 PKR, 1 EUR=305 PKR, 1 GBP=355 PKR."
    )

    st.markdown("---")
    use_llm = st.checkbox("Enable Gemini AI layer (optional)", value=False)
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if use_llm:
        if gemini_key and gemini_key != "your-gemini-api-key-here":
            st.success("✅ Gemini API key loaded from .env file")
        else:
            gemini_key = st.text_input("Gemini API Key", type="password",
                                        help="Get a free key at https://aistudio.google.com/apikey")
            if gemini_key:
                os.environ["GEMINI_API_KEY"] = gemini_key
        st.caption("Gemini automatically runs on the same auto-detected category "
                   "columns above — no need to list columns manually.")

    st.markdown("---")
    st.caption("Rules + ML always run automatically. Gemini (if enabled) runs on the same auto-detected columns.")

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1].lower()

    # ---- Load data, remembering original format ----
    try:
        if file_ext == "csv":
            header_df = pd.read_csv(uploaded_file, nrows=0)
            dtype_map = _build_forced_text_dtype_map(header_df.columns)
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, low_memory=False, dtype=dtype_map)
            output_format = "csv"
        else:
            header_df = pd.read_excel(uploaded_file, nrows=0)
            dtype_map = _build_forced_text_dtype_map(header_df.columns)
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file, dtype=dtype_map)
            output_format = "xlsx"
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()

    st.success(f"Loaded **{uploaded_file.name}** — {df.shape[0]} rows, {df.shape[1]} columns")

    with st.expander("👀 Preview raw data (first 10 rows)"):
        st.dataframe(df.head(10), use_container_width=True)

    # ---- Profile BEFORE cleaning ----
    before_profile = profile_dataset(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Missing values", int(before_profile["missing_values"]["missing_count"].sum()))
    col2.metric("Duplicate rows", int(before_profile["duplicates"]["exact_duplicate_rows"]))
    col3.metric("Outliers detected", int(before_profile["outliers"]["outlier_count"].sum()))

    with st.expander("📊 Full data quality profile"):
        st.write("**Missing values by column**")
        st.dataframe(before_profile["missing_values"], use_container_width=True)
        st.write("**Duplicates**")
        st.json(before_profile["duplicates"])
        st.write("**Outliers by column (Z-score method)**")
        st.dataframe(before_profile["outliers"], use_container_width=True)

    # ---------------------------------------------------------------------
    # Run cleaning
    # ---------------------------------------------------------------------
    if st.button("🚀 Clean My Data", type="primary"):
        with st.spinner("Running Rule-based cleaning..."):
            df_clean = run_rule_based_cleaning(df.copy(), convert_currency=convert_currency)

        if auto_fuzzy:
            fuzzy_columns = auto_detect_categorical_columns(df_clean)
            if fuzzy_columns:
                st.caption(f"🔎 Auto-detected category columns: {', '.join(fuzzy_columns)}")
            else:
                st.caption("🔎 No category-like columns detected for fuzzy matching.")
        else:
            fuzzy_columns = [c.strip() for c in fuzzy_input.split(",") if c.strip()]

        with st.spinner("Running ML-based cleaning (imputation, outlier detection, fuzzy matching)..."):
            df_clean = run_ml_cleaning(df_clean, fuzzy_columns=fuzzy_columns, auto_detect_fuzzy=False)

        valid_gemini_key = gemini_key and gemini_key != "your-gemini-api-key-here"
        if use_llm and valid_gemini_key:
            if fuzzy_columns:
                with st.spinner(f"Running Gemini AI cleaning on: {', '.join(fuzzy_columns)}..."):
                    df_clean, llm_messages = run_llm_cleaning(df_clean, text_columns=fuzzy_columns)
                for msg in llm_messages:
                    if msg.lower().startswith("failed") or "error" in msg.lower():
                        st.error(f"🔴 Gemini: {msg}")
                    elif "partially" in msg.lower():
                        st.warning(f"🟡 Gemini: {msg}")
                    else:
                        st.success(f"🟢 Gemini: {msg}")
            else:
                st.info("No category-like text columns detected — skipping Gemini layer.")
        elif use_llm and not valid_gemini_key:
            st.warning("Gemini enabled but no valid API key found — skipping LLM layer.")

        st.session_state["df_clean"] = df_clean
        st.session_state["before_profile"] = before_profile
        st.session_state["output_format"] = output_format
        st.session_state["original_name"] = uploaded_file.name.rsplit(".", 1)[0]
        st.success("✅ Cleaning complete!")

    # ---------------------------------------------------------------------
    # Results + downloads
    # ---------------------------------------------------------------------
    if "df_clean" in st.session_state:
        df_clean = st.session_state["df_clean"]
        before_profile = st.session_state["before_profile"]
        output_format = st.session_state["output_format"]
        original_name = st.session_state["original_name"]

        st.subheader("✨ Cleaned Data Preview")
        st.dataframe(df_clean.head(20), use_container_width=True)

        after_missing = int(df_clean.isnull().sum().sum())
        before_missing = int(before_profile["missing_values"]["missing_count"].sum())
        before_rows = before_profile["shape"][0]
        after_rows = df_clean.shape[0]

        m1, m2, m3 = st.columns(3)
        m1.metric("Rows", after_rows, delta=int(after_rows - before_rows))
        m2.metric("Missing values", after_missing, delta=int(after_missing - before_missing), delta_color="inverse")
        if "__is_outlier__" in df_clean.columns:
            m3.metric("ML-flagged outliers", int(df_clean["__is_outlier__"].sum()))

        # ---- Prepare downloadable file in ORIGINAL format ----
        if output_format == "csv":
            buffer = io.StringIO()
            df_clean.to_csv(buffer, index=False)
            file_bytes = buffer.getvalue().encode("utf-8")
            mime = "text/csv"
            out_name = f"{original_name}_cleaned.csv"
        else:
            buffer = io.BytesIO()
            df_clean.to_excel(buffer, index=False, engine="openpyxl")
            file_bytes = buffer.getvalue()
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            out_name = f"{original_name}_cleaned.xlsx"

        # ---- Generate report ----
        os.makedirs("output", exist_ok=True)
        report_path = generate_report(before_profile, df_clean, output_path="output/cleaning_report.md")
        with open(report_path, "r") as f:
            report_text = f.read()

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                label=f"⬇️ Download Cleaned File (.{output_format})",
                data=file_bytes,
                file_name=out_name,
                mime=mime,
                type="primary",
            )
        with dl2:
            st.download_button(
                label="📄 Download Cleaning Report (.md)",
                data=report_text,
                file_name=f"{original_name}_cleaning_report.md",
                mime="text/markdown",
            )
else:
    st.info("👆 Upload a CSV or Excel file to get started.")
    st.caption("Not sure what to try? Use the sample messy dataset in `data/sample_messy_data.csv`.")