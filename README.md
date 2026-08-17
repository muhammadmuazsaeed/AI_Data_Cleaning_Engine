# AI Data Cleaning Engine

A hybrid data cleaning system with a **web interface** that combines **Rule-Based logic**, **Machine Learning**, and an optional **Gemini LLM layer** to automatically detect and fix common data quality problems — upload any messy CSV/Excel file and download it clean, in the same format.

## Why "Hybrid AI"?

| Layer | What it does | Technique |
|---|---|---|
| **Rules** | Fast, deterministic fixes | Trim whitespace, remove exact duplicates, fix data types, standardize dates, cap outliers (IQR) |
| **Type Detection** | Makes messy columns usable by the other layers | Converts placeholder text ("Unknown"/"-"/"N/A") to real missing values, parses messy currency/number text ("$40,943.32", "Rs. 31,103") into real numbers, standardizes mixed date formats, converts yes/no/true/false text into real booleans, fixes/validates malformed emails |
| **ML** | Learns patterns from the data itself | KNN Imputation (missing values), Isolation Forest (anomaly detection), Fuzzy Matching (merging similar text categories, with safeguards against merging opposites like "Active"/"Inactive") |
| **LLM (Gemini)** | Understands meaning/context, not just patterns | Google Gemini API standardizes messy free text using semantic understanding (e.g. abbreviations like "NYC" -> "New York") |

### Why a separate "Type Detection" layer?
Outlier detection and imputation only work on columns pandas recognizes as
numeric. If a price column is stored as text ("$40,943.32"), it's invisible
to those steps no matter how good the ML is. `src/type_cleaner.py` runs
first and converts messy numeric/date/boolean/email columns into their real
data types, so every later step can actually see and clean them.

Gemini is used because it has a **free tier**, so the LLM layer doesn't require a paid API key to try out.

## Supported Data Sources
- **CSV** files (small or large — auto-chunks files with 500,000+ rows)
- **Excel** files (.xlsx/.xls)
- **Databases** (SQLite, PostgreSQL, MySQL — via the CLI, `main.py`)

## Project Structure
```
project_v2/
├── app.py                   # ⭐ Streamlit WEB APP — upload, clean, download
├── main.py                  # CLI version (for database sources / scripting)
├── data/                    # sample dataset
├── output/                  # cleaned data + report get saved here (CLI mode)
├── src/
│   ├── ingestion.py          # loads CSV / Excel / Database
│   ├── profiler.py           # analyzes data quality BEFORE cleaning
│   ├── type_cleaner.py       # detects & fixes real data types (numbers/dates/booleans/emails)
│   ├── rule_cleaner.py       # Layer 1: rule-based cleaning
│   ├── ml_cleaner.py         # Layer 2: ML-based cleaning
│   ├── llm_cleaner.py        # Layer 3: Gemini LLM-based cleaning (optional)
│   └── report.py             # generates the before/after report
├── requirements.txt
└── README.md
```

## Installation
```bash
pip install -r requirements.txt
```

## Option A: Run the Web App (recommended)
```bash
streamlit run app.py
```
This opens a browser window where you can:
1. Upload a CSV or Excel file
2. See a data quality summary (missing values, duplicates, outliers)
3. Click **"Clean My Data"** — Rules + ML run automatically, and category
   columns (like city/country/status) are **auto-detected** for fuzzy
   spelling-matching — no manual column names needed
4. Optionally tick **"Enable Gemini AI layer"**, paste a free Gemini API key
   (get one at https://aistudio.google.com/apikey) — Gemini automatically
   runs on the same auto-detected category columns, no manual column list needed
5. Download the cleaned file — **in the same format you uploaded**
   (CSV → CSV, Excel → Excel) — plus a downloadable cleaning report

## Option B: Run via Command Line (CLI)
Useful for database sources or automation/scripting.

```bash
# CSV
python main.py --source csv --filepath data/sample_messy_data.csv --output output/cleaned_data.csv --fuzzy_columns city

# Excel
python main.py --source excel --filepath data/mydata.xlsx --output output/cleaned_data.csv

# Database
python main.py --source database --conn "sqlite:///data/mydata.db" --table customers --output output/cleaned_data.csv

# With Gemini LLM layer enabled
export GEMINI_API_KEY="your-free-key-here"
python main.py --source csv --filepath data/sample_messy_data.csv --use_llm --llm_columns city notes --output output/cleaned_data.csv
```
If no `GEMINI_API_KEY` is set, the LLM layer is automatically skipped and the pipeline still runs fine with just Rules + ML.

## What You Get as Output
1. **Cleaned file** — same format as the upload (CSV → CSV, XLSX → XLSX)
2. **Cleaning report** (Markdown) showing:
   - Rows before/after
   - Missing values before/after
   - Duplicates found and removed
   - Outliers detected (both statistical and ML-based)

## How the Pipeline Works (Flow)
```
Upload File (CSV/Excel)
        ↓
Profile Data (find issues)
        ↓
Rule-Based Cleaning  → trim, dedupe, fix types, cap outliers
        ↓
ML-Based Cleaning    → KNN imputation, Isolation Forest, fuzzy matching
        ↓
Gemini Cleaning (optional) → semantic text standardization
        ↓
Download Cleaned File (same format) + Report
```

## Getting a Free Gemini API Key
1. Go to https://aistudio.google.com/apikey
2. Sign in with a Google account
3. Click "Create API Key" — it's free within Gemini's free tier usage limits
4. Paste it into the app's sidebar (or set it as `GEMINI_API_KEY` env variable for CLI use)

## Notes for Presentation
- Explain the **3-layer architecture** — this is the main "AI" selling point: not everything needs an LLM call, so the design stays fast and cost-efficient, and the LLM layer is fully optional.
- The **web app (Streamlit)** is the primary way users interact with the engine — no coding needed, just upload and download.
- Be ready to briefly explain:
  - *KNN Imputer*: fills a missing value by looking at the K most similar rows and averaging their values, instead of just using the column mean.
  - *Isolation Forest*: an unsupervised ML model that isolates anomalies faster than normal points because they're "easier to separate" in the data — used here to flag rows for review rather than silently deleting them.
  - *Gemini layer*: only used for semantic/text understanding tasks that rules and ML can't handle well (e.g. recognizing "the big apple" means "New York").
