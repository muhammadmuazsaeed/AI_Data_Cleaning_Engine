import pandas as pd
from sqlalchemy import create_engine

# Any file/table bigger than this many rows is treated as "large"
# and read in chunks instead of all at once.
LARGE_DATA_ROW_THRESHOLD = 500_000
CHUNK_SIZE = 100_000

# Columns matching these name hints are forced to load as text, not numbers.
# Otherwise pandas' own CSV/Excel type inference silently converts values
# like phone number "03001234567" into the number 3001234567.0, destroying
# the leading zero before any of our own cleaning code even runs.
_FORCE_TEXT_NAME_HINTS = [
    "phone", "mobile", "fax", "zip", "postal", "pincode", "pin_code",
    "ssn", "cnic", "nic", "passport", "account_no", "acc_no",
]


def _build_forced_text_dtype_map(columns) -> dict:
    return {
        col: str for col in columns
        if any(hint in str(col).lower() for hint in _FORCE_TEXT_NAME_HINTS)
    }


def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV file. Automatically chunks very large files."""
    # Peek at the file size to decide small vs large strategy
    try:
        row_count = sum(1 for _ in open(filepath, "r", encoding="utf-8", errors="ignore")) - 1
    except Exception:
        row_count = 0

    # Peek at just the header so identifier-like columns (phone, zip, etc.)
    # can be forced to load as text instead of getting silently parsed as
    # numbers (which would destroy leading zeros).
    header_df = pd.read_csv(filepath, nrows=0)
    dtype_map = _build_forced_text_dtype_map(header_df.columns)

    if row_count > LARGE_DATA_ROW_THRESHOLD:
        print(f"[ingestion] Large CSV detected (~{row_count} rows). Reading in chunks...")
        chunks = pd.read_csv(filepath, chunksize=CHUNK_SIZE, low_memory=False, dtype=dtype_map)
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(filepath, low_memory=False, dtype=dtype_map)

    print(f"[ingestion] Loaded CSV: {filepath} -> {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def load_excel(filepath: str, sheet_name=0) -> pd.DataFrame:
    """Load an Excel file (.xlsx/.xls). sheet_name can be name, index, or None (all sheets)."""
    header_df = pd.read_excel(filepath, sheet_name=sheet_name if not isinstance(sheet_name, type(None)) else 0, nrows=0)
    dtype_map = _build_forced_text_dtype_map(header_df.columns)

    df = pd.read_excel(filepath, sheet_name=sheet_name, dtype=dtype_map)
    if isinstance(df, dict):
        # multiple sheets returned -> concatenate with a sheet source column
        frames = []
        for name, sheet_df in df.items():
            sheet_df = sheet_df.copy()
            sheet_df["__source_sheet__"] = name
            frames.append(sheet_df)
        df = pd.concat(frames, ignore_index=True)
    print(f"[ingestion] Loaded Excel: {filepath} -> {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def load_database(connection_string: str, table_name: str = None, query: str = None) -> pd.DataFrame:
    
    engine = create_engine(connection_string)

    if query:
        df = pd.read_sql_query(query, engine)
    elif table_name:
        df = pd.read_sql_table(table_name, engine)
    else:
        raise ValueError("Provide either table_name or query to load from database.")

    print(f"[ingestion] Loaded DB data -> {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def load_data(source_type: str, **kwargs) -> pd.DataFrame:
   
    source_type = source_type.lower()
    if source_type == "csv":
        return load_csv(kwargs["filepath"])
    elif source_type == "excel":
        return load_excel(kwargs["filepath"], sheet_name=kwargs.get("sheet_name", 0))
    elif source_type == "database":
        return load_database(
            kwargs["connection_string"],
            table_name=kwargs.get("table_name"),
            query=kwargs.get("query"),
        )
    else:
        raise ValueError(f"Unsupported source_type: {source_type}")
