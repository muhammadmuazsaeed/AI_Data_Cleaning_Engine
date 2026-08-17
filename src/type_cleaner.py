import re
import pandas as pd
import numpy as np

_MISSING_PLACEHOLDERS = {
    "", "unknown", "unk", "n/a", "na", "null", "none", "nan", "-", "--",
    "?", "not available", "not specified", "undefined", "no data",
    "tbd", "to be decided", "to be determined", "n\\a", "missing",
    "tbc", "to be confirmed",
}

# These words are ambiguous: "Pending" is a legitimate real value in a
# workflow/status column (order_status, application_status) but is really
# a null placeholder in a descriptive column (year, department, major) --
# a department or major can't literally "be" pending, only a process can.
# So these are only treated as missing OUTSIDE columns whose name suggests
# a genuine status/workflow field.
_CONTEXTUAL_MISSING_VALUES = {"pending", "tba", "to be announced"}

_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_TRUE_VALUES = {"yes", "y", "true", "1", "t", "1.0"}
_FALSE_VALUES = {"no", "n", "false", "0", "f", "0.0"}

# Columns matching these name hints can never legitimately be negative --
# this is a hard business rule, not a statistical judgment call, so it's
# enforced directly instead of relying on IQR/Isolation Forest to "notice".
_NON_NEGATIVE_NAME_HINTS = [
    "price", "cost", "amount", "revenue", "income",
    "quantity", "qty", "count", "total", "balance", "payment",
]
_MUST_BE_POSITIVE_NAME_HINTS = ["salary", "wage", "stipend", "fee", "tuition"]  # 0 is also invalid, not just negative
_PERCENT_NAME_HINTS = ["pct", "percent", "percentage"]  # must be within [0, 100]

_EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ---------------------------------------------------------------------------
# 1. Missing value placeholders -> real NaN
# ---------------------------------------------------------------------------
def standardize_missing_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts text placeholders that really mean "missing" (Unknown, -, N/A,
    null, empty string, etc.) into actual NaN, so they get treated as missing
    data everywhere else in the pipeline instead of becoming their own
    fake "category".
    """
    obj_cols = df.select_dtypes(include=["object"]).columns
    total_converted = 0
    for col in obj_cols:
        is_status_field = "status" in col.lower()
        placeholder_set = _MISSING_PLACEHOLDERS if is_status_field else (
            _MISSING_PLACEHOLDERS | _CONTEXTUAL_MISSING_VALUES
        )
        mask = df[col].astype(str).str.strip().str.lower().isin(placeholder_set)
        total_converted += int(mask.sum())
        df.loc[mask, col] = np.nan
    if total_converted:
        print(f"[type_cleaner] Converted {total_converted} placeholder values "
              f"(e.g. 'Unknown', '-', 'N/A', 'Pending') to real missing values.")
    return df


# ---------------------------------------------------------------------------
# 2. Messy numeric / currency text -> real numeric dtype
# ---------------------------------------------------------------------------
_CURRENCY_TOKEN_RE = re.compile(r"(rs\.?|pkr|usd|inr|eur|gbp|\$|€|£|,|\s)", flags=re.IGNORECASE)
_CURRENCY_TOKEN_RE_SIMPLE = re.compile(r"[,\s]")
_PURE_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")

# Recurring-payment suffixes that commonly appear glued to a salary/amount
# figure ("260,407/month", "50000 per year", "12000 monthly"). These must be
# stripped as TEXT before numeric parsing -- otherwise the parser correctly
# refuses to guess a number out of "260,407/month" (good, that's not a pure
# number), the value becomes NaN, and it then gets silently replaced by a
# KNN-imputed guess with NO relationship to the real original salary. That's
# far worse than a parsing failure: it looks like a plausible number while
# actually being fabricated.
_PERIOD_SUFFIX_RE = re.compile(
    r"(/\s*(month|mo|year|yr|annum|week|wk|day)\b"
    r"|\bper\s+(month|year|week|day|annum)\b"
    r"|\b(monthly|yearly|annually|weekly|daily)\b)",
    flags=re.IGNORECASE,
)

# IMPORTANT: whether a "$"/"€"/"£" mark means "convert this to PKR at an
# exchange rate" or is just formatting noise (the value was ALWAYS in PKR,
# someone just typed a stray "$" sign) is NOT something that can be safely
# guessed from the symbol alone -- it varies dataset to dataset. Applying a
# blind exchange-rate multiplier by default is dangerous: it can silently
# inflate perfectly correct numbers by 280x when the symbol was never a
# real currency indicator. So currency conversion is OFF by default --
# symbols are stripped as pure formatting cleanup only. Turn it on
# explicitly (convert_currency=True) only when you know the data genuinely
# mixes currencies and needs converting to one consistent scale.
_CURRENCY_RATES_TO_PKR = {"usd": 280.0, "eur": 305.0, "gbp": 355.0}
_CURRENCY_MARKER_RE = re.compile(r"(?:\$|\busd\b|€|\beur\b|£|\bgbp\b)", flags=re.IGNORECASE)


def _detect_currency_rate(text: str):
    """Returns the PKR exchange rate for the currency marker found in text, or None if it's already PKR/unmarked."""
    if re.search(r"(\$|\busd\b)", text, flags=re.IGNORECASE):
        return _CURRENCY_RATES_TO_PKR["usd"]
    if re.search(r"(€|\beur\b)", text, flags=re.IGNORECASE):
        return _CURRENCY_RATES_TO_PKR["eur"]
    if re.search(r"(£|\bgbp\b)", text, flags=re.IGNORECASE):
        return _CURRENCY_RATES_TO_PKR["gbp"]
    return None


def _parse_numeric_value(value, convert_currency: bool = False):
    
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in _MISSING_PLACEHOLDERS:
        return np.nan
    if text in _WORD_NUMBERS:
        return float(_WORD_NUMBERS[text])

    text = _PERIOD_SUFFIX_RE.sub("", text)
    fx_rate = _detect_currency_rate(text) if convert_currency else None
    cleaned = _CURRENCY_TOKEN_RE.sub("", text)
    if cleaned in ("", "-", "."):
        return np.nan
    if not _PURE_NUMBER_RE.match(cleaned):
        return np.nan  # leftover letters/junk -> this was never really a number
    try:
        result = float(cleaned)
    except ValueError:
        return np.nan
    return result * fx_rate if fx_rate else result


_NON_QUANTITY_NAME_HINTS = [
    "phone", "mobile", "fax", "zip", "postal", "pincode", "pin_code",
    "ssn", "cnic", "nic", "passport", "account_no", "acc_no",
]

# ---------------------------------------------------------------------------
# 2b. GPA columns -- handled BEFORE the generic numeric parser, because a
#     GPA column often mixes two different scales in the same column:
#     plain 4.0-scale numbers ("3.5") and percentage entries ("90.2%") that
#     need converting to the SAME scale, not just having the "%" stripped.
# ---------------------------------------------------------------------------
_GPA_NAME_HINTS = ["gpa", "grade_point"]


def auto_detect_gpa_columns(df: pd.DataFrame) -> list:
    return [c for c in df.select_dtypes(include=["object"]).columns
            if any(hint in c.lower() for hint in _GPA_NAME_HINTS)]


def standardize_gpa_columns(df: pd.DataFrame, columns=None, scale_max: float = 4.0) -> pd.DataFrame:
    
    cols = columns if columns is not None else auto_detect_gpa_columns(df)
    for col in cols:
        if col not in df.columns:
            continue

        def _parse_gpa(v):
            if pd.isna(v):
                return np.nan
            text = str(v).strip().lower()
            if text in _MISSING_PLACEHOLDERS:
                return np.nan
            is_percent = text.endswith("%")
            text = text.rstrip("%").strip()
            text = _CURRENCY_TOKEN_RE_SIMPLE.sub("", text)  # strip stray commas/spaces
            try:
                num = float(text)
            except ValueError:
                return np.nan
            return (num / 100.0) * scale_max if is_percent else num

        df[col] = df[col].apply(_parse_gpa)
        invalid = (df[col] < 0) | (df[col] > scale_max)
        n_invalid = int(invalid.sum())
        if n_invalid:
            df.loc[invalid, col] = np.nan
        print(f"[type_cleaner] '{col}': converted percentage entries to a {scale_max}-point "
              f"scale; {n_invalid} out-of-range value(s) set to missing for re-imputation.")
    return df


# ---------------------------------------------------------------------------
# 2c. Currency conversion using a DEDICATED currency column, when one
#     exists (e.g. a separate "currency" column with values PKR/USD/EUR/GBP
#     next to an "amount" column). This is far more reliable than sniffing
#     for a "$" symbol inside the amount text, because some rows may store
#     the amount as a plain number with NO embedded symbol at all -- the
#     currency column is the only way to know their real currency. When
#     such a column pair exists, it takes priority over symbol-based
#     detection for that column.
# ---------------------------------------------------------------------------
_CURRENCY_CODE_VOCAB = {"pkr", "usd", "eur", "gbp", "inr"}
_AMOUNT_COL_HINTS = ["amount", "price", "fee", "salary", "total", "cost", "balance", "tuition"]


def auto_detect_currency_column_pairs(df: pd.DataFrame) -> list:
    """Finds (amount_column, currency_column) pairs -- a numeric-ish column paired with an explicit currency-code column."""
    currency_cols = []
    for col in df.select_dtypes(include=["object"]).columns:
        series = df[col].dropna().astype(str).str.strip().str.lower()
        if series.empty:
            continue
        if series.isin(_CURRENCY_CODE_VOCAB).mean() >= 0.9:
            currency_cols.append(col)

    pairs = []
    for ccol in currency_cols:
        for col in df.columns:
            if col == ccol or col not in df.select_dtypes(include=["object"]).columns:
                continue
            if any(hint in col.lower() for hint in _AMOUNT_COL_HINTS):
                pairs.append((col, ccol))
    return pairs


def convert_currency_via_column(df: pd.DataFrame, amount_col: str, currency_col: str, target: str = "pkr") -> pd.DataFrame:
    
    rates = {"usd": 280.0, "eur": 305.0, "gbp": 355.0, "pkr": 1.0, "inr": 3.3}

    raw = df[amount_col].apply(lambda v: _parse_numeric_value(v, convert_currency=False))
    currency = df[currency_col].astype(str).str.strip().str.lower()

    for cur in currency.dropna().unique():
        group_mask = currency == cur
        group_vals = raw[group_mask].dropna()
        if len(group_vals) < 10:
            continue
        q1, q3 = group_vals.quantile(0.25), group_vals.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_capped = int(((raw[group_mask] < lower) | (raw[group_mask] > upper)).sum())
        raw.loc[group_mask] = raw[group_mask].clip(lower=lower, upper=upper)
        if n_capped:
            print(f"[type_cleaner] Capped {n_capped} outlier value(s) within currency='{cur}' "
                  f"(before conversion, so a bad raw value can't get inflated by the exchange rate).")

    rate_map = currency.map(lambda c: rates.get(c, 1.0))
    df[amount_col] = raw * rate_map
    print(f"[type_cleaner] Converted '{amount_col}' to {target.upper()} using '{currency_col}' as the "
          f"authoritative currency for each row.")
    return df


def auto_detect_multi_currency_columns(df: pd.DataFrame) -> list:
   
    candidates = []
    for col in df.select_dtypes(include=["object"]).columns:
        series = df[col].dropna().astype(str)
        if series.empty:
            continue
        if series.str.contains(_CURRENCY_MARKER_RE).any():
            candidates.append(col)
    return candidates


def auto_detect_numeric_text_columns(df: pd.DataFrame, min_parse_ratio: float = 0.7) -> list:
    
    candidates = []
    for col in df.select_dtypes(include=["object"]).columns:
        if any(hint in col.lower() for hint in _NON_QUANTITY_NAME_HINTS):
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        parsed = series.apply(_parse_numeric_value)
        parse_ratio = parsed.notna().mean()
        if parse_ratio >= min_parse_ratio:
            candidates.append(col)
    return candidates


def clean_numeric_text_columns(df: pd.DataFrame, columns=None, convert_currency: bool = False) -> pd.DataFrame:
    """
    Converts messy numeric-text columns into real float columns.

    convert_currency: OFF by default -- see the note above _parse_numeric_value
    for why blind $/€/£ -> PKR conversion is unsafe as a default behavior.
    """
    cols = columns if columns is not None else auto_detect_numeric_text_columns(df)
    for col in cols:
        if col not in df.columns:
            continue
        df[col] = df[col].apply(lambda v: _parse_numeric_value(v, convert_currency=convert_currency))
        note = " (foreign-currency amounts converted to PKR)" if convert_currency else ""
        print(f"[type_cleaner] Converted '{col}' to numeric (was messy currency/text){note}.")
    return df


# ---------------------------------------------------------------------------
# 3. Mixed date formats -> standardized YYYY-MM-DD
# ---------------------------------------------------------------------------
_DATE_PATTERN_RE = re.compile(
    r"^\s*\d{1,4}\s*[-/]\s*[A-Za-z0-9]{1,9}\s*[-/]\s*\d{1,4}\s*$"
)


def auto_detect_date_columns(df: pd.DataFrame, min_parse_ratio: float = 0.7) -> list:
    """
    Finds columns that look like dates -- either by name (contains 'date',
    'dob', 'time') or by successfully parsing as dates. Only considers
    text (object) columns, and requires values to actually look like a
    date (contain separators like "-" or "/") before attempting to parse
    them -- otherwise a purely numeric column (e.g. "price": 4875.89)
    can get misread as a date by pandas' lenient date parser.
    """
    candidates = []
    for col in df.select_dtypes(include=["object"]).columns:
        name_hint = any(k in col.lower() for k in ["date", "dob", "_at", "time"])
        series = df[col].dropna().astype(str)
        if series.empty:
            continue

        date_like_ratio = series.str.match(_DATE_PATTERN_RE).mean()
        if date_like_ratio < 0.5 and not name_hint:
            continue  # doesn't look like a date at all -- skip parsing attempt

        sample = series.sample(min(200, len(series)), random_state=0)
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed", dayfirst=False)
        parse_ratio = parsed.notna().mean()
        if name_hint or parse_ratio >= min_parse_ratio:
            if parse_ratio >= 0.5:  # still require it to be plausibly date-like
                candidates.append(col)
    return candidates


def standardize_dates_auto(df: pd.DataFrame, columns=None) -> pd.DataFrame:
   
    cols = columns if columns is not None else auto_detect_date_columns(df)
    for col in cols:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col].astype(str), errors="coerce", format="mixed", dayfirst=False)
        n_before_missing = df[col].isna().sum()
        n_after_missing = parsed.isna().sum()

        has_time_component = (parsed.dt.time != pd.Timestamp("00:00:00").time()).any()
        output_format = "%Y-%m-%d %H:%M:%S" if has_time_component else "%Y-%m-%d"
        df[col] = parsed.dt.strftime(output_format)

        print(f"[type_cleaner] Standardized dates in '{col}' to {output_format} "
              f"({n_after_missing - n_before_missing} values could not be parsed)"
              f"{' -- time-of-day preserved' if has_time_component else ''}.")
    return df


# ---------------------------------------------------------------------------
# 4b. Gender standardization -- handled separately from generic fuzzy
#     category matching because single-letter abbreviations ("F", "M")
#     are too short for edit-distance/prefix matching to safely catch
#     against their full words ("Female", "Male") without risking false
#     matches elsewhere. Like booleans, this uses a small fixed vocabulary.
# ---------------------------------------------------------------------------
_GENDER_MAP = {
    "m": "Male", "male": "Male",
    "f": "Female", "female": "Female",
    "o": "Other", "other": "Other", "non-binary": "Other", "nonbinary": "Other",
}


def auto_detect_gender_columns(df: pd.DataFrame) -> list:
    candidates = []
    for col in df.select_dtypes(include=["object"]).columns:
        if "gender" in col.lower() or col.lower() == "sex":
            candidates.append(col)
    return candidates


def standardize_gender_columns(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Maps M/F/Male/Female/etc. variants to a consistent Male/Female/Other."""
    cols = columns if columns is not None else auto_detect_gender_columns(df)
    for col in cols:
        if col not in df.columns:
            continue

        def _map_gender(v):
            if pd.isna(v):
                return np.nan
            key = str(v).strip().lower()
            return _GENDER_MAP.get(key, v)  # leave unrecognized values untouched

        df[col] = df[col].apply(_map_gender)
        print(f"[type_cleaner] Standardized gender values in '{col}'.")
    return df


# ---------------------------------------------------------------------------
# 4c. Academic year-of-study standardization -- "Freshman"/"1st Year",
#     "Sophomore"/"2nd Year" are completely different words, not spelling
#     variants, so generic fuzzy matching can never safely merge them.
#     Like gender, this is a small, well-known, bounded vocabulary.
#
#     Rather than a fixed literal list (which breaks the moment someone
#     writes "Fresh." instead of "Freshman"), matching works in two layers:
#       1. A lookup of extremely common exact abbreviations (Jr, Sr, Fy, Sy...)
#       2. A punctuation-stripped PREFIX check against the canonical word
#          ("fresh" is a prefix of "freshman", "soph" of "sophomore", etc.)
#          -- this catches "Fresh.", "Soph.", "Jun.", "Sen." and similar
#          without needing to enumerate every possible abbreviation.
# ---------------------------------------------------------------------------
_YEAR_CANONICAL = {
    "1st Year": ["freshman", "freshmen"],
    "2nd Year": ["sophomore"],
    "3rd Year": ["junior"],
    "4th Year": ["senior"],
}
_YEAR_EXACT_MAP = {
    "1st year": "1st Year", "1st": "1st Year", "first year": "1st Year",
    "year 1": "1st Year", "fy": "1st Year", "yr1": "1st Year",
    "2nd year": "2nd Year", "2nd": "2nd Year", "second year": "2nd Year",
    "year 2": "2nd Year", "sy": "2nd Year", "yr2": "2nd Year",
    "3rd year": "3rd Year", "3rd": "3rd Year", "third year": "3rd Year",
    "year 3": "3rd Year", "jr": "3rd Year", "yr3": "3rd Year",
    "4th year": "4th Year", "4th": "4th Year", "fourth year": "4th Year",
    "year 4": "4th Year", "sr": "4th Year", "yr4": "4th Year",
}
_MIN_YEAR_PREFIX_LEN = 3


def _normalize_year_text(v: str) -> str:
    text = str(v).strip().lower()
    text = re.sub(r"[.\-_]", "", text)   # "Fresh." / "Soph." -> "fresh" / "soph"
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _match_year_canonical(norm_text: str):
    if norm_text in _YEAR_EXACT_MAP:
        return _YEAR_EXACT_MAP[norm_text]
    for canonical, words in _YEAR_CANONICAL.items():
        for word in words:
            if norm_text == word:
                return canonical
            if len(norm_text) >= _MIN_YEAR_PREFIX_LEN and word.startswith(norm_text):
                return canonical  # e.g. "fresh" / "soph" / "jun" / "sen" as a prefix
    return None


def auto_detect_year_of_study_columns(df: pd.DataFrame) -> list:
    """Finds columns using academic-year vocabulary (1st Year/Freshman/etc.)."""
    candidates = []
    for col in df.select_dtypes(include=["object"]).columns:
        if "year" not in col.lower():
            continue
        series = df[col].dropna().astype(str)
        if series.empty:
            continue
        match_ratio = series.apply(lambda v: _match_year_canonical(_normalize_year_text(v)) is not None).mean()
        if match_ratio >= 0.5:
            candidates.append(col)
    return candidates


def standardize_year_of_study_columns(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Maps Freshman/Sophomore/Junior/Senior and common abbreviations to a consistent '1st Year'.../'4th Year' label."""
    cols = columns if columns is not None else auto_detect_year_of_study_columns(df)
    for col in cols:
        if col not in df.columns:
            continue

        def _map_year(v):
            if pd.isna(v):
                return np.nan
            match = _match_year_canonical(_normalize_year_text(v))
            return match if match else v  # leave genuinely unrecognized values untouched

        df[col] = df[col].apply(_map_year)
        print(f"[type_cleaner] Standardized academic year values in '{col}'.")
    return df


# ---------------------------------------------------------------------------
# 4. Messy boolean text -> real True/False
# ---------------------------------------------------------------------------
def auto_detect_boolean_columns(df: pd.DataFrame) -> list:
    """Finds text columns whose values are entirely drawn from yes/no-style vocab."""
    candidates = []
    bool_vocab = _TRUE_VALUES | _FALSE_VALUES
    for col in df.select_dtypes(include=["object"]).columns:
        series = df[col].dropna().astype(str).str.strip().str.lower()
        if series.empty:
            continue
        # No nunique cap here -- membership in bool_vocab is already a strong
        # enough signal on its own (a real category column won't score 90%+
        # membership in {yes,no,y,n,true,false,1,0,t,f} by coincidence).
        if series.isin(bool_vocab).mean() >= 0.9:
            candidates.append(col)
    return candidates


def standardize_boolean_columns(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Converts messy yes/no/true/false/1/0 text into real Python booleans."""
    cols = columns if columns is not None else auto_detect_boolean_columns(df)
    for col in cols:
        if col not in df.columns:
            continue

        def _map_bool(v):
            if pd.isna(v):
                return np.nan
            text = str(v).strip().lower()
            if text in _TRUE_VALUES:
                return True
            if text in _FALSE_VALUES:
                return False
            return np.nan

        df[col] = df[col].apply(_map_bool)
        df[col] = df[col].astype("boolean")  # pandas nullable boolean dtype (supports NA)
        print(f"[type_cleaner] Standardized '{col}' to True/False boolean values.")
    return df


# ---------------------------------------------------------------------------
# 5. Malformed emails
# ---------------------------------------------------------------------------
def auto_detect_email_columns(df: pd.DataFrame) -> list:
    candidates = []
    for col in df.select_dtypes(include=["object"]).columns:
        if "email" in col.lower():
            candidates.append(col)
            continue
        series = df[col].dropna().astype(str)
        if series.empty:
            continue
        if series.str.contains("@").mean() >= 0.5:
            candidates.append(col)
    return candidates


def clean_email_columns(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    
    cols = columns if columns is not None else auto_detect_email_columns(df)
    for col in cols:
        if col not in df.columns:
            continue

        def _fix_email(v):
            if pd.isna(v):
                return np.nan
            text = str(v).strip().replace(" ", "")
            text = re.sub(r"@{2,}", "@", text)  # collapse "a@@b.com" -> "a@b.com"
            if _EMAIL_REGEX.match(text):
                return text
            return np.nan  # unrecoverable -> treat as missing

        before_valid = df[col].notna().sum()
        df[col] = df[col].apply(_fix_email)
        after_valid = df[col].notna().sum()
        print(f"[type_cleaner] Cleaned '{col}': {before_valid - after_valid} malformed "
              f"emails could not be recovered and were set to missing.")
    return df


# ---------------------------------------------------------------------------
# 6. Category column detection + casing standardization
# ---------------------------------------------------------------------------
def auto_detect_categorical_columns(df: pd.DataFrame, max_unique_ratio: float = 0.5,
                                     max_avg_length: int = 40, min_unique: int = 2,
                                     max_unique: int = 200) -> list:
   
    candidates = []
    n_rows = len(df)
    if n_rows == 0:
        return candidates

    already_handled = set(auto_detect_gender_columns(df)) | set(auto_detect_year_of_study_columns(df))

    for col in df.select_dtypes(include=["object"]).columns:
        series = df[col].dropna().astype(str)
        if series.empty:
            continue

        # Skip columns that are clearly emails or dates -- these are handled
        # by their own dedicated cleaners and should never get casing/fuzzy
        # "category" treatment (e.g. turning an email into Title Case).
        if series.str.contains("@").mean() >= 0.3:
            continue
        name_hint = any(k in col.lower() for k in ["email", "date", "dob", "_at", "time"])
        if name_hint:
            continue

        # Skip columns already handled by their own dedicated standardizer
        # (gender, academic year-of-study) -- re-applying generic Title Case
        # on top is redundant at best, and actively wrong for values like
        # "3rd Year" (Python's .title() capitalizes after the digit too,
        # producing "3Rd Year").
        if col in already_handled:
            continue

        n_unique = series.nunique()
        unique_ratio = n_unique / len(series)
        avg_len = series.str.len().mean()

        looks_like_category = (
            min_unique <= n_unique <= max_unique
            and unique_ratio <= max_unique_ratio
            and avg_len <= max_avg_length
        )
        if looks_like_category:
            candidates.append(col)

    return candidates


def standardize_category_casing(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    
    cols = columns if columns is not None else auto_detect_categorical_columns(df)
    for col in cols:
        if col not in df.columns:
            continue
        df[col] = df[col].apply(lambda v: v.strip().title() if isinstance(v, str) else v)
    if cols:
        print(f"[type_cleaner] Standardized casing for category columns: {cols}")
    return df


_NAME_COLUMN_RE = re.compile(r"(^|_)name($|_)", flags=re.IGNORECASE)


def auto_detect_name_columns(df: pd.DataFrame) -> list:
   
    candidates = []
    for col in df.select_dtypes(include=["object"]).columns:
        if _NAME_COLUMN_RE.search(col.lower()) and "username" not in col.lower():
            candidates.append(col)
    return candidates


def standardize_name_casing(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Title-cases person-name columns regardless of how many unique values they have."""
    cols = columns if columns is not None else auto_detect_name_columns(df)
    for col in cols:
        if col not in df.columns:
            continue
        df[col] = df[col].apply(lambda v: v.strip().title() if isinstance(v, str) else v)
    if cols:
        print(f"[type_cleaner] Standardized casing for name columns: {cols}")
    return df


# ---------------------------------------------------------------------------
# 7. Shared helper: detect ID-like columns (used to exclude them from
#    outlier detection/capping, since a unique ID isn't a meaningful value)
# ---------------------------------------------------------------------------
_ID_NAME_RE = re.compile(r"(^|_)(id|no|num|number|code)$", flags=re.IGNORECASE)


def drop_id_like_columns(df: pd.DataFrame, columns: list) -> list:
    
    return [col for col in columns if not _ID_NAME_RE.search(col.lower())]


# ---------------------------------------------------------------------------
# 8. Domain-rule bounds (things that are logically impossible, not just
#    statistically unusual -- IQR/Isolation Forest only catch "unusual
#    relative to the rest of the data", which can miss impossible values
#    if enough of the data shares the same problem)
# ---------------------------------------------------------------------------
_STUDENT_CONTEXT_HINTS = ["gpa", "semester_fee", "tuition", "major", "enrollment", "grade_level"]


def enforce_domain_bounds(df: pd.DataFrame, age_max: int = 120) -> pd.DataFrame:
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    is_student_dataset = any(
        any(hint in col.lower() for hint in _STUDENT_CONTEXT_HINTS) for col in df.columns
    )
    age_min_exclusive = 14 if is_student_dataset else 0

    for col in numeric_cols:
        col_lower = col.lower()

        if col_lower == "age" or col_lower.endswith("_age"):
            invalid = (df[col] <= age_min_exclusive) | (df[col] > age_max)
            n_invalid = int(invalid.sum())
            if n_invalid:
                df.loc[invalid, col] = np.nan
                context_note = " (student dataset detected, so a realistic minimum age was used)" if is_student_dataset else ""
                print(f"[type_cleaner] '{col}': {n_invalid} value(s) outside valid "
                      f"range ({age_min_exclusive}, {age_max}] set to missing for re-imputation{context_note}.")
            continue

        if any(hint in col_lower for hint in _PERCENT_NAME_HINTS):
            invalid = (df[col] < 0) | (df[col] > 100)
            n_invalid = int(invalid.sum())
            if n_invalid:
                df.loc[invalid, col] = np.nan
                print(f"[type_cleaner] '{col}': {n_invalid} value(s) outside valid "
                      f"range [0, 100] set to missing for re-imputation.")
            continue

        if any(hint in col_lower for hint in _MUST_BE_POSITIVE_NAME_HINTS):
            invalid = df[col] <= 0  # a $0 salary for an active employee is not realistic
            n_invalid = int(invalid.sum())
            if n_invalid:
                df.loc[invalid, col] = np.nan
                print(f"[type_cleaner] '{col}': {n_invalid} zero/negative value(s) "
                      f"(zero/negative is unrealistic here) set to missing for re-imputation.")
            continue

        if any(hint in col_lower for hint in _NON_NEGATIVE_NAME_HINTS):
            invalid = df[col] < 0
            n_invalid = int(invalid.sum())
            if n_invalid:
                df.loc[invalid, col] = np.nan
                print(f"[type_cleaner] '{col}': {n_invalid} negative value(s) "
                      f"(logically impossible) set to missing for re-imputation.")

    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_type_cleaning(df: pd.DataFrame, convert_currency: bool = False):
    
    df = standardize_missing_placeholders(df)

    multi_currency_cols = []
    if convert_currency:
        # Priority 1: dedicated currency-code column (reliable ground truth)
        currency_pairs = auto_detect_currency_column_pairs(df)
        for amount_col, currency_col in currency_pairs:
            df = convert_currency_via_column(df, amount_col, currency_col)
            multi_currency_cols.append(amount_col)
        # Priority 2: embedded symbol/text detection for any remaining columns
        already_handled = {c for c, _ in currency_pairs}
        symbol_based_cols = [c for c in auto_detect_multi_currency_columns(df) if c not in already_handled]
        multi_currency_cols.extend(symbol_based_cols)

    df = standardize_dates_auto(df)
    df = standardize_boolean_columns(df)
    df = standardize_gender_columns(df)
    df = standardize_year_of_study_columns(df)
    df = clean_email_columns(df)
    df = standardize_gpa_columns(df)
    df = clean_numeric_text_columns(df, convert_currency=convert_currency)
    df = enforce_domain_bounds(df)
    df = standardize_category_casing(df)
    df = standardize_name_casing(df)
    return df, multi_currency_cols
