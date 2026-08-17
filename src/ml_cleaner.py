import re
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import IsolationForest
from rapidfuzz import fuzz

from type_cleaner import auto_detect_categorical_columns, drop_id_like_columns  # shared detection logic

# Prefixes that flip a word's meaning -- used to stop fuzzy matching from
# merging opposites like "Active"/"Inactive" or "Legal"/"Illegal" just
# because they're textually similar.
_NEGATION_PREFIXES = ["in", "im", "un", "non", "dis", "ir", "anti"]

# Minimum stem length before the "one is a prefix of the other" rule kicks
# in (e.g. "cloth" is a prefix of "clothing") -- avoids short words like
# "an" matching almost anything.
_MIN_PREFIX_LEN = 4


_KNN_ROW_LIMIT = 20_000  # above this, KNN's O(n^2) neighbor search becomes impractically slow


def impute_missing_knn(df: pd.DataFrame, columns=None, n_neighbors: int = 5) -> pd.DataFrame:
    
    numeric_cols = columns if columns else list(df.select_dtypes(include=[np.number]).columns)
    numeric_cols = [c for c in numeric_cols if df[c].isnull().any()]

    if not numeric_cols:
        return df

    # Decide up front which columns look "integer-like" from their
    # ORIGINAL (pre-imputation) values, and capture their observed range.
    integer_like = {}
    for col in numeric_cols:
        observed = df[col].dropna()
        if len(observed) == 0:
            continue
        frac_whole = np.isclose(observed % 1, 0).mean()
        if frac_whole >= 0.98:
            integer_like[col] = (observed.min(), observed.max())

    if len(df) > _KNN_ROW_LIMIT:
        imputer = SimpleImputer(strategy="median")
        method_note = f"median (dataset has {len(df)} rows, too large for KNN to run in reasonable time)"
    else:
        imputer = KNNImputer(n_neighbors=n_neighbors)
        method_note = "KNN"

    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    for col, (obs_min, obs_max) in integer_like.items():
        df[col] = df[col].round().clip(lower=obs_min, upper=obs_max)

    print(f"[ml_cleaner] {method_note}-imputed missing values in columns: {numeric_cols}"
          + (f" (rounded to whole numbers: {list(integer_like)})" if integer_like else ""))
    return df


def impute_categorical_mode(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    
    cols = columns if columns is not None else auto_detect_categorical_columns(df)
    # extra safety: never mode-impute obvious identifier/contact columns
    # even if they slipped through the categorical detector
    blocked_hints = ["email", "phone", "mobile", "contact", "name", "id", "address"]
    cols = [c for c in cols if not any(h in c.lower() for h in blocked_hints)]

    filled_summary = {}
    for col in cols:
        if col not in df.columns:
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        mode_series = df[col].mode(dropna=True)
        if mode_series.empty:
            continue  # entire column is missing -- nothing sensible to fill with
        mode_value = mode_series.iloc[0]
        df[col] = df[col].fillna(mode_value)
        filled_summary[col] = (n_missing, mode_value)

    for col, (n_missing, mode_value) in filled_summary.items():
        print(f"[ml_cleaner] Filled {n_missing} missing value(s) in '{col}' "
              f"with most common value: '{mode_value}'.")
    return df


def detect_outliers_isolation_forest(df: pd.DataFrame, columns=None, contamination: float = 0.02) -> pd.DataFrame:
    
    numeric_cols = columns if columns else list(df.select_dtypes(include=[np.number]).columns)
    numeric_cols = [c for c in numeric_cols if c != "__is_outlier__"]
    numeric_cols = drop_id_like_columns(df, numeric_cols)
    numeric_df = df[numeric_cols].dropna()

    if numeric_df.empty or len(numeric_df.columns) == 0:
        df["__is_outlier__"] = False
        print("[ml_cleaner] No usable numeric columns for outlier detection "
              "(check that numeric-text columns were converted first).")
        return df

    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(numeric_df)  # -1 = outlier, 1 = normal

    df["__is_outlier__"] = False
    df.loc[numeric_df.index, "__is_outlier__"] = preds == -1
    outlier_count = int((df["__is_outlier__"]).sum())
    print(f"[ml_cleaner] Isolation Forest flagged {outlier_count} rows as outliers "
          f"(using columns: {list(numeric_df.columns)}, contamination={contamination}).")
    return df


# Common whole-value acronyms (mostly academic/business degree & department
# names) -- expanded to their full form before comparison so e.g. "CS" and
# "Computer Science" normalize to the exact same stem and merge safely.
_ACRONYM_VALUE_MAP = {
    "cs": "computer science", "cse": "computer science",
    "it": "information technology", "ee": "electrical engineering",
    "me": "mechanical engineering", "ce": "civil engineering",
    "hr": "human resources", "bba": "business administration",
    "mba": "business administration", "ai": "artificial intelligence",
    "econ": "economics",
    "ib": "internet banking",
}

# Common word-level truncations seen inside multi-word category values
# (e.g. "Bus. Admin" -> "business administration", "Elec. Engg" ->
# "electrical engineering", "Atm W/D" -> "atm withdrawal"). Applied
# token-by-token after acronym lookup.
_WORD_ABBREV_MAP = {
    "bus": "business", "admin": "administration", "elec": "electrical",
    "engg": "engineering", "mgmt": "management",
    "dept": "department", "info": "information", "tech": "technology",
    "sci": "science", "comm": "commerce", "acct": "accounting",
    "fin": "finance", "mech": "mechanical",
    "wd": "withdrawal", "net": "internet", "mob": "mobile",
    "txn": "transaction", "pos": "point of sale", "pmt": "payment",
    "ops": "operations", "cust": "customer", "bio": "biology",
}


def _expand_abbreviations(text: str) -> str:
    """Expands known acronyms/word-truncations to their full form before stemming."""
    if text in _ACRONYM_VALUE_MAP:
        return _ACRONYM_VALUE_MAP[text]
    tokens = text.split(" ")
    expanded = [_WORD_ABBREV_MAP.get(t, t) for t in tokens]
    return " ".join(expanded)


def _stem(word: str) -> str:
    """
    Very small, dependency-free plural stemmer: Groceries -> Grocery,
    Toys -> Toy, Resources -> Resource (NOT the wrong "Resourc").
    """
    w = word.lower().strip()
    w = re.sub(r"\s*&\s*", " and ", w)  # "Home & Kitchen" -> "home and kitchen"
    w = re.sub(r"-", " ", w)             # "Drop-Out" -> "drop out" (keep as separate words)
    w = re.sub(r"[^a-z0-9 ]", "", w)     # drop stray punctuation
    w = re.sub(r"\s+", " ", w).strip()
    w = _expand_abbreviations(w)
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith("es"):
        # Only strip BOTH letters for unambiguous "-es" plurals (boxes->box,
        # watches->watch, dishes->dish). Words ending in plain "-ses" are
        # ambiguous (classes->class needs both stripped, but resources->
        # resource / sales->sale / prices->price only need the final "s"
        # stripped) -- default to the safer single-letter strip, since
        # common business terms (resources, sales, prices, services) are
        # far more likely to appear as category values than words like
        # "classes"/"buses".
        if w.endswith(("xes", "ches", "shes")):
            return w[:-2]
        return w[:-1]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _is_negation_pair(a: str, b: str) -> bool:
    """True if one word is the other with a negation prefix (Active/Inactive, Legal/Illegal)."""
    for prefix in _NEGATION_PREFIXES:
        if a == prefix + b or b == prefix + a:
            return True
    return False


def _is_prefix_variant(a: str, b: str) -> bool:
   
    shorter, longer = sorted([a, b], key=len)
    if len(shorter) < _MIN_PREFIX_LEN:
        return False
    return longer.startswith(shorter)


_MIN_TOKEN_PREFIX_LEN = 2


def _is_token_prefix_variant(a: str, b: str) -> bool:
    
    tokens_a, tokens_b = a.split(" "), b.split(" ")
    if len(tokens_a) != len(tokens_b) or len(tokens_a) < 2:
        return False
    for ta, tb in zip(tokens_a, tokens_b):
        shorter, longer = sorted([ta, tb], key=len)
        if len(shorter) < _MIN_TOKEN_PREFIX_LEN or not longer.startswith(shorter):
            return False
    return True


def fuzzy_standardize_categories(df: pd.DataFrame, column: str, similarity_threshold: int = 85) -> pd.DataFrame:
   
    if column not in df.columns:
        return df

    values = df[column].dropna().astype(str).unique().tolist()
    n = len(values)
    if n <= 1:
        return df

    freq = df[column].astype(str).value_counts()
    stems = {v: _stem(v) for v in values}

    # --- union-find setup ---
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path halving
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            si, sj = stems[values[i]], stems[values[j]]
            if _is_negation_pair(si, sj):
                continue  # never merge opposites, regardless of similarity score
            score = fuzz.ratio(si, sj)
            same_ignoring_spaces = si.replace(" ", "") == sj.replace(" ", "")
            token_prefix_match = _is_token_prefix_variant(si, sj)
            if score >= similarity_threshold or _is_prefix_variant(si, sj) or same_ignoring_spaces or token_prefix_match:
                union(i, j)

    groups = {}
    for i, v in enumerate(values):
        groups.setdefault(find(i), []).append(v)

    mapping = {}
    for group_values in groups.values():
        canonical = max(group_values, key=lambda v: freq.get(v, 0))
        for v in group_values:
            mapping[v] = canonical

    df[column] = df[column].astype(str).map(mapping).fillna(df[column])
    changed = sum(1 for k, v in mapping.items() if k != v)
    print(f"[ml_cleaner] Fuzzy-standardized '{column}': merged {changed} variant spellings.")
    return df


def run_ml_cleaning(df: pd.DataFrame, fuzzy_columns=None, auto_detect_fuzzy: bool = True,
                     fill_categorical_missing: bool = False) -> pd.DataFrame:
  
    df = impute_missing_knn(df)
    df = detect_outliers_isolation_forest(df)

    if fuzzy_columns:
        target_columns = fuzzy_columns
    elif auto_detect_fuzzy:
        target_columns = auto_detect_categorical_columns(df)
        if target_columns:
            print(f"[ml_cleaner] Auto-detected category columns for fuzzy matching: {target_columns}")
    else:
        target_columns = []

    for col in target_columns:
        df = fuzzy_standardize_categories(df, col)

    if fill_categorical_missing:
        df = impute_categorical_mode(df, columns=target_columns if target_columns else None)
    else:
        remaining = {col: int(df[col].isna().sum()) for col in target_columns if df[col].isna().sum() > 0}
        if remaining:
            print(f"[ml_cleaner] Left as missing (not guessed/imputed) -- consistent with "
                  f"email/phone treatment: {remaining}")

    return df
