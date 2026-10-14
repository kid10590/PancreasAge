import os
import re
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import DATA_ROOT  # noqa: E402

"""Automatically scan the ICD-10 code table to identify patients with pancreatic and major somatic diseases, extract the first matching disease code and its corresponding onset date, flag whether it occurred before the exam, and output the augmented annotation results."""

INPUT_CSV = os.path.join(DATA_ROOT, "icd-10-all.csv")
OUTPUT_CSV = os.path.join(DATA_ROOT, "icd-10-all_with_flags.csv")
ICD_COL = "icd-10"   # target ICD-10 column name
DATE_COL = "date"     # exam date column name (must be parseable by pandas.to_datetime)
EID_COL = "eid"       # subject ID column

# ICD-10 conditions to match
EXACT_STARTS = [
    "K85", #acute pancreatitis
    "C25",  #malignant neoplasm of pancreas
    "D13.6", #benign neoplasm of pancreas
    "K86.2",   #other diseases of pancreas
    "Q45.0",    #pancreatic agenesis
    "S36.2",#pancreatic injury
    "K86"  
    
]
LETTER_RANGES = [
    ("I", 0, 52),   # I00–I52   # various heart diseases
    ("I", 10, 15),  # I10–I15   # hypertension
    ("N", 0, 8),    # N00–N08   # nephritis
    ("E", 10, 14),  # E10–E14   # diabetes
    ("C", 0, 97),   # C00–C97   # malignant neoplasms
    ("Q", 0, 99),   # Q00–Q99   # genetic/congenital diseases
]

code_regex = re.compile(r"^[A-Za-z]\d{2}(?:\.[A-Za-z0-9]+)?")

def scan_all_matches(icd_field: str, row: pd.Series, exam_dt: pd.Timestamp):
    """
    Iterate over all tokens in this row's icd-10 field and tally the relationship to the exam time:
    returns dict: {any_match, pre, post_or_same, unknown}
    """
    stats = {"any_match": False, "pre": 0, "post_or_same": 0, "unknown": 0}
    if not isinstance(icd_field, str) or not icd_field.strip() or pd.isna(exam_dt):
        return stats

    parts = [p.strip() for p in icd_field.split("|")]
    for idx, part in enumerate(parts, start=0):
        code = normalize_code(part)
        if not code or not matches_target(code):
            continue
        stats["any_match"] = True
        col = f"p41280_a{idx}"
        onset_raw = row[col] if col in row.index else pd.NA
        onset_dt = pd.to_datetime(onset_raw, errors="coerce")
        if pd.isna(onset_dt):
            stats["unknown"] += 1
        else:
            if onset_dt < exam_dt:
                stats["pre"] += 1
            else:
                stats["post_or_same"] += 1
    return stats

def normalize_code(token: str) -> str | None:
    """
    Extract the ICD-10 code portion (dropping the description); returns a standard form like 'I10' or 'K86.1'.
    """
    if not isinstance(token, str):
        return None
    token = token.strip()
    if not token:
        return None
    # ICD-10 is before the space, or extracted with a regex
    m = code_regex.match(token.upper())
    return m.group(0) if m else None

def in_letter_range(code: str, letter: str, start: int, end: int) -> bool:
    """
    Check whether code falls within a letter + two-digit range such as I00–I52 (endpoints included).
    """
    m = re.match(r"^([A-Z])(\d{2})(?:\..*)?$", code)
    if not m:
        return False
    if m.group(1) != letter:
        return False
    num = int(m.group(2))
    return start <= num <= end

def matches_target(code: str) -> bool:
    """
    Match as specified:
    - K85 (including subcategories)
    - K86.0–K86.1 (including subcategories), K86.2 (including subcategories)
    - C25, D13.6, Q45.0, S36.2 (including subcategories)
    - I00–I52, I10–I15, N00–N08, E10–E14, C00–C97, Q00–Q99
    """
    up = code.upper()

    # K86.0–K86.1, K86.2 (including their subcategories)
    if re.match(r"^K86\.(0|1|2)(?:\..*)?$", up):
        return True

    # Other exact-prefix matches (including subcategories)
    for p in EXACT_STARTS:
        if up.startswith(p):
            return True

    # Letter-range matches
    for letter, s, e in LETTER_RANGES:
        if in_letter_range(up, letter, s, e):
            return True
    return False

def find_first_match_index(icd_field: str) -> tuple[int | None, str | None]:
    """
    Find the first matching entry from left to right in the pipe-separated ICD column (0-based index).
    Returns (index_0_based, normalized ICD code); returns (None, None) if not found.
    """
    if not isinstance(icd_field, str) or not icd_field.strip():
        return None, None
    parts = [p.strip() for p in icd_field.split("|")]
    for idx, part in enumerate(parts, start=0):
        code = normalize_code(part)
        if code and matches_target(code):
            return idx, code
    return None, None

def find_first_match_with_onset(icd_field: str, row: pd.Series):
    """
    Prefer to return the first entry that both matches and has an onset date;
    if all matching entries lack an onset date, return the earliest match with onset as NaT.
    Returns (idx_0_based, code, onset_dt)
    """
    if not isinstance(icd_field, str) or not icd_field.strip():
        return None, None, pd.NaT

    parts = [p.strip() for p in icd_field.split("|")]

    first_idx, first_code = None, None
    # First record the earliest match (ignoring time), then keep looking for the first match that has a time
    for idx, part in enumerate(parts, start=0):
        code = normalize_code(part)
        if code and matches_target(code):
            if first_idx is None:
                first_idx, first_code = idx, code
            col = f"p41280_a{idx}"  # note: indices are 0-based, corresponding to p41280_a0, a1, ...
            onset_raw = row[col] if col in row.index else pd.NA
            onset_dt = pd.to_datetime(onset_raw, errors="coerce")
            if pd.notna(onset_dt):
                return idx, code, onset_dt

    # No match with a time found, fall back to the earliest match (onset as NaT)
    if first_idx is not None:
        return first_idx, first_code, pd.NaT
    return None, None, pd.NaT

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Input file not found: {INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    missing_cols = [c for c in [EID_COL, ICD_COL, DATE_COL] if c not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        sys.exit(1)

    # Parse dates
    df["_exam_date"] = pd.to_datetime(df[DATE_COL], errors="coerce")

    # Process row by row
    out = {
        EID_COL: [],
        "matched": [],             # whether any target ICD was hit (1/0)
        "matched_index": [],       # selected match index (0-based)
        "matched_code": [],        # selected ICD code
        "onset_date": [],          # onset date of the selected entry
        "exam_date": [],           # date
        "onset_before_exam": [],   # 1: onset before exam; 0: onset after or same day; NaN: missing date
    }

    # Collect all possible p41280_a column names to avoid checking existence row by row
    pcols = [c for c in df.columns if c.startswith("p41280_a")]
    # Suggested column names are p41280_a1, p41280_a2, ...; if your column name format differs, adjust this accordingly.
    def onset_col_name(i: int) -> str:
        return f"p41280_a{i}"

    for _, row in df.iterrows():
        eid = row[EID_COL]
        icd_field = row.get(ICD_COL, "")

        idx_any, _ = find_first_match_index(icd_field)
        idx, code, onset_dt = find_first_match_with_onset(icd_field, row)

        exam_dt = row["_exam_date"]
        stats = scan_all_matches(icd_field, row, exam_dt)

        out[EID_COL].append(eid)
        out["exam_date"].append(exam_dt)
        out["matched"].append(1 if idx_any is not None else 0)
        out["matched_index"].append(idx if idx is not None else pd.NA)
        out["matched_code"].append(code if code is not None else pd.NA)
        out["onset_date"].append(onset_dt)

        if pd.notna(onset_dt) and pd.notna(exam_dt):
            out["onset_before_exam"].append(1 if onset_dt < exam_dt else 0)
        else:
            out["onset_before_exam"].append(pd.NA)

        # Additional statistics columns
        out.setdefault("any_match", []).append(1 if stats["any_match"] else 0)
        out.setdefault("pre_exam_count", []).append(stats["pre"])
        out.setdefault("post_or_same_count", []).append(stats["post_or_same"])
        out.setdefault("unknown_onset_count", []).append(stats["unknown"])
        any_pre = stats["pre"] > 0
        out.setdefault("any_pre_exam", []).append(1 if any_pre else 0)

        # Health determination
        if not stats["any_match"]:
            is_healthy = 1
        elif stats["pre"] > 0:
            is_healthy = 0
        elif stats["pre"] == 0 and stats["post_or_same"] > 0 and stats["unknown"] == 0:
            is_healthy = 1
        else:
            is_healthy = pd.NA
        out.setdefault("is_healthy", []).append(is_healthy)

    result = pd.DataFrame(out)
    # Merge back into the original table; to output only the result columns, save result alone
    merged = df.merge(
        result[[EID_COL, "matched", "matched_index", "matched_code",
                "onset_date", "exam_date", "onset_before_exam",
                "any_match", "pre_exam_count", "post_or_same_count",
                "unknown_onset_count", "any_pre_exam", "is_healthy"]],
        on=EID_COL, how="left"
    )

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Done → {OUTPUT_CSV}")
    print(f"- Matched: {int(result['matched'].sum())} / {len(result)}")

if __name__ == "__main__":
    main()
