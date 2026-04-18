import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz

# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------
MATCH_EXACT = "Exact Match"
MATCH_VALUE_MISMATCH = "Value Mismatch"
MATCH_OPEN_2B = "Open in 2B"
MATCH_OPEN_BOOKS = "Open in Books"
MATCH_FUZZY = "Fuzzy Match"
MATCH_FUZZY_CONSUMED = "Fuzzy Consumed"
MATCH_GSTIN_MISMATCH = "GSTIN Mismatch"
MATCH_PAN = "PAN Match (GSTIN Variation)"
MATCH_PAN_CONSUMED = "PAN Consumed"

# -------------------------------------------------
def normalize_doc(series):
    return (
        series.fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )

# -------------------------------------------------
def validate_columns(df, required_cols, df_name):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")

# -------------------------------------------------
def compute_diffs(df):
    df["IGST Diff"] = df["IGST Amount_PUR"] - df["IGST Amount_2B"]
    df["CGST Diff"] = df["CGST Amount_PUR"] - df["CGST Amount_2B"]
    df["SGST Diff"] = df["SGST Amount_PUR"] - df["SGST Amount_2B"]
    return df

# -------------------------------------------------
def process_reco(
    gst_df,
    pur_df,
    doc_threshold=60,
    tax_tolerance=10,
    gstin_mismatch_tolerance=5,
):

    gst = gst_df.copy()
    pur = pur_df.copy()

    # ---------------- VALIDATION ----------------
    gst_required = [
        "Supplier GSTIN", "Document Number", "Document Date",
        "Return Period", "Taxable Value", "Supplier Name",
        "IGST Amount", "CGST Amount", "SGST Amount", "Invoice Value",
    ]

    pur_required = [
        "GSTIN Of Vendor/Customer", "Reference Document No.",
        "Taxable Amount", "Document Date",
        "Vendor/Customer Name", "IGST Amount", "CGST Amount",
        "SGST Amount", "Invoice Value", "FI Document Number"
    ]

    validate_columns(gst, gst_required, "2B File")
    validate_columns(pur, pur_required, "Purchase File")

    # ---------------- PREP ----------------
    pur["Vendor/Customer GSTIN"] = pur["GSTIN Of Vendor/Customer"]
    gst["Return Period"] = gst["Return Period"].astype(str)

    gst["doc_norm"] = normalize_doc(gst["Document Number"])
    pur["doc_norm"] = normalize_doc(pur["Reference Document No."])

    pur.rename(columns={"GSTIN Of Vendor/Customer": "Supplier GSTIN"}, inplace=True)

    # ---------------- AGG ----------------
    gst_agg = gst.groupby(["Supplier GSTIN", "doc_norm"], as_index=False).agg({
        "Document Number": "first",
        "Return Period": "first",
        "Supplier Name": "first",
        "Document Date": "first",
        "IGST Amount": "sum",
        "CGST Amount": "sum",
        "SGST Amount": "sum",
        "Taxable Value": "sum",
        "Invoice Value": "sum",
    })

    pur_agg = pur.groupby(["Supplier GSTIN", "doc_norm"], as_index=False).agg({
        "Reference Document No.": "first",
        "Vendor/Customer GSTIN": "first",
        "FI Document Number": "first",
        "Vendor/Customer Name": "first",
        "Document Date": "first",
        "Taxable Amount": "sum",
        "IGST Amount": "sum",
        "CGST Amount": "sum",
        "SGST Amount": "sum",
        "Invoice Value": "sum",
    })

    # ---------------- MERGE ----------------
    merged = gst_agg.merge(
        pur_agg,
        on=["Supplier GSTIN", "doc_norm"],
        how="outer",
        suffixes=["_2B", "_PUR"],
        indicator=True,
    )

    # ---------------- NUMERIC CLEAN ----------------
    numeric_cols = [
        "IGST Amount_2B", "CGST Amount_2B", "SGST Amount_2B",
        "Invoice Value_2B", "Taxable Value",
        "IGST Amount_PUR", "CGST Amount_PUR", "SGST Amount_PUR",
        "Invoice Value_PUR", "Taxable Amount",
    ]

    for col in numeric_cols:
        if col not in merged.columns:
            merged[col] = 0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    # ---------------- INITIAL MATCH ----------------
    merged = compute_diffs(merged)
    merged["Match_Status"] = None
    merged["Fuzzy Score"] = 0.0

    both_mask = merged["_merge"] == "both"

    tax_condition = (
        (merged["IGST Diff"].abs() <= tax_tolerance) &
        (merged["CGST Diff"].abs() <= tax_tolerance) &
        (merged["SGST Diff"].abs() <= tax_tolerance)
    )

    merged.loc[both_mask & tax_condition, "Match_Status"] = MATCH_EXACT
    merged.loc[both_mask & ~tax_condition, "Match_Status"] = MATCH_VALUE_MISMATCH
    merged.loc[merged["_merge"] == "left_only", "Match_Status"] = MATCH_OPEN_2B
    merged.loc[merged["_merge"] == "right_only", "Match_Status"] = MATCH_OPEN_BOOKS

    # ---------------- FUZZY MATCH ----------------
    for gstin in merged["Supplier GSTIN"].dropna().unique():

        open_2b = merged[(merged["Supplier GSTIN"] == gstin) & (merged["Match_Status"] == MATCH_OPEN_2B)]
        open_books = merged[(merged["Supplier GSTIN"] == gstin) & (merged["Match_Status"] == MATCH_OPEN_BOOKS)]

        for left_idx in open_2b.index:

            left_doc = str(merged.at[left_idx, "Document Number"])
            if not left_doc or open_books.empty:
                continue

            candidates = open_books.copy()

            candidates["tax_score"] = (
                (candidates["IGST Amount_PUR"] - merged.at[left_idx, "IGST Amount_2B"]).abs() +
                (candidates["CGST Amount_PUR"] - merged.at[left_idx, "CGST Amount_2B"]).abs() +
                (candidates["SGST Amount_PUR"] - merged.at[left_idx, "SGST Amount_2B"]).abs()
            )

            candidates = candidates[candidates["tax_score"] <= tax_tolerance * 2]

            if candidates.empty:
                continue

            match = process.extractOne(
                left_doc,
                dict(zip(candidates.index, candidates["Reference Document No."].astype(str))),
                scorer=fuzz.partial_token_set_ratio,
                score_cutoff=doc_threshold
            )

            if match:
                _, score, right_idx = match

                for col in [
                    "Reference Document No.", "FI Document Number",
                    "Vendor/Customer GSTIN", "Vendor/Customer Name",
                    "IGST Amount_PUR", "CGST Amount_PUR", "SGST Amount_PUR",
                    "Taxable Amount", "Invoice Value_PUR"
                ]:
                    merged.at[left_idx, col] = merged.at[right_idx, col]

                merged.at[left_idx, "Match_Status"] = MATCH_FUZZY
                merged.at[left_idx, "Fuzzy Score"] = score
                merged.at[right_idx, "Match_Status"] = MATCH_FUZZY_CONSUMED

                open_books = open_books.drop(index=right_idx)

    # ---------------- GSTIN MISMATCH ----------------
    open_2b = merged[merged["Match_Status"] == MATCH_OPEN_2B]
    open_books = merged[merged["Match_Status"] == MATCH_OPEN_BOOKS]

    for left_idx in open_2b.index:

        doc = merged.at[left_idx, "doc_norm"]
        if not doc:
            continue

        left_val = merged.at[left_idx, "Invoice Value_2B"]
        left_igst = merged.at[left_idx, "IGST Amount_2B"]
        left_cgst = merged.at[left_idx, "CGST Amount_2B"]
        left_sgst = merged.at[left_idx, "SGST Amount_2B"]

        possible = open_books[open_books["doc_norm"] == doc]

        for right_idx in possible.index:

            right_val = merged.at[right_idx, "Invoice Value_PUR"]
            right_igst = merged.at[right_idx, "IGST Amount_PUR"]
            right_cgst = merged.at[right_idx, "CGST Amount_PUR"]
            right_sgst = merged.at[right_idx, "SGST Amount_PUR"]

            if (
                abs(left_val - right_val) <= gstin_mismatch_tolerance and
                abs(left_igst - right_igst) <= tax_tolerance and
                abs(left_cgst - right_cgst) <= tax_tolerance and
                abs(left_sgst - right_sgst) <= tax_tolerance
            ):
                merged.at[left_idx, "Match_Status"] = MATCH_GSTIN_MISMATCH
                merged.at[right_idx, "Match_Status"] = MATCH_GSTIN_MISMATCH
                break

    # ---------------- PAN MATCH ----------------
    open_2b_for_pan = merged[merged["Match_Status"] == MATCH_OPEN_2B]
    open_books_for_pan = merged[merged["Match_Status"] == MATCH_OPEN_BOOKS]

    merged["PAN_2B"] = merged["Supplier GSTIN"].astype(str).str[2:12]
    merged["PAN_PUR"] = merged["Vendor/Customer GSTIN"].astype(str).str[2:12]

    for left_idx in open_2b_for_pan.index:

        pan = merged.at[left_idx, "PAN_2B"]
        doc = merged.at[left_idx, "doc_norm"]

        if not pan or not doc:
            continue

        candidates = merged[
            (merged.index.isin(open_books_for_pan.index)) &
            (merged["PAN_PUR"] == pan) &
            (merged["doc_norm"] == doc)
        ]

        for right_idx in candidates.index:
            merged.at[left_idx, "Match_Status"] = MATCH_PAN
            merged.at[right_idx, "Match_Status"] = MATCH_PAN_CONSUMED
            break

    # ---------------- FINAL MATCH (IGNORE GSTIN) ----------------
    open_2b_final = merged[merged["Match_Status"] == MATCH_OPEN_2B]
    open_books_final = merged[merged["Match_Status"] == MATCH_OPEN_BOOKS]

    for left_idx in open_2b_final.index:

        doc = merged.at[left_idx, "doc_norm"]
        if not doc:
            continue

        candidates = open_books_final[open_books_final["doc_norm"] == doc]

        for right_idx in candidates.index:
            merged.at[left_idx, "Match_Status"] = "Doc Match (Ignore GSTIN)"
            merged.at[right_idx, "Match_Status"] = "Doc Consumed (Ignore GSTIN)"
            break

    # ---------------- CLEANUP ----------------
    merged = merged[~merged["Match_Status"].isin([
        MATCH_FUZZY_CONSUMED,
        MATCH_PAN_CONSUMED,
        "Doc Consumed (Ignore GSTIN)"
    ])]

    merged.drop(columns=["PAN_2B", "PAN_PUR", "_merge"], inplace=True, errors="ignore")

    return merged
