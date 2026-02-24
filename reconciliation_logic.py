import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz


# -------------------------------------------------
# 1️⃣ NORMALIZE DOCUMENT
# -------------------------------------------------
def normalize_doc(series):
    return (
        series.fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )


# -------------------------------------------------
# 2️⃣ COLUMN VALIDATION
# -------------------------------------------------
def validate_columns(df, required_cols, df_name):
    missing = [col for col in df.columns if col not in required_cols]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


# -------------------------------------------------
# 3️⃣ MAIN RECONCILIATION FUNCTION
# -------------------------------------------------
def process_reco(
    gst_df,
    pur_df,
    doc_threshold=85,
    tax_tolerance=10,
    gstin_mismatch_tolerance=20,
):

    gst = gst_df.copy()
    pur = pur_df.copy()

    # Required columns
    gst_required = [
        "Supplier GSTIN", "Document Number", "Document Date",
        "Supplier Name", "IGST Amount", "CGST Amount",
        "SGST Amount", "Invoice Value",
    ]

    pur_required = [
        "GSTIN Of Vendor/Customer", "Reference Document No.",
        "Vendor/Customer Name", "IGST Amount",
        "CGST Amount", "SGST Amount", "Invoice Value",
    ]

    validate_columns(gst, gst_required, "2B File")
    validate_columns(pur, pur_required, "Purchase File")

    # Normalize document numbers
    gst["doc_norm"] = normalize_doc(gst["Document Number"])
    pur["doc_norm"] = normalize_doc(pur["Reference Document No."])

    pur.rename(columns={"GSTIN Of Vendor/Customer": "Supplier GSTIN"}, inplace=True)

    # Aggregation
    gst_agg = gst.groupby(["Supplier GSTIN", "doc_norm"], as_index=False).agg({
        "Document Number": "first",
        "Supplier Name": "first",
        "Document Date": "first",
        "IGST Amount": "sum",
        "CGST Amount": "sum",
        "SGST Amount": "sum",
        "Invoice Value": "sum",
    })

    pur_agg = pur.groupby(["Supplier GSTIN", "doc_norm"], as_index=False).agg({
        "Reference Document No.": "first",
        "Vendor/Customer Name": "first",
        "IGST Amount": "sum",
        "CGST Amount": "sum",
        "SGST Amount": "sum",
        "Invoice Value": "sum",
    })

    # Merge
    merged = gst_agg.merge(
        pur_agg,
        on=["Supplier GSTIN", "doc_norm"],
        how="outer",
        suffixes=("_2B", "_PUR"),
        indicator=True,
    )

    # Clean numeric columns
    numeric_cols = [
        "IGST Amount_2B", "CGST Amount_2B", "SGST Amount_2B", "Invoice Value_2B",
        "IGST Amount_PUR", "CGST Amount_PUR", "SGST Amount_PUR", "Invoice Value_PUR",
    ]

    for col in numeric_cols:
        if col not in merged.columns:
            merged[col] = 0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    merged["Match_Status"] = None
    merged["Fuzzy Score"] = 0.0

    # Diff calculation
    merged["IGST Diff"] = merged["IGST Amount_PUR"] - merged["IGST Amount_2B"]
    merged["CGST Diff"] = merged["CGST Amount_PUR"] - merged["CGST Amount_2B"]
    merged["SGST Diff"] = merged["SGST Amount_PUR"] - merged["SGST Amount_2B"]
    merged["Invoice Diff"] = merged["Invoice Value_PUR"] - merged["Invoice Value_2B"]

    both_mask = merged["_merge"] == "both"

    tax_condition = (
        (merged["IGST Diff"].abs() <= tax_tolerance)
        & (merged["CGST Diff"].abs() <= tax_tolerance)
        & (merged["SGST Diff"].abs() <= tax_tolerance)
        & (merged["Invoice Diff"].abs() <= tax_tolerance)
    )

    merged.loc[both_mask & tax_condition, "Match_Status"] = "Exact Match"
    merged.loc[both_mask & ~tax_condition, "Match_Status"] = "Exact Doc - Value Mismatch"
    merged.loc[merged["_merge"] == "left_only", "Match_Status"] = "Open in 2B"
    merged.loc[merged["_merge"] == "right_only", "Match_Status"] = "Open in Books"

    # -------------------------------------------------
    # FUZZY MATCHING (FIXED)
    # -------------------------------------------------
    for gstin in merged["Supplier GSTIN"].dropna().unique():

        open_2b = merged[
            (merged["Supplier GSTIN"] == gstin) &
            (merged["Match_Status"] == "Open in 2B")
        ]

        open_books = merged[
            (merged["Supplier GSTIN"] == gstin) &
            (merged["Match_Status"] == "Open in Books")
        ]

        if open_books.empty:
            continue

        for left_idx in open_2b.index:

            left_doc = merged.at[left_idx, "doc_norm"]
            left_invoice = merged.at[left_idx, "Invoice Value_2B"]

            candidate_grp = open_books[
                (open_books["Invoice Value_PUR"] - left_invoice).abs() <= tax_tolerance
            ]

            if candidate_grp.empty:
                continue

            candidate_dict = dict(zip(candidate_grp.index, candidate_grp["doc_norm"]))

            match = process.extractOne(
                left_doc,
                candidate_dict,
                scorer=fuzz.ratio,
                score_cutoff=doc_threshold,
            )

            if match:
                _, score, right_idx = match

                # 🔥 COPY PURCHASE VALUES INTO LEFT ROW
                for col in [
                    "Reference Document No._PUR",
                    "Vendor/Customer Name_PUR",
                    "IGST Amount_PUR",
                    "CGST Amount_PUR",
                    "SGST Amount_PUR",
                    "Invoice Value_PUR",
                ]:
                    merged.at[left_idx, col] = merged.at[right_idx, col]

                # Update status
                merged.at[left_idx, "Match_Status"] = "Fuzzy Match"
                merged.at[left_idx, "Fuzzy Score"] = score
                merged.at[right_idx, "Match_Status"] = "Fuzzy Consumed"

                # Recalculate diffs
                merged.at[left_idx, "IGST Diff"] = (
                    merged.at[left_idx, "IGST Amount_PUR"]
                    - merged.at[left_idx, "IGST Amount_2B"]
                )
                merged.at[left_idx, "CGST Diff"] = (
                    merged.at[left_idx, "CGST Amount_PUR"]
                    - merged.at[left_idx, "CGST Amount_2B"]
                )
                merged.at[left_idx, "SGST Diff"] = (
                    merged.at[left_idx, "SGST Amount_PUR"]
                    - merged.at[left_idx, "SGST Amount_2B"]
                )
                merged.at[left_idx, "Invoice Diff"] = (
                    merged.at[left_idx, "Invoice Value_PUR"]
                    - merged.at[left_idx, "Invoice Value_2B"]
                )

    merged = merged[merged["Match_Status"] != "Fuzzy Consumed"]

    # -------------------------------------------------
    # GSTIN MISMATCH CHECK
    # -------------------------------------------------
    open_2b = merged[merged["Match_Status"] == "Open in 2B"]
    open_books = merged[merged["Match_Status"] == "Open in Books"]

    for left_idx in open_2b.index:

        left_doc = merged.at[left_idx, "doc_norm"]
        left_val = merged.at[left_idx, "Invoice Value_2B"]

        possible = open_books[open_books["doc_norm"] == left_doc]

        for right_idx in possible.index:

            right_val = merged.at[right_idx, "Invoice Value_PUR"]

            if abs(left_val - right_val) <= gstin_mismatch_tolerance:

                merged.at[left_idx, "Match_Status"] = "GSTIN Mismatch"
                merged.at[right_idx, "Match_Status"] = "GSTIN Mismatch"

    merged.drop(columns=["_merge"], inplace=True)

    return merged
