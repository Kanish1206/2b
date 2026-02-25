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
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


# -------------------------------------------------
# 3️⃣ MAIN RECON FUNCTION
# -------------------------------------------------
def process_reco(gst_df, pur_df, doc_threshold=85, tax_tolerance=10):

    gst = gst_df.copy()
    pur = pur_df.copy()

    # ---------------- REQUIRED COLUMNS ----------------
    gst_required = [
        "Supplier GSTIN", "Document Number", "Document Date",
        "Return Period", "Taxable Value",
        "Supplier Name", "IGST Amount", "CGST Amount",
        "SGST Amount", "Invoice Value",
    ]

    pur_required = [
        "GSTIN Of Vendor/Customer", "Reference Document No.",
        "Taxable Amount", "Document Date", "FI Document Number",
        "Vendor/Customer Code", "Vendor/Customer Name",
        "IGST Amount", "CGST Amount",
        "SGST Amount", "Invoice Value",
    ]

    validate_columns(gst, gst_required, "2B File")
    validate_columns(pur, pur_required, "Purchase File")

    # ---------------- NORMALIZE ----------------
    gst["doc_norm"] = normalize_doc(gst["Document Number"])
    pur["doc_norm"] = normalize_doc(pur["Reference Document No."])

    pur.rename(columns={"GSTIN Of Vendor/Customer": "Supplier GSTIN"}, inplace=True)

    # ---------------- AGGREGATION ----------------
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
        "Vendor/Customer Name": "first",
        "Vendor/Customer Code": "first",
        "Document Date": "first",
        "FI Document Number": "first",
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
        suffixes=("_2B", "_PUR"),
        indicator=True,
    )

    # ---------------- NUMERIC CLEANING ----------------
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

    # ---------------- DIFF CALCULATION ----------------
    merged["IGST Diff"] = merged["IGST Amount_PUR"] - merged["IGST Amount_2B"]
    merged["CGST Diff"] = merged["CGST Amount_PUR"] - merged["CGST Amount_2B"]
    merged["SGST Diff"] = merged["SGST Amount_PUR"] - merged["SGST Amount_2B"]
    merged["Invoice Diff"] = merged["Invoice Value_PUR"] - merged["Invoice Value_2B"]
    merged["Taxable Diff"] = merged["Taxable Amount"] - merged["Taxable Value"]

    merged["Match_Status"] = None
    merged["Fuzzy Score"] = 0.0

    both_mask = merged["_merge"] == "both"

    tax_condition = (
        (merged["IGST Diff"].abs() <= tax_tolerance) &
        (merged["CGST Diff"].abs() <= tax_tolerance) &
        (merged["SGST Diff"].abs() <= tax_tolerance) &
        (merged["Invoice Diff"].abs() <= tax_tolerance) &
        (merged["Taxable Diff"].abs() <= tax_tolerance)
    )

    merged.loc[both_mask & tax_condition, "Match_Status"] = "Exact Match"
    merged.loc[both_mask & ~tax_condition, "Match_Status"] = "Exact Doc - Value Mismatch"
    merged.loc[merged["_merge"] == "left_only", "Match_Status"] = "Open in 2B"
    merged.loc[merged["_merge"] == "right_only", "Match_Status"] = "Open in Books"

    # ---------------- GSTIN MISMATCH LOGIC ----------------
    merged["GSTIN_Match_With"] = None

    open_rows = merged[
        merged["Match_Status"].isin(["Open in 2B", "Open in Books"])
    ]

    for idx in open_rows.index:

        row = merged.loc[idx]

        igst = row["IGST Amount_2B"] if row["IGST Amount_2B"] != 0 else row["IGST Amount_PUR"]
        cgst = row["CGST Amount_2B"] if row["CGST Amount_2B"] != 0 else row["CGST Amount_PUR"]
        sgst = row["SGST Amount_2B"] if row["SGST Amount_2B"] != 0 else row["SGST Amount_PUR"]
        doc  = row["doc_norm"]

        potential = merged[
            (merged.index != idx) &
            (merged["Supplier GSTIN"] != row["Supplier GSTIN"]) &
            (merged["doc_norm"] == doc) &
            (merged["IGST Amount_2B"].sub(igst).abs() <= tax_tolerance) &
            (merged["CGST Amount_2B"].sub(cgst).abs() <= tax_tolerance) &
            (merged["SGST Amount_2B"].sub(sgst).abs() <= tax_tolerance)
        ]

        if not potential.empty:
            match_idx = potential.index[0]
            merged.at[idx, "Match_Status"] = "GSTIN Mismatch"
            merged.at[idx, "GSTIN_Match_With"] = merged.at[match_idx, "Supplier GSTIN"]

    merged.drop(columns=["_merge"], inplace=True)

    return merged
