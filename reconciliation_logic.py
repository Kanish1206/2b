import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz


def normalize_doc(series):
    return (
        series.fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )


def validate_columns(df, required_cols, df_name):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def process_reco(
    gst_df,
    pur_df,
    doc_threshold=85,
    tax_tolerance=10,
    gstin_mismatch_tolerance=20,
):

    gst = gst_df.copy()
    pur = pur_df.copy()

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

    gst["doc_norm"] = normalize_doc(gst["Document Number"])
    pur["doc_norm"] = normalize_doc(pur["Reference Document No."])

    pur.rename(columns={"GSTIN Of Vendor/Customer": "Supplier GSTIN"}, inplace=True)

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

    merged = gst_agg.merge(
        pur_agg,
        on=["Supplier GSTIN", "doc_norm"],
        how="outer",
        suffixes=("_2B", "_PUR"),
        indicator=True,
    )

    # Ensure numeric columns
    for col in [
        "IGST Amount_2B", "CGST Amount_2B", "SGST Amount_2B", "Invoice Value_2B",
        "IGST Amount_PUR", "CGST Amount_PUR", "SGST Amount_PUR", "Invoice Value_PUR",
    ]:
        if col not in merged.columns:
            merged[col] = 0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    merged["Match_Status"] = None
    merged["Fuzzy Score"] = 0.0

    # Diff
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

    # 🔥 FUZZY MATCHING FIXED
    for gstin in merged["Supplier GSTIN"].dropna().unique():

        open_2b = merged[
            (merged["Supplier GSTIN"] == gstin) &
            (merged["Match_Status"] == "Open in 2B")
        ]

        open_books = merged[
            (merged["Supplier GSTIN"] == gstin) &
            (merged["Match_Status"] == "Open in Books")
        ]

        for left_idx in open_2b.index:

            left_doc = merged.at[left_idx, "doc_norm"]
            left_invoice = merged.at[left_idx, "Invoice Value_2B"]

            candidates = open_books[
                (open_books["Invoice Value_PUR"] - left_invoice).abs() <= tax_tolerance
            ]

            if candidates.empty:
                continue

            candidate_dict = dict(zip(candidates.index, candidates["doc_norm"]))

            match = process.extractOne(
                left_doc,
                candidate_dict,
                scorer=fuzz.ratio,
                score_cutoff=doc_threshold,
            )

            if match:
                _, score, right_idx = match

                # Copy correct column names (NO _PUR suffix here)
                merged.at[left_idx, "Reference Document No."] = merged.at[right_idx, "Reference Document No."]
                merged.at[left_idx, "Vendor/Customer Name"] = merged.at[right_idx, "Vendor/Customer Name"]

                merged.at[left_idx, "IGST Amount_PUR"] = merged.at[right_idx, "IGST Amount_PUR"]
                merged.at[left_idx, "CGST Amount_PUR"] = merged.at[right_idx, "CGST Amount_PUR"]
                merged.at[left_idx, "SGST Amount_PUR"] = merged.at[right_idx, "SGST Amount_PUR"]
                merged.at[left_idx, "Invoice Value_PUR"] = merged.at[right_idx, "Invoice Value_PUR"]

                merged.at[left_idx, "Match_Status"] = "Fuzzy Match"
                merged.at[left_idx, "Fuzzy Score"] = score
                merged.at[right_idx, "Match_Status"] = "Fuzzy Consumed"

    merged = merged[merged["Match_Status"] != "Fuzzy Consumed"]

    merged.drop(columns=["_merge"], inplace=True)

    return merged
