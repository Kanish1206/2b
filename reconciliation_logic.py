import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz


# -------------------------------------------------
# Robust Column Cleaning
# -------------------------------------------------
def clean_columns(df):
    df.columns = (
        df.columns
        .astype(str)
        .str.replace(r'\xa0', '', regex=True)   # remove non-breaking spaces
        .str.replace(r'\s+', ' ', regex=True)   # normalize multiple spaces
        .str.strip()
        .str.replace("'", "", regex=False)
        .str.replace('"', "", regex=False)
        .str.upper()                            # make everything uppercase
    )
    return df


# -------------------------------------------------
# Normalize Document Number
# -------------------------------------------------
def normalize_doc(series):
    return (
        series.astype(str)
        .str.upper()
        .str.replace(r'[^A-Z0-9]', '', regex=True)
        .replace('NAN', '')
    )


# -------------------------------------------------
# Main Reconciliation Function
# -------------------------------------------------
def process_reco(gst, pur,
                 doc_threshold=85,
                 tax_tolerance=10,
                 gstin_mismatch_tolerance=20):

    gst = gst.copy()
    pur = pur.copy()

    # Clean columns first
    gst = clean_columns(gst)
    pur = clean_columns(pur)

    # -------------------------------------------------
    # Required Columns (UPPERCASE)
    # -------------------------------------------------
    gst_required = [
        "SUPPLIER GSTIN",
        "DOCUMENT NUMBER",
        "SUPPLIER NAME",
        "DOCUMENT DATE",
        "IGST AMOUNT",
        "CGST AMOUNT",
        "SGST AMOUNT",
        "INVOICE VALUE"
    ]

    pur_required = [
        "GSTIN OF VENDOR/CUSTOMER",
        "REFERENCE DOCUMENT NO.",
        "VENDOR/CUSTOMER NAME",
        "FI DOCUMENT NUMBER",
        "IGST AMOUNT",
        "CGST AMOUNT",
        "SGST AMOUNT",
        "INVOICE VALUE"
    ]

    missing_gst = [col for col in gst_required if col not in gst.columns]
    missing_pur = [col for col in pur_required if col not in pur.columns]

    if missing_gst:
        raise ValueError(f"Missing GST Columns: {missing_gst}")

    if missing_pur:
        raise ValueError(f"Missing Purchase Columns: {missing_pur}")

    # -------------------------------------------------
    # Normalize Document Numbers
    # -------------------------------------------------
    gst["DOC_NORM"] = normalize_doc(gst["DOCUMENT NUMBER"])
    pur["DOC_NORM"] = normalize_doc(pur["REFERENCE DOCUMENT NO."])

    # -------------------------------------------------
    # Aggregate GST
    # -------------------------------------------------
    gst_agg = (
        gst.groupby(["SUPPLIER GSTIN", "DOC_NORM"], as_index=False)
        .agg({
            "DOCUMENT NUMBER": "first",
            "SUPPLIER NAME": "first",
            "DOCUMENT DATE": "first",
            "IGST AMOUNT": "sum",
            "CGST AMOUNT": "sum",
            "SGST AMOUNT": "sum",
            "INVOICE VALUE": "sum"
        })
    )

    # -------------------------------------------------
    # Aggregate Purchase
    # -------------------------------------------------
    pur_agg = (
        pur.groupby(
            ["GSTIN OF VENDOR/CUSTOMER", "DOC_NORM", "FI DOCUMENT NUMBER"],
            as_index=False
        )
        .agg({
            "REFERENCE DOCUMENT NO.": "first",
            "VENDOR/CUSTOMER NAME": "first",
            "IGST AMOUNT": "sum",
            "CGST AMOUNT": "sum",
            "SGST AMOUNT": "sum",
            "INVOICE VALUE": "sum"
        })
        .rename(columns={
            "GSTIN OF VENDOR/CUSTOMER": "SUPPLIER GSTIN"
        })
    )

    # -------------------------------------------------
    # Exact Merge
    # -------------------------------------------------
    merged = gst_agg.merge(
        pur_agg,
        on=["SUPPLIER GSTIN", "DOC_NORM"],
        how="outer",
        suffixes=("_2B", "_PUR"),
        indicator=True
    )

    merged["MATCH_STATUS"] = None
    merged["FUZZY SCORE"] = 0.0

    # -------------------------------------------------
    # Tax Differences
    # -------------------------------------------------
    merged["IGST DIFF"] = (
        merged["IGST AMOUNT_PUR"].fillna(0)
        - merged["IGST AMOUNT_2B"].fillna(0)
    )

    merged["CGST DIFF"] = (
        merged["CGST AMOUNT_PUR"].fillna(0)
        - merged["CGST AMOUNT_2B"].fillna(0)
    )

    merged["SGST DIFF"] = (
        merged["SGST AMOUNT_PUR"].fillna(0)
        - merged["SGST AMOUNT_2B"].fillna(0)
    )

    merged["INVOICE DIFF"] = (
        merged["INVOICE VALUE_PUR"].fillna(0)
        - merged["INVOICE VALUE_2B"].fillna(0)
    )

    both_mask = merged["_merge"] == "both"

    tax_match_condition = (
        (merged["IGST DIFF"].abs() <= tax_tolerance) &
        (merged["CGST DIFF"].abs() <= tax_tolerance) &
        (merged["SGST DIFF"].abs() <= tax_tolerance)
    )

    merged.loc[both_mask & tax_match_condition, "MATCH_STATUS"] = "Exact Match"

    merged.loc[both_mask & ~tax_match_condition,
               "MATCH_STATUS"] = "Exact Doc - Tax Mismatch"

    merged.loc[merged["_merge"] == "left_only",
               "MATCH_STATUS"] = "Open in 2B"

    merged.loc[merged["_merge"] == "right_only",
               "MATCH_STATUS"] = "Open in Books"

    # -------------------------------------------------
    # Fuzzy Matching (Same GSTIN)
    # -------------------------------------------------
    left_df = merged[merged["MATCH_STATUS"] == "Open in 2B"]
    right_df = merged[merged["MATCH_STATUS"] == "Open in Books"]

    for gstin, left_grp in left_df.groupby("SUPPLIER GSTIN"):

        right_grp = right_df[right_df["SUPPLIER GSTIN"] == gstin]

        for left_idx in left_grp.index:

            left_doc = merged.at[left_idx, "DOC_NORM"]
            candidates = right_grp["DOC_NORM"].to_dict()

            match = process.extractOne(
                left_doc,
                candidates,
                scorer=fuzz.ratio,
                score_cutoff=doc_threshold,
                processor=None
            )

            if match:
                _, score, right_idx = match

                igst_diff = abs(merged.at[right_idx, "IGST AMOUNT_PUR"] -
                                merged.at[left_idx, "IGST AMOUNT_2B"])
                cgst_diff = abs(merged.at[right_idx, "CGST AMOUNT_PUR"] -
                                merged.at[left_idx, "CGST AMOUNT_2B"])
                sgst_diff = abs(merged.at[right_idx, "SGST AMOUNT_PUR"] -
                                merged.at[left_idx, "SGST AMOUNT_2B"])

                if (
                    igst_diff <= tax_tolerance and
                    cgst_diff <= tax_tolerance and
                    sgst_diff <= tax_tolerance
                ):
                    merged.at[left_idx, "MATCH_STATUS"] = "Fuzzy Match"
                    merged.at[left_idx, "FUZZY SCORE"] = score
                    merged.at[right_idx, "MATCH_STATUS"] = "Fuzzy Consumed"

    merged = merged[merged["MATCH_STATUS"] != "Fuzzy Consumed"]

    # -------------------------------------------------
    # GSTIN Mismatch (Doc match + tax 0–20)
    # -------------------------------------------------
    open_2b = merged[merged["MATCH_STATUS"] == "Open in 2B"]
    open_books = merged[merged["MATCH_STATUS"] == "Open in Books"]

    for left_idx in open_2b.index:

        left_doc = merged.at[left_idx, "DOC_NORM"]
        possible = open_books[open_books["DOC_NORM"] == left_doc]

        for right_idx in possible.index:

            igst_diff = abs(
                merged.at[left_idx, "IGST AMOUNT_2B"] -
                merged.at[right_idx, "IGST AMOUNT_PUR"]
            )
            cgst_diff = abs(
                merged.at[left_idx, "CGST AMOUNT_2B"] -
                merged.at[right_idx, "CGST AMOUNT_PUR"]
            )
            sgst_diff = abs(
                merged.at[left_idx, "SGST AMOUNT_2B"] -
                merged.at[right_idx, "SGST AMOUNT_PUR"]
            )

            if (
                igst_diff <= gstin_mismatch_tolerance and
                cgst_diff <= gstin_mismatch_tolerance and
                sgst_diff <= gstin_mismatch_tolerance
            ):
                merged.at[left_idx, "MATCH_STATUS"] = "GSTIN Mismatch"
                merged.at[right_idx, "MATCH_STATUS"] = "GSTIN Mismatch"

    merged.drop(columns=["_merge"], inplace=True)

    return merged
