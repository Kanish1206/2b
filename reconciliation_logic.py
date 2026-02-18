import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz


def normalize_doc(series):
    return (
        series.astype(str)
        .str.upper()
        .str.replace(r'[^A-Z0-9]', '', regex=True)
        .replace('NAN', '')
    )


def process_reco(gst, pur, doc_threshold=85, tax_tolerance=10, gstin_mismatch_tolerance=20):

    gst = gst.copy()
    pur = pur.copy()

    # -------------------------------------------------
    # 1️⃣ Normalize Documents
    # -------------------------------------------------
    gst["doc_norm"] = normalize_doc(gst["Document Number"])
    pur["doc_norm"] = normalize_doc(pur["Reference Document No."])

    # -------------------------------------------------
    # 2️⃣ Aggregate on NORMALIZED doc
    # -------------------------------------------------
    gst_agg = (
        gst.groupby(["Supplier GSTIN", "doc_norm"], as_index=False)
        .agg({
            "Document Number": "first",
            "Supplier Name": "first",
            "Document Date": "first",
            "IGST Amount": "sum",
            "CGST Amount": "sum",
            "SGST Amount": "sum",
            "Invoice Value": "sum"
        })
    )

    pur_agg = (
        pur.groupby(
            ["GSTIN Of Vendor/Customer", "doc_norm", "FI Document Number"],
            as_index=False
        )
        .agg({
            "Reference Document No.": "first",
            "Vendor/Customer Name": "first",
            "IGST Amount": "sum",
            "CGST Amount": "sum",
            "SGST Amount": "sum",
            "Invoice Value": "sum"
        })
        .rename(columns={
            "GSTIN Of Vendor/Customer": "Supplier GSTIN"
        })
    )

    # -------------------------------------------------
    # 3️⃣ Exact Merge (GSTIN + Doc)
    # -------------------------------------------------
    merged = gst_agg.merge(
        pur_agg,
        on=["Supplier GSTIN", "doc_norm"],
        how="outer",
        suffixes=("_2B", "_PUR"),
        indicator=True
    )

    merged["Match_Status"] = None
    merged["Fuzzy Score"] = 0.0

    # -------------------------------------------------
    # 4️⃣ Tax Difference Columns
    # -------------------------------------------------
    merged["IGST Diff"] = (
        merged["IGST Amount_PUR"].fillna(0)
        - merged["IGST Amount_2B"].fillna(0)
    )

    merged["CGST Diff"] = (
        merged["CGST Amount_PUR"].fillna(0)
        - merged["CGST Amount_2B"].fillna(0)
    )

    merged["SGST Diff"] = (
        merged["SGST Amount_PUR"].fillna(0)
        - merged["SGST Amount_2B"].fillna(0)
    )

    merged["Invoice Diff"] = (
        merged["Invoice Value_PUR"].fillna(0)
        - merged["Invoice Value_2B"].fillna(0)
    )

    both_mask = merged["_merge"] == "both"

    tax_match_condition = (
        (merged["IGST Diff"].abs() <= tax_tolerance) &
        (merged["CGST Diff"].abs() <= tax_tolerance) &
        (merged["SGST Diff"].abs() <= tax_tolerance)
    )

    merged.loc[both_mask & tax_match_condition, "Match_Status"] = "Exact Match"

    merged.loc[
        both_mask & ~tax_match_condition,
        "Match_Status"
    ] = "Exact Doc - Tax Mismatch"

    merged.loc[
        merged["_merge"] == "left_only",
        "Match_Status"
    ] = "Open in 2B"

    merged.loc[
        merged["_merge"] == "right_only",
        "Match_Status"
    ] = "Open in Books"

    # -------------------------------------------------
    # 5️⃣ Fuzzy Matching (Unmatched Only)
    # -------------------------------------------------
    left_df = merged[merged["Match_Status"] == "Open in 2B"].copy()
    right_df = merged[merged["Match_Status"] == "Open in Books"].copy()

    for gstin, left_grp in left_df.groupby("Supplier GSTIN"):

        right_grp = right_df[right_df["Supplier GSTIN"] == gstin]

        if right_grp.empty:
            continue

        for left_idx in left_grp.index:

            left_doc = merged.at[left_idx, "doc_norm"]

            candidates = right_grp["doc_norm"].to_dict()

            match = process.extractOne(
                left_doc,
                candidates,
                scorer=fuzz.ratio,
                score_cutoff=doc_threshold,
                processor=None
            )

            if match:
                _, score, right_idx = match

                igst_diff = abs(merged.at[right_idx, "IGST Amount_PUR"] - 
                                merged.at[left_idx, "IGST Amount_2B"])
                cgst_diff = abs(merged.at[right_idx, "CGST Amount_PUR"] - 
                                merged.at[left_idx, "CGST Amount_2B"])
                sgst_diff = abs(merged.at[right_idx, "SGST Amount_PUR"] - 
                                merged.at[left_idx, "SGST Amount_2B"])

                if (
                    igst_diff <= tax_tolerance and
                    cgst_diff <= tax_tolerance and
                    sgst_diff <= tax_tolerance
                ):

                    merged.at[left_idx, "Match_Status"] = "Fuzzy Match"
                    merged.at[left_idx, "Fuzzy Score"] = score

                    pur_cols = [c for c in merged.columns if c.endswith("_PUR")]
                    for col in pur_cols:
                        merged.at[left_idx, col] = merged.at[right_idx, col]

                    merged.at[right_idx, "Match_Status"] = "Fuzzy Consumed"

    merged = merged[merged["Match_Status"] != "Fuzzy Consumed"]

    # -------------------------------------------------
    # 6️⃣ GSTIN Mismatch Check
    # -------------------------------------------------
    open_2b = merged[merged["Match_Status"] == "Open in 2B"]
    open_books = merged[merged["Match_Status"] == "Open in Books"]

    for left_idx in open_2b.index:

        left_doc = merged.at[left_idx, "doc_norm"]

        possible = open_books[open_books["doc_norm"] == left_doc]

        if possible.empty:
            continue

        for right_idx in possible.index:

            igst_diff = abs(
                (merged.at[left_idx, "IGST Amount_2B"] or 0)
                - (merged.at[right_idx, "IGST Amount_PUR"] or 0)
            )

            cgst_diff = abs(
                (merged.at[left_idx, "CGST Amount_2B"] or 0)
                - (merged.at[right_idx, "CGST Amount_PUR"] or 0)
            )

            sgst_diff = abs(
                (merged.at[left_idx, "SGST Amount_2B"] or 0)
                - (merged.at[right_idx, "SGST Amount_PUR"] or 0)
            )

            if (
                0 <= igst_diff <= gstin_mismatch_tolerance and
                0 <= cgst_diff <= gstin_mismatch_tolerance and
                0 <= sgst_diff <= gstin_mismatch_tolerance
            ):

                merged.at[left_idx, "Match_Status"] = "GSTIN Mismatch"
                merged.at[right_idx, "Match_Status"] = "GSTIN Mismatch"

    merged.drop(columns=["_merge"], inplace=True)

    return merged
