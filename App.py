import streamlit as st
import pandas as pd
import io
import time
import reconciliation_logic as reco_logic

# ---------------- HELPER FUNCTIONS (To ensure standalone execution) ----------------
def extract_pan(series):
    # Standard GSTIN format: First 2 chars = State Code, Next 10 chars = PAN
    return series.astype(str).str[2:12]

def fmt_amt(val):
    try:
        return f"₹{float(val):,.2f}"
    except (ValueError, TypeError):
        return "₹0.00"

ALL_STATUSES = [
    "Exact Match", "Fuzzy Match", "Mismatch", 
    "Open in 2B", "Open in Books", 
    "Manual Match", "Manual Match (Consumed)"
]

# Ensure fallback for reco_logic constants if not defined in user's file
MATCH_OPEN_2B = getattr(reco_logic, 'MATCH_OPEN_2B', "Open in 2B")
MATCH_OPEN_BOOKS = getattr(reco_logic, 'MATCH_OPEN_BOOKS', "Open in Books")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="GST Reco Pro | Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- STATE INITIALIZATION ----------------
if "result_df" not in st.session_state:
    st.session_state["result_df"] = None
if "manual_matches" not in st.session_state:
    st.session_state["manual_matches"] = []

# ---------------- ULTRA-MODERN CSS INJECTION ----------------
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #F0F4F8; }
    
    /* Core Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes pulseButton {
        0% { box-shadow: 0 0 0 0 rgba(249, 115, 22, 0.7); transform: scale(1); }
        50% { box-shadow: 0 0 0 15px rgba(249, 115, 22, 0); transform: scale(1.02); }
        100% { box-shadow: 0 0 0 0 rgba(249, 115, 22, 0); transform: scale(1); }
    }
    @keyframes pulseGreen {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); transform: scale(1); }
        50% { box-shadow: 0 0 0 15px rgba(16, 185, 129, 0); transform: scale(1.02); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); transform: scale(1); }
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes popIn {
        0% { opacity: 0; transform: scale(0.9); }
        100% { opacity: 1; transform: scale(1); }
    }
    @keyframes floatDrop {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
        100% { transform: translateY(0px); }
    }

    .animate-fade { animation: fadeIn 0.6s ease-out forwards; }
    .animate-left { animation: slideInLeft 0.6s ease-out forwards; }
    .animate-right { animation: slideInRight 0.6s ease-out forwards; }
    .floating-anim { animation: floatDrop 3s ease-in-out infinite; }

    /* Hero Header */
    .hero-header {
        background: linear-gradient(-45deg, #0F172A, #1E3A8A, #F97316, #EA580C);
        background-size: 400% 400%;
        animation: gradientBG 8s ease infinite, fadeIn 0.6s ease-out forwards;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(30, 58, 138, 0.2);
    }
    
    /* KPI Cards */
    .kpi-container { display: flex; justify-content: space-between; gap: 1rem; margin-bottom: 2rem; }
    .kpi-card {
        background: white; padding: 1.5rem; border-radius: 12px; flex: 1;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-bottom: 4px solid #3B82F6;
        text-align: center; transition: transform 0.3s ease;
        animation: fadeIn 0.8s ease-out forwards;
    }
    .kpi-card:hover { transform: translateY(-5px); box-shadow: 0 8px 15px rgba(0,0,0,0.1); }
    .kpi-card.orange { border-bottom: 4px solid #F97316; }
    .kpi-value { font-size: 2.2rem; font-weight: 800; color: #0F172A; margin: 0.5rem 0; }
    .kpi-label { font-size: 0.9rem; color: #64748B; text-transform: uppercase; font-weight: 600; }

    /* Standard Action Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #F97316 0%, #EA580C 100%);
        color: white; border: none; padding: 0.8rem 2rem; border-radius: 50px; 
        font-weight: bold; font-size: 1.1rem; transition: all 0.3s ease;
        animation: pulseButton 2s infinite; 
    }
    .stButton>button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 6px 20px rgba(249, 115, 22, 0.5);
        color: white; animation: none; 
    }
    
    /* UNDO BUTTON SPECIFIC STYLE */
    button[disabled] {
        background: #CBD5E1 !important;
        box-shadow: none !important;
        animation: none !important;
        cursor: not-allowed;
    }
    div:nth-child(2) > div > button { 
        /* Targeting the second button column for Undo */
        background: linear-gradient(135deg, #64748B 0%, #475569 100%);
        animation: none;
    }
    div:nth-child(2) > div > button:hover:not([disabled]) {
        box-shadow: 0 6px 20px rgba(100, 116, 139, 0.5);
    }
    
    /* 📥 DOWNLOAD BUTTON OVERRIDE (Green Glowing Pulse) */
    [data-testid="stDownloadButton"] button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4) !important;
        animation: pulseGreen 2.5s infinite !important;
        width: 100%;
        margin-top: 10px;
    }
    [data-testid="stDownloadButton"] button:hover {
        transform: translateY(-2px) scale(1.03) !important;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.6) !important;
        animation: none !important;
    }

    /* Filter & Search / Manual Match Maker Custom CSS */
    .side-header-2b {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white; padding: 12px 15px; border-radius: 8px; 
        font-weight: 700; font-size: 1.1rem; margin-bottom: 15px;
        box-shadow: 0 4px 10px rgba(59, 130, 246, 0.2);
    }
    .side-header-pur {
        background: linear-gradient(135deg, #EA580C 0%, #F97316 100%);
        color: white; padding: 12px 15px; border-radius: 8px; 
        font-weight: 700; font-size: 1.1rem; margin-bottom: 15px;
        box-shadow: 0 4px 10px rgba(249, 115, 22, 0.2);
    }
    .mm-row-card {
        background: white; border-left: 5px solid #3B82F6;
        padding: 12px 15px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 10px; font-size: 0.9rem; line-height: 1.5; color: #1E293B;
        transition: all 0.2s ease; animation: popIn 0.4s ease-out forwards;
    }
    .mm-row-card:hover { transform: translateX(5px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .mm-row-card b { color: #0F172A; }
    
    /* Ledger Container Frame */
    .ledger-container {
        background: white; padding: 1.5rem; border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        border: 1px solid #E2E8F0;
        margin-top: 20px;
    }

    /* Empty State */
    .empty-state {
        background: white; padding: 4rem 2rem; text-align: center;
        border-radius: 16px; border: 2px dashed #CBD5E1; color: #64748B; margin-top: 2rem;
        transition: all 0.3s ease;
    }
    .empty-state:hover { border-color: #3B82F6; background-color: #F8FAFC; }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER SECTION ----------------
st.markdown("""
    <div class="hero-header">
        <h1 style='margin:0; font-size: 3rem; font-weight: 800; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
            <span class="floating-anim" style="display:inline-block;">⚡</span> GST Intelligence Hub
        </h1>
        <p style='margin:5px 0 0 0; font-size: 1.2rem; opacity: 0.9;'>Automated GSTR-2B vs Books Reconciliation</p>
    </div>
""", unsafe_allow_html=True)

# ---------------- UPLOAD ZONE ----------------
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 📘 GSTR-2B Data")
    gst_file = st.file_uploader("Drop GSTR-2B Excel here", type=["xlsx"], key="gst", label_visibility="collapsed")
with col2:
    st.markdown("#### 📙 Purchase Register")
    pur_file = st.file_uploader("Drop Purchase Books Excel here", type=["xlsx"], key="pur", label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- PROCESS TRIGGER ----------------
if gst_file and pur_file:
    if st.session_state["result_df"] is None:
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            run_btn = st.button("🚀 INITIATE PROCESS", use_container_width=True)

        if run_btn:
            with st.status("⚡ Initiating Intelligence Engine...", expanded=True) as status:
                st.write("📥 Ingesting GSTR-2B and Purchase Data...")
                time.sleep(0.5)
                
                df_2b = pd.read_excel(gst_file)
                df_books = pd.read_excel(pur_file)
                df_2b.columns = df_2b.columns.str.strip()
                df_books.columns = df_books.columns.str.strip()
                
                st.write("🔍 Running Fuzzy Logic & Exact Match Algorithms...")
                result_df = reco_logic.process_reco(df_2b, df_books)
                
                st.write("📊 Finalizing Discrepancy Analytics...")
                time.sleep(0.5)
                
                status.update(label="✅ Reconciliation Complete!", state="complete", expanded=False)
            
            st.session_state["result_df"] = result_df
            st.balloons()
            st.rerun()

# ---------------- RESULTS DISPLAY & MANUAL MATCHING ----------------
if st.session_state["result_df"] is not None:
    result_df = st.session_state["result_df"]
    
    st.markdown('<div class="animate-fade">', unsafe_allow_html=True)
    
    # --- CALCULATE METRICS ---
    total = len(result_df)
    is_match = result_df["Match_Status"].str.contains("Match", case=False, na=False)
    is_fuzzy = result_df["Match_Status"].str.contains("Fuzzy", case=False, na=False)
    matched = (is_match & ~is_fuzzy).sum()
    unmatched = total - matched

    # --- CUSTOM KPI CARDS ---
    st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-card" style="animation-delay: 0.1s;">
                <div class="kpi-label">Total Invoices Processed</div>
                <div class="kpi-value">{total:,}</div>
            </div>
            <div class="kpi-card" style="border-bottom-color: #10B981; animation-delay: 0.3s;">
                <div class="kpi-label">Perfect Matches</div>
                <div class="kpi-value" style="color: #10B981;">{matched:,}</div>
            </div>
            <div class="kpi-card orange" style="animation-delay: 0.5s;">
                <div class="kpi-label">Discrepancies / Open</div>
                <div class="kpi-value" style="color: #F97316;">{unmatched:,}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    #  SPLIT FILTER + SEARCH  ──  2B (left) | Books (right)
    # ════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    present_statuses = sorted(result_df["Match_Status"].dropna().unique().tolist())
    ordered_statuses = [s for s in ALL_STATUSES if s in present_statuses]
    ordered_statuses += [s for s in present_statuses if s not in ordered_statuses]

    st.markdown('<div class="animate-fade" style="animation-delay: 0.2s;">', unsafe_allow_html=True)
    st.markdown("### 🔍 Filter & Search")
    fs_left, fs_right = st.columns(2)

    with fs_left:
        st.markdown('<div class="side-header-2b animate-left">📘 GSTR-2B — Filter & Search</div>', unsafe_allow_html=True)

        twoB_status = st.multiselect(
            "Filter by Status",
            options=ordered_statuses,
            default=[],
            placeholder="Select one or more statuses…",
            key="filter_2b"
        )
        sb2, sv2 = st.columns([1, 2])
        with sb2:
            twoB_search_by = st.selectbox("Search by", options=["— None —", "GSTIN", "PAN"], key="search_by_2b")
        with sv2:
            twoB_search_val = st.text_input("2B search value", placeholder="Type GSTIN or PAN…", key="search_val_2b", label_visibility="collapsed")

    with fs_right:
        st.markdown('<div class="side-header-pur animate-right">📙 Purchase Register — Filter & Search</div>', unsafe_allow_html=True)

        pur_status = st.multiselect(
            "Filter by Status",
            options=ordered_statuses,
            default=[],
            placeholder="Select one or more statuses…",
            key="filter_pur"
        )
        sbp, svp = st.columns([1, 2])
        with sbp:
            pur_search_by = st.selectbox("Search by", options=["— None —", "GSTIN", "PAN"], key="search_by_pur")
        with svp:
            pur_search_val = st.text_input("Books search value", placeholder="Type GSTIN or PAN…", key="search_val_pur", label_visibility="collapsed")

    # ── Check if any panel has input ─────────────────────────
    twoB_has_input  = bool(twoB_status) or (twoB_search_by != "— None —" and twoB_search_val.strip())
    books_has_input = bool(pur_status)  or (pur_search_by  != "— None —" and pur_search_val.strip())
    any_input       = twoB_has_input or books_has_input

    # ── Build display dataframe ──────────────────────────────
    work = result_df.copy()
    work["_PAN_2B"]  = extract_pan(work.get("Supplier GSTIN", pd.Series(dtype=str)))
    work["_PAN_PUR"] = extract_pan(work.get("Vendor/Customer GSTIN", pd.Series(dtype=str)))

    if any_input:
        masks = []
        if twoB_has_input:
            m = pd.Series(True, index=work.index)
            if twoB_status:
                m &= work["Match_Status"].isin(twoB_status)
            if twoB_search_by != "— None —" and twoB_search_val.strip():
                q = twoB_search_val.strip().upper()
                if twoB_search_by == "GSTIN":
                    col_gstin = work.get("Supplier GSTIN", pd.Series("", index=work.index))
                    m &= col_gstin.fillna("").astype(str).str.upper().str.contains(q, regex=False)
                else:
                    m &= work["_PAN_2B"].str.contains(q, regex=False)
            masks.append(m)

        if books_has_input:
            m = pd.Series(True, index=work.index)
            if pur_status:
                m &= work["Match_Status"].isin(pur_status)
            if pur_search_by != "— None —" and pur_search_val.strip():
                q = pur_search_val.strip().upper()
                if pur_search_by == "GSTIN":
                    col_gstin = work.get("Vendor/Customer GSTIN", pd.Series("", index=work.index))
                    m &= col_gstin.fillna("").astype(str).str.upper().str.contains(q, regex=False)
                else:
                    m &= work["_PAN_PUR"].str.contains(q, regex=False)
            masks.append(m)

        combined = masks[0]
        for m in masks[1:]:
            combined = combined | m

        filtered = work[combined]
        display_df = filtered.drop(columns=["_PAN_2B", "_PAN_PUR"], errors="ignore")
        
        st.markdown("#### ✨ Filtered Results")
        st.dataframe(display_df, use_container_width=True, height=250)
    else:
        filtered = pd.DataFrame(columns=work.columns)

    st.markdown('</div>', unsafe_allow_html=True)


    # ════════════════════════════════════════════════════════
    #  MANUAL MATCH MAKER (Only displays based on search)
    # ════════════════════════════════════════════════════════
    st.markdown('<div class="animate-fade" style="animation-delay: 0.3s;">', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🤝 Manual Match Maker")

    if not any_input:
        st.info("👆 Please use the **Filter & Search** fields above to populate the Manual Match Maker.")
    else:
        # Extract open rows strictly from the FILTERED data
        open_2b_rows    = filtered[filtered["Match_Status"] == MATCH_OPEN_2B]
        open_books_rows = filtered[filtered["Match_Status"] == MATCH_OPEN_BOOKS]

        if open_2b_rows.empty and open_books_rows.empty:
             st.success("✨ No 'Open' records found in your current search results.")
        else:
            st.markdown(
                "<small style='color:#64748B;'>Tick <b>one row</b> on each side, then click "
                "<b>✅ Confirm Match</b>.</small>",
                unsafe_allow_html=True
            )

            mm_left, mm_right = st.columns(2)
            sel_2b_idx    = None
            sel_books_idx = None

            # ── 2B side ──────────────────────────────────────────
            with mm_left:
                st.markdown('<div class="side-header-2b animate-left">📘 Open in 2B</div>', unsafe_allow_html=True)

                if open_2b_rows.empty:
                    st.info("No 'Open in 2B' rows in filtered view.")
                else:
                    for df_idx, row in open_2b_rows.iterrows():
                        gstin = str(row.get("Supplier GSTIN",  "—"))
                        doc   = str(row.get("Document Number", "—"))
                        igst  = fmt_amt(row.get("IGST Amount_2B", 0))
                        cgst  = fmt_amt(row.get("CGST Amount_2B", 0))
                        sgst  = fmt_amt(row.get("SGST Amount_2B", 0))

                        chk_col, info_col = st.columns([0.07, 0.93])
                        with chk_col:
                            checked = st.checkbox("", key=f"chk_2b_{df_idx}", label_visibility="collapsed")
                        with info_col:
                            st.markdown(
                                f"<div class='mm-row-card'>"
                                f"<b>GSTIN :</b> {gstin}<br>"
                                f"<b>Doc No:</b> {doc}<br>"
                                f"<b>IGST:</b> {igst} &nbsp;|&nbsp; "
                                f"<b>CGST:</b> {cgst} &nbsp;|&nbsp; "
                                f"<b>SGST:</b> {sgst}"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        if checked:
                            sel_2b_idx = df_idx

            # ── Books side ───────────────────────────────────────
            with mm_right:
                st.markdown('<div class="side-header-pur animate-right">📙 Open in Books</div>', unsafe_allow_html=True)

                if open_books_rows.empty:
                    st.info("No 'Open in Books' rows in filtered view.")
                else:
                    for df_idx, row in open_books_rows.iterrows():
                        gstin = str(row.get("Vendor/Customer GSTIN",   "—"))
                        doc   = str(row.get("Reference Document No.", "—"))
                        igst  = fmt_amt(row.get("IGST Amount_PUR", 0))
                        cgst  = fmt_amt(row.get("CGST Amount_PUR", 0))
                        sgst  = fmt_amt(row.get("SGST Amount_PUR", 0))

                        chk_col, info_col = st.columns([0.07, 0.93])
                        with chk_col:
                            checked = st.checkbox("", key=f"chk_bk_{df_idx}", label_visibility="collapsed")
                        with info_col:
                            st.markdown(
                                f"<div class='mm-row-card' style='border-left-color: #F97316;'>"
                                f"<b>GSTIN :</b> {gstin}<br>"
                                f"<b>Doc No:</b> {doc}<br>"
                                f"<b>IGST:</b> {igst} &nbsp;|&nbsp; "
                                f"<b>CGST:</b> {cgst} &nbsp;|&nbsp; "
                                f"<b>SGST:</b> {sgst}"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        if checked:
                            sel_books_idx = df_idx

            # ── Confirm & Undo buttons ────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Split into two columns for Confirm and Undo
            _, btn_col1, btn_col2, _ = st.columns([1, 1.5, 1.5, 1])
            
            with btn_col1:
                confirm_btn = st.button("✅ Confirm Match", use_container_width=True, key="confirm_manual")
                
            with btn_col2:
                # Check if there's anything to undo
                num_matches = len(st.session_state["manual_matches"])
                undo_btn = st.button(f"⏪ Undo Last Match ({num_matches})", 
                                     use_container_width=True, 
                                     disabled=(num_matches == 0))

            # --- UNDO LOGIC ---
            if undo_btn:
                # Pop the last saved state
                last_match = st.session_state["manual_matches"].pop()
                live_df = st.session_state["result_df"].copy()
                
                # Restore the rows to exactly how they were before the match
                live_df.loc[last_match["idx_2b"]] = last_match["old_row_2b"]
                live_df.loc[last_match["idx_books"]] = last_match["old_row_books"]
                
                st.session_state["result_df"] = live_df
                st.success("⏪ Last manual match was successfully undone!")
                time.sleep(1)
                st.rerun()

            # --- CONFIRM LOGIC ---
            if confirm_btn:
                if sel_2b_idx is None or sel_books_idx is None:
                    st.warning("⚠️ Please select exactly one row from each side before confirming.")
                else:
                    live_df = st.session_state["result_df"].copy()

                    # 1️⃣ CAPTURE SNAPSHOT BEFORE MODIFYING (For Undo)
                    old_row_2b = live_df.loc[sel_2b_idx].copy()
                    old_row_books = live_df.loc[sel_books_idx].copy()

                    # 2️⃣ COPY PURCHASE DATA
                    pur_copy_cols = [
                        c for c in live_df.columns
                        if c.endswith("_PUR") or c in [
                            "Reference Document No.", "FI Document Number",
                            "Vendor/Customer Name",  "Vendor/Customer GSTIN"
                        ]
                    ]
                    for col in pur_copy_cols:
                        if col in live_df.columns:
                            live_df.at[sel_2b_idx, col] = live_df.at[sel_books_idx, col]

                    # 3️⃣ RECALCULATE TAX DIFFS
                    for tax in ["IGST", "CGST", "SGST"]:
                        p_col = f"{tax} Amount_PUR"
                        b_col = f"{tax} Amount_2B"
                        d_col = f"{tax} Diff"
                        if p_col in live_df.columns and b_col in live_df.columns:
                            live_df.at[sel_2b_idx, d_col] = (
                                pd.to_numeric(live_df.at[sel_2b_idx, p_col], errors="coerce") -
                                pd.to_numeric(live_df.at[sel_2b_idx, b_col], errors="coerce")
                            )

                    # 4️⃣ UPDATE STATUS
                    live_df.at[sel_2b_idx,    "Match_Status"] = "Manual Match"
                    live_df.at[sel_books_idx, "Match_Status"] = "Manual Match (Consumed)"

                    # 5️⃣ SAVE TO SESSION STATE
                    st.session_state["result_df"] = live_df
                    
                    # Store as a dictionary so we know exactly what to revert
                    st.session_state["manual_matches"].append({
                        "idx_2b": sel_2b_idx,
                        "old_row_2b": old_row_2b,
                        "idx_books": sel_books_idx,
                        "old_row_books": old_row_books
                    })
                    
                    st.success("✅ Rows matched and marked as **Manual Match**!")
                    time.sleep(1)
                    st.rerun()
                    
    st.markdown('</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    #  DETAILED LEDGER & EXPORT (Animated Box Area)
    # ════════════════════════════════════════════════════════
    st.markdown("<hr style='margin-top: 3rem;'>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="ledger-container animate-fade" style="animation-delay: 0.5s;">
            <h3 style="margin-top: 0;"><span class="floating-anim" style="display:inline-block;">📋</span> Master Reconciliation Ledger</h3>
            <p style="color:#64748B; font-size: 0.9rem; margin-bottom: 1.5rem;">Below is the complete overview of all records, including system logic output and any manual matches you've processed.</p>
    """, unsafe_allow_html=True)
    
    st.dataframe(
        result_df.style.map(
            lambda x: "background-color: #FFEDD5" if x in ["Mismatch", MATCH_OPEN_2B, MATCH_OPEN_BOOKS] 
                      else ("background-color: #DCFCE7" if x == "Manual Match" else ""), 
            subset=["Match_Status"]
        ),
        use_container_width=True, 
        height=350
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        result_df.to_excel(writer, index=False)
    
    _, dl_col, _ = st.columns([1, 2, 1])
    with dl_col:
        st.download_button(
            label="📥 DOWNLOAD FINAL REPORT (EXCEL)",
            data=output.getvalue(),
            file_name="GST_Reco_Smart_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    st.markdown('</div>', unsafe_allow_html=True) # Closes ledger-container

elif not gst_file or not pur_file:
    st.markdown("""
        <div class="empty-state animate-fade">
            <div class="floating-anim" style="font-size: 3rem; margin-bottom: 10px;">🚀</div>
            <h2 style="margin-bottom: 10px;">Awaiting Data Injection</h2>
            <p>Upload your <b>GSTR-2B</b> and <b>Purchase Register</b> files above to trigger the reconciliation engine.</p>
        </div>
    """, unsafe_allow_html=True)
