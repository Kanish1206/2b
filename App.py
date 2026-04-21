import streamlit as st
import pandas as pd
import io
import reconciliation_logic as reco_logic

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="GST Audit Engine | Premium",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- ULTRA-PREMIUM DARK CSS ----------------
st.markdown("""
<style>
    /* Premium Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    /* Core App Background */
    html, body, [class*="st-"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #09090B !important; /* Deepest black/zinc */
        color: #FAFAFA !important;
    }

    /* Hide default streamlit elements */
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Typography Classes */
    .gradient-text {
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        letter-spacing: -1.5px;
        margin-bottom: 0px;
    }
    .sub-heading {
        color: #A1A1AA;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* Bento Box Grid System */
    .bento-card {
        background: rgba(24, 24, 27, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 24px;
        transition: all 0.3s ease;
        height: 100%;
    }
    .bento-card:hover {
        border: 1px solid rgba(255, 255, 255, 0.15);
        transform: translateY(-4px);
        box-shadow: 0 20px 40px -15px rgba(0,0,0,0.5);
    }

    /* Metric Styling inside Bento */
    .metric-label {
        font-size: 0.9rem;
        color: #A1A1AA;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.2;
    }
    
    /* Status Colors */
    .text-success { color: #34D399; }
    .text-danger { color: #FB7185; }
    .text-info { color: #38BDF8; }

    /* Custom File Uploader */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(24, 24, 27, 0.4) !important;
        border: 1px dashed rgba(255, 255, 255, 0.2) !important;
        border-radius: 16px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #38BDF8 !important;
        background: rgba(56, 189, 248, 0.05) !important;
    }

    /* The 'Execute' Button */
    div.stButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 0rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        width: 100% !important;
        box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.39) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.6) !important;
    }

    /* Download Button */
    div.stDownloadButton > button {
        background: rgba(24, 24, 27, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 12px !important;
        font-weight: 500 !important;
    }
    div.stDownloadButton > button:hover {
        border-color: #34D399 !important;
        color: #34D399 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div>
    <h1 class="gradient-text">Nexus Recon Engine</h1>
    <p class="sub-heading">Intelligent GSTR-2B & Purchase Ledger synchronization protocol.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- FILE UPLOAD (BENTO ROW 1) ----------------
st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
        <div style="margin-bottom: -40px; z-index: 10; position: relative; padding-left: 10px;">
            <span style="color: #38BDF8; font-weight: 600; font-size: 0.9rem;">NODE 01</span>
        </div>
    """, unsafe_allow_html=True)
    gst_file = st.file_uploader("GSTR-2B Portal Extract", type=["xlsx"])

with col2:
    st.markdown("""
        <div style="margin-bottom: -40px; z-index: 10; position: relative; padding-left: 10px;">
            <span style="color: #818CF8; font-weight: 600; font-size: 0.9rem;">NODE 02</span>
        </div>
    """, unsafe_allow_html=True)
    pur_file = st.file_uploader("ERP Purchase Register", type=["xlsx"])

# ---------------- MAIN LOGIC ----------------
st.markdown("<br><br>", unsafe_allow_html=True)

if gst_file and pur_file:
    try:
        df_2b = pd.read_excel(gst_file)
        df_books = pd.read_excel(pur_file)

        df_2b.columns = df_2b.columns.str.strip()
        df_books.columns = df_books.columns.str.strip()
        
        # Action Center
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            run_btn = st.button("⚡ Initialize Audit Sequence")

        if run_btn:
            with st.spinner("Synchronizing datasets & mapping tax parameters..."):
                # Call your logic
                result_df = reco_logic.process_reco(df_2b, df_books)

            # ---------------- BENTO METRICS ROW ----------------
            total = len(result_df)
            matched = result_df["Match_Status"].str.contains("Match", case=False, na=False).sum()
            unmatched = total - matched
            match_pct = (matched / total) * 100 if total > 0 else 0

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)

            with m1:
                st.markdown(f"""
                <div class="bento-card">
                    <div class="metric-label">Processed Vectors</div>
                    <div class="metric-value">{total:,}</div>
                </div>
                """, unsafe_allow_html=True)

            with m2:
                st.markdown(f"""
                <div class="bento-card" style="box-shadow: inset 0 2px 0 0 rgba(52, 211, 153, 0.5);">
                    <div class="metric-label">Perfect Matches</div>
                    <div class="metric-value text-success">{matched:,}</div>
                </div>
                """, unsafe_allow_html=True)

            with m3:
                st.markdown(f"""
                <div class="bento-card" style="box-shadow: inset 0 2px 0 0 rgba(251, 113, 133, 0.5);">
                    <div class="metric-label">Anomalies Detected</div>
                    <div class="metric-value text-danger">{unmatched:,}</div>
                </div>
                """, unsafe_allow_html=True)

            with m4:
                st.markdown(f"""
                <div class="bento-card" style="box-shadow: inset 0 2px 0 0 rgba(56, 189, 248, 0.5);">
                    <div class="metric-label">System Confidence</div>
                    <div class="metric-value text-info">{match_pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # ---------------- DATA TABLE ROW ----------------
            st.markdown("<br><br><div class='sub-heading' style='color:#FFFFFF; font-weight:600;'>Audit Log Output</div>", unsafe_allow_html=True)
            
            # Use Streamlit's native dataframe which auto-adapts to dark mode nicely
            st.dataframe(
                result_df, 
                use_container_width=True, 
                height=400,
                hide_index=True
            )

            # ---------------- EXPORT ROW ----------------
            st.markdown("<br>", unsafe_allow_html=True)
            _, exp_col = st.columns([3, 1])
            
            with exp_col:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    result_df.to_excel(writer, index=False)

                st.download_button(
                    "⭳ Export Final Matrix (.xlsx)",
                    data=output.getvalue(),
                    file_name="Nexus_Audit_Matrix.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

    except Exception as e:
        st.error(f"System Fault Detected: {str(e)}")

else:
    # High-end Empty State
    st.markdown("""
        <div style="text-align: center; padding: 5rem 2rem; background: rgba(24, 24, 27, 0.4); border-radius: 24px; border: 1px dashed rgba(255, 255, 255, 0.1); margin-top: 3rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.5;">🛰️</div>
            <h3 style="color: #FAFAFA; font-weight: 600;">System Standby</h3>
            <p style="color: #A1A1AA; font-size: 1.1rem;">Awaiting telemetry data. Upload both ledgers to initialize the array.</p>
        </div>
    """, unsafe_allow_html=True)
