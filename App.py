import streamlit as st
import pandas as pd
import io
import reconciliation_logic as reco_logic

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="GST Reco Pro",
    page_icon="🧿",
    layout="wide"
)

# ---------------- BLUE & ORANGE THEME CSS ----------------
st.markdown("""
<style>
    /* Global Font & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
        background-color: #F4F7FE !important;
    }

    /* Clean Streamlit elements */
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Centered Hero Section */
    .hero-box {
        text-align: center;
        padding: 2rem 1rem 3rem 1rem;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    .hero-highlight {
        color: #EA580C;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #64748B;
        font-weight: 500;
    }

    /* --- 1. FIXED FILE UPLOADER CSS --- */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #93C5FD !important;
        background-color: #EFF6FF !important;
        border-radius: 12px !important;
        padding: 2rem !important;
    }

    /* Nuke all native Streamlit content inside the upload button */
    [data-testid="stFileUploadDropzone"] button * {
        display: none !important;
        opacity: 0 !important;
        visibility: hidden !important;
    }

    /* Base upload button styling */
    [data-testid="stFileUploadDropzone"] button {
        background-color: #FFFFFF !important;
        border: 2px solid #2563EB !important;
        border-radius: 8px !important;
        position: relative;
        padding: 0.5rem 2rem !important;
        min-width: 180px !important;
        min-height: 42px !important;
    }

    /* Inject the custom upload text securely */
    [data-testid="stFileUploadDropzone"] button::after {
        content: "📁 Browse Files" !important;
        position: absolute;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        left: 0;
        right: 0;
        top: 50%;
        transform: translateY(-50%);
        color: #2563EB !important;
        font-weight: 600;
        font-size: 14px;
        text-align: center;
    }

    /* Hide the 'drag and drop' text */
    [data-testid="stFileUploadDropzone"] section > div > div > span {
        display: none !important;
    }

    /* --- 2. THE "NUCLEAR OPTION" FOR RUN & DOWNLOAD BUTTONS --- */
    
    /* 🔴 RUN BUTTON (Solid Orange) */
    [data-testid="stButton"] button {
        background: #EA580C !important;
        background-color: #EA580C !important; 
        border: 2px solid #EA580C !important;
        border-radius: 8px !important;
        width: 100% !important;
        padding: 0.6rem !important;
        transition: all 0.2s ease-in-out;
    }
    
    /* Force ALL text inside the Run button to be white */
    [data-testid="stButton"] button p, 
    [data-testid="stButton"] button span, 
    [data-testid="stButton"] button div {
        color: #FFFFFF !important; 
        font-weight: 800 !important;
        font-size: 16px !important;
    }

    [data-testid="stButton"] button:hover {
        background-color: #C2410C !important;
        border-color: #C2410C !important;
        box-shadow: 0 4px 10px rgba(234, 88, 12, 0.4) !important;
    }

    /* 🟢 DOWNLOAD BUTTON (Solid Green) */
    [data-testid="stDownloadButton"] button {
        background: #10B981 !important;
        background-color: #10B981 !important; 
        border: 2px solid #10B981 !important;
        border-radius: 8px !important;
        width: 100% !important;
        padding: 0.6rem !important;
        transition: all 0.2s ease-in-out;
    }
    
    /* Force ALL text inside the Download button to be white */
    [data-testid="stDownloadButton"] button p, 
    [data-testid="stDownloadButton"] button span, 
    [data-testid="stDownloadButton"] button div {
        color: #FFFFFF !important; 
        font-weight: 800 !important;
        font-size: 16px !important;
    }

    [data-testid="stDownloadButton"] button:hover {
        background-color: #059669 !important;
        border-color: #059669 !important;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.4) !important;
    }

    /* Metric Cards */
    .metric-row {
        display: flex;
        gap: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        flex: 1;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border-top: 5px solid #2563EB;
        transition: transform 0.2s ease;
    }
    .border-orange { border-top-color: #EA580C !important; }
    .border-navy { border-top-color: #1E3A8A !important; }
    .border-light-blue { border-top-color: #38BDF8 !important; }

    .stat-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .stat-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
    }
    .text-orange { color: #EA580C !important; }
    .text-blue { color: #2563EB !important; }

    /* Section Headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HERO SECTION ----------------
st.markdown("""
<div class="hero-box">
    <div class="hero-title">GST Reco <span class="hero-highlight">Pro</span></div>
    <div class="hero-subtitle">Seamless GSTR-2B & Purchase Register Synchronization</div>
</div>
""", unsafe_allow_html=True)

# ---------------- FILE UPLOAD SECTION ----------------
st.markdown('<div class="section-header">1. Upload Datasets</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    gst_file = st.file_uploader("Upload GSTR-2B Portal Data", type=["xlsx"], key="gst_upload")

with col2:
    pur_file = st.file_uploader("Upload ERP Purchase Register", type=["xlsx"], key="pur_upload")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- MAIN LOGIC ----------------
if gst_file and pur_file:
    try:
        df_2b = pd.read_excel(gst_file)
        df_books = pd.read_excel(pur_file)

        df_2b.columns = df_2b.columns.str.strip()
        df_books.columns = df_books.columns.str.strip()
        
        _, center_btn, _ = st.columns([1, 2, 1])
        with center_btn:
            run_btn = st.button("▶ Run Reconciliation Process")

        if run_btn:
            with st.spinner("Processing data arrays..."):
                result_df = reco_logic.process_reco(df_2b, df_books)

            # ---------------- METRICS DASHBOARD ----------------
            st.markdown('<br><div class="section-header">2. Reconciliation Summary</div>', unsafe_allow_html=True)
            
            total = len(result_df)
            matched = result_df["Match_Status"].str.contains("Match", case=False, na=False).sum()
            unmatched = total - matched
            match_pct = (matched / total) * 100 if total > 0 else 0

            st.markdown(f"""
            <div class="metric-row">
                <div class="stat-card border-navy">
                    <div class="stat-label">Total Records</div>
                    <div class="stat-value">{total:,}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Matched Invoices</div>
                    <div class="stat-value text-blue">{matched:,}</div>
                </div>
                <div class="stat-card border-orange">
                    <div class="stat-label">Discrepancies</div>
                    <div class="stat-value text-orange">{unmatched:,}</div>
                </div>
                <div class="stat-card border-light-blue">
                    <div class="stat-label">Accuracy Rate</div>
                    <div class="stat-value">{match_pct:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-header">3. Detailed Audit Matrix</div>', unsafe_allow_html=True)
            st.dataframe(result_df, use_container_width=True, height=400)

            st.markdown('<br><div class="section-header">4. Export Results</div>', unsafe_allow_html=True)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                result_df.to_excel(writer, index=False)

            dl_col1, _ = st.columns([1, 3])
            with dl_col1:
                st.download_button(
                    label="📥 Download Excel File",
                    data=output.getvalue(),
                    file_name="GST_Reconciliation_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

    except Exception as e:
        st.error(f"Error Processing Files: {str(e)}")
else:
    st.info("ℹ️ Please upload both the GSTR-2B and Purchase Register files above to begin the audit.")
