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
        background-color: #F4F7FE !important; /* Very soft blue-grey */
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
        color: #1E3A8A; /* Deep Navy Blue */
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    .hero-highlight {
        color: #EA580C; /* Vibrant Orange */
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #64748B;
        font-weight: 500;
    }
    [data-testid="stFileUploadDropzone"] button {
        position: relative !important;
        color: transparent !important; 
        background-color: #FFFFFF !important;
        border: 2px solid #2563EB !important;
        border-radius: 8px !important;
        min-width: 150px !important;
        height: 42px !important;
        overflow: hidden !important;
    }

    /* 2. Nuke the hidden Streamlit SVG icon */
    [data-testid="stFileUploadDropzone"] button svg {
        display: none !important;
    }

    /* 3. Inject our own clean, perfectly aligned text */
    [data-testid="stFileUploadDropzone"] button::after {
        content: "📁 Browse Files" !important;
        position: absolute !important;
        color: #2563EB !important; /* Navy/Royal Blue */
        font-weight: 600 !important;
        font-size: 14px !important;
        font-family: 'Inter', sans-serif !important;
        left: 50% !important;
        top: 50% !important;
        transform: translate(-50%, -50%) !important;
        width: 100% !important;
        text-align: center !important;
        pointer-events: none !important; /* Ensures the button is still clickable */
    }
    
    /* 4. Add a nice hover effect to the new button */
    [data-testid="stFileUploadDropzone"] button:hover {
        background-color: #EFF6FF !important;
    }
    /* Flexbox Card Container for Perfect Alignment */
    .metric-row {
        display: flex;
        gap: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Individual Metric Cards */
    .stat-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        flex: 1;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border-top: 5px solid #2563EB; /* Default Blue Top */
        transition: transform 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* Specific Card Border Colors */
    .border-orange { border-top-color: #EA580C !important; }
    .border-navy { border-top-color: #1E3A8A !important; }
    .border-light-blue { border-top-color: #38BDF8 !important; }

    /* Card Text Alignment */
    .stat-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
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

    /* Primary Action Button (Orange) */
    div.stButton > button {
        background-color: #EA580C !important; 
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: background-color 0.2s ease !important;
    }
    div.stButton > button:hover {
        background-color: #C2410C !important; /* Darker Orange on hover */
    }

    /* Secondary Action Button (Blue - Download) */
    div.stDownloadButton > button {
        background-color: #2563EB !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
    }
    div.stDownloadButton > button:hover {
        background-color: #1D4ED8 !important; /* Darker Blue on hover */
    }

    /* File Uploader styling */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #93C5FD !important; /* Light blue dashed border */
        background-color: #EFF6FF !important; /* Very light blue background */
        border-radius: 12px !important;
    }
    /* Fix for the overlapping "Upload" text */
    [data-testid="stFileUploadDropzone"] div {
        align-items: center !important;
    }
    
    /* Resets the button padding so text has room to breathe */
    [data-testid="stFileUploadDropzone"] button {
        width: auto !important;
        padding: 0.5rem 1rem !important;
        display: inline-flex !important;
        justify-content: center !important;
    }
    
    /* Hides the default Streamlit SVG icon to stop the text bleeding */
    [data-testid="stFileUploadDropzone"] svg {
        display: none !important; 
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
    gst_file = st.file_uploader("Upload GSTR-2B Portal Data", type=["xlsx"])

with col2:
    pur_file = st.file_uploader("Upload ERP Purchase Register", type=["xlsx"])

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- MAIN LOGIC ----------------
if gst_file and pur_file:
    try:
        df_2b = pd.read_excel(gst_file)
        df_books = pd.read_excel(pur_file)

        df_2b.columns = df_2b.columns.str.strip()
        df_books.columns = df_books.columns.str.strip()
        
        # Centered Execution Button
        _, center_btn, _ = st.columns([1, 2, 1])
        with center_btn:
            run_btn = st.button("▶ Run Reconciliation Process")

        if run_btn:
            with st.spinner("Processing data arrays..."):
                # Call your logic
                result_df = reco_logic.process_reco(df_2b, df_books)

            # ---------------- METRICS DASHBOARD ----------------
            st.markdown('<br><div class="section-header">2. Reconciliation Summary</div>', unsafe_allow_html=True)
            
            total = len(result_df)
            matched = result_df["Match_Status"].str.contains("Match", case=False, na=False).sum()
            unmatched = total - matched
            match_pct = (matched / total) * 100 if total > 0 else 0

            # Perfectly aligned flexbox cards
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

            # ---------------- DATA TABLE ----------------
            st.markdown('<div class="section-header">3. Detailed Audit Matrix</div>', unsafe_allow_html=True)
            st.dataframe(result_df, use_container_width=True, height=400)

            # ---------------- EXPORT ----------------
            st.markdown('<br><div class="section-header">4. Export Results</div>', unsafe_allow_html=True)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                result_df.to_excel(writer, index=False)

            # Using columns to align the download button nicely
            dl_col1, dl_col2 = st.columns([1, 3])
            with dl_col1:
                st.download_button(
                    "📥 Download Excel File",
                    data=output.getvalue(),
                    file_name="GST_Reconciliation_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

    except Exception as e:
        st.error(f"Error Processing Files: {str(e)}")

else:
    # Clean Empty State
    st.info("ℹ️ Please upload both the GSTR-2B and Purchase Register files above to begin the audit.")
