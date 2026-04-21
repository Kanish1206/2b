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

# ---------------- BEAUTIFIED CSS ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }

    /* Clean Streamlit elements */
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Hero Section */
    .hero-container {
        padding: 3rem 1rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.3);
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #EA580C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        color: #64748B;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
    }

    /* Customizing the File Uploader to look modern */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #CBD5E1 !important;
        background: #ffffff !important;
        border-radius: 15px !important;
        padding: 2rem !important;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #2563EB !important;
        background: #F8FAFC !important;
    }

    /* Section Headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1E293B;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Metric Cards */
    .metric-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.2rem;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        border-left: 6px solid #2563EB;
        transition: transform 0.2s;
    }
    .stat-card:hover { transform: translateY(-5px); }
    .stat-card.orange { border-left-color: #EA580C; }
    .stat-card.light-blue { border-left-color: #38BDF8; }
    
    .stat-label { font-size: 0.8rem; color: #64748B; font-weight: 700; text-transform: uppercase; }
    .stat-value { font-size: 1.8rem; font-weight: 800; color: #1E293B; margin-top: 5px;}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Primary Button */
    div.stButton > button {
        border-radius: 12px !important;
        height: 3em !important;
        background: #EA580C !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 4px 14px 0 rgba(234, 88, 12, 0.39) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">GST Reco Pro</div>
    <div class="hero-subtitle">Automated Intelligent Audit & Reconciliation</div>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.markdown("### ⚙️ Action Panel")
    st.info("Upload your files in the main area to enable reconciliation.")
    run_btn = st.button("🚀 Start Reconciliation", use_container_width=True)
    st.divider()
    st.markdown("### 🛠 Support")
    st.caption("Contact help@gstreco.pro for assistance.")

# ---------------- MAIN UI ----------------
st.markdown('<div class="section-header">📂 Step 1: Data Acquisition</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    gst_file = st.file_uploader("GSTR-2B Portal Data", type=["xlsx"], help="Upload the Excel exported from GST Portal")

with col2:
    pur_file = st.file_uploader("ERP Purchase Register", type=["xlsx"], help="Upload your internal Tally/SAP/ERP register")

# ---------------- LOGIC EXECUTION ----------------
if gst_file and pur_file:
    try:
        df_2b = pd.read_excel(gst_file)
        df_books = pd.read_excel(pur_file)
        
        # Strip whitespace from columns
        df_2b.columns = df_2b.columns.str.strip()
        df_books.columns = df_books.columns.str.strip()

        if run_btn:
            with st.spinner("✨ Analyzing records and matching invoices..."):
                # Call logic from your external file
                result_df = reco_logic.process_reco(df_2b, df_books)

                # Summary Calculations
                total = len(result_df)
                matched = result_df["Match_Status"].str.contains("Match", case=False, na=False).sum()
                unmatched = total - matched
                match_pct = (matched / total) * 100 if total > 0 else 0

                # ---------------- RESULTS ----------------
                st.markdown('<div class="section-header">📊 Step 2: Insights Dashboard</div>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-row">
                    <div class="stat-card">
                        <div class="stat-label">Total Records</div>
                        <div class="stat-value">{total:,}</div>
                    </div>
                    <div class="stat-card" style="border-left-color: #10B981;">
                        <div class="stat-label">Successful Matches</div>
                        <div class="stat-value" style="color: #10B981;">{matched:,}</div>
                    </div>
                    <div class="stat-card orange">
                        <div class="stat-label">Mismatches / Missing</div>
                        <div class="stat-value" style="color: #EA580C;">{unmatched:,}</div>
                    </div>
                    <div class="stat-card light-blue">
                        <div class="stat-label">Accuracy Score</div>
                        <div class="stat-value">{match_pct:.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="section-header">🔍 Step 3: Audit Matrix</div>', unsafe_allow_html=True)
                st.dataframe(result_df, use_container_width=True, height=450)

                # Export Section
                st.markdown('<div class="section-header">💾 Step 4: Export & Finalize</div>', unsafe_allow_html=True)
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    result_df.to_excel(writer, index=False)

                st.download_button(
                    label="📥 Download Detailed Reconciliation Report",
                    data=output.getvalue(),
                    file_name="GST_Audit_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.warning("👈 Files ready. Click 'Start Reconciliation' in the sidebar to begin.")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")

else:
    # Clean Empty State
    st.container()
    st.markdown("""
        <div style="text-align: center; padding: 50px; color: #64748B;">
            <h3>Waiting for Data...</h3>
            <p>Please upload both Excel files to visualize the reconciliation matrix.</p>
        </div>
    """, unsafe_allow_html=True)
