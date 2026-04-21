import streamlit as st
import pandas as pd
import io
import reconciliation_logic as reco_logic

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="GST Reco Pro | Advanced Audit",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- MODERN NEUMORPHIC CSS ----------------
st.markdown("""
<style>
    /* Global Background & Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
        background-color: #F8FAFC;
    }

    /* Remove Streamlit Header/Footer */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Modern Header */
    .hero-section {
        padding: 2rem 0rem;
        text-align: left;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        color: #64748B;
        font-size: 1.1rem;
    }

    /* Glassmorphism Metric Cards */
    .card-container {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    .modern-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        flex: 1;
        transition: transform 0.2s ease;
    }
    .modern-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .card-label {
        font-size: 0.875rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .card-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.5rem;
    }

    /* Custom File Uploader Styling */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #CBD5E1 !important;
        border-radius: 12px !important;
        background-color: #FFFFFF !important;
        padding: 2rem !important;
    }

    /* Modern Buttons */
    div.stButton > button {
        background-color: #2563EB !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        background-color: #1D4ED8 !important;
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3) !important;
    }

    /* Download Button - Secondary Action */
    div.stDownloadButton > button {
        background-color: #059669 !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 600 !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HERO SECTION ----------------
st.markdown("""
<div class="hero-section">
    <div class="hero-title">🛡️ GST Reco <span style='color:#2563EB'>Pro</span></div>
    <div class="hero-subtitle">High-fidelity reconciliation engine for GSTR-2B and Purchase Books.</div>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.write("Adjust matching sensitivity and rules.")
    threshold = st.slider("Matching Threshold (₹)", 0.0, 10.0, 1.0)
    st.divider()
    st.info("System Version: 2.0.4-Stable")

# ---------------- FILE UPLOAD ----------------
st.markdown("### 📂 Data Ingestion")
col1, col2 = st.columns(2)

with col1:
    gst_file = st.file_uploader("Step 1: Upload GSTR-2B (Portal Data)", type=["xlsx"])

with col2:
    pur_file = st.file_uploader("Step 2: Upload Purchase Register (ERP Data)", type=["xlsx"])

# ---------------- MAIN LOGIC ----------------
if gst_file and pur_file:
    try:
        df_2b = pd.read_excel(gst_file)
        df_books = pd.read_excel(pur_file)

        df_2b.columns = df_2b.columns.str.strip()
        df_books.columns = df_books.columns.str.strip()

        st.markdown("---")
        
        # Action Center
        c_btn1, c_btn2, c_btn3 = st.columns([1,2,1])
        with c_btn2:
            run_btn = st.button("🚀 Process Reconciliation")

        if run_btn:
            with st.spinner("Analyzing transaction patterns..."):
                # Mocking result if reco_logic isn't provided, otherwise calls it
                result_df = reco_logic.process_reco(df_2b, df_books)

            # ---------------- MODERN SUMMARY CARDS ----------------
            total = len(result_df)
            matched = result_df["Match_Status"].str.contains("Match", case=False, na=False).sum()
            unmatched = total - matched
            match_pct = (matched / total) * 100 if total > 0 else 0

            st.markdown(f"""
            <div class="card-container">
                <div class="modern-card">
                    <div class="card-label">Total Invoices</div>
                    <div class="card-value">{total:,}</div>
                </div>
                <div class="modern-card" style="border-left: 5px solid #10B981;">
                    <div class="card-label">Fully Matched</div>
                    <div class="card-value" style="color: #10B981;">{matched:,}</div>
                </div>
                <div class="modern-card" style="border-left: 5px solid #EF4444;">
                    <div class="card-label">Discrepancies</div>
                    <div class="card-value" style="color: #EF4444;">{unmatched:,}</div>
                </div>
                <div class="modern-card" style="border-left: 5px solid #2563EB;">
                    <div class="card-label">Accuracy Rate</div>
                    <div class="card-value" style="color: #2563EB;">{match_pct:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ---------------- DATA TABLE ----------------
            st.markdown("### 📋 Detailed Audit Trail")
            st.dataframe(
                result_df.style.background_gradient(subset=None, cmap='BuGn', axis=0), 
                use_container_width=True, 
                height=450
            )

            # ---------------- EXPORT ----------------
            st.markdown("### 📥 Export Result")
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                result_df.to_excel(writer, index=False)

            st.download_button(
                "⬇️ Download Reconciliation Report (.xlsx)",
                data=output.getvalue(),
                file_name="GST_Audit_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"⚠️ Critical Processing Error: {str(e)}")

else:
    # Empty State
    st.markdown("""
        <div style="text-align: center; padding: 4rem; background: #FFFFFF; border-radius: 16px; border: 1px solid #E2E8F0; margin-top: 2rem;">
            <p style="color: #64748B; font-size: 1.2rem;">Waiting for document uploads to begin analysis.</p>
        </div>
    """, unsafe_allow_html=True)
