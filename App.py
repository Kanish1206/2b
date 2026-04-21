import streamlit as st
import pandas as pd
import io
import reconciliation_logic as reco_logic

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="GST Reco Pro | Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM STYLING (The "Modern" Look) ----------------
st.markdown("""
    <style>
    /* Main Background and Font */
    .stApp {
        background-color: #f8f9fb;
    }
    
    /* Custom Header Gradient */
    .header-container {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 50%, #F59E0B 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    /* Metric Cards Styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    
    /* Custom Card Containers */
    .reco-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #F59E0B;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }

    /* Modern Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);
        color: white;
    }
    
    /* Status indicators */
    .status-box {
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER SECTION ----------------
st.markdown("""
    <div class="header-container">
        <h1 style='margin:0; font-size: 2.5rem;'>⚡ GST Reco Pro</h1>
        <p style='margin:0; opacity: 0.9;'>Modern Purchase Register vs GSTR-2B Intelligence</p>
    </div>
""", unsafe_allow_html=True)

# ---------------- UPLOAD ZONE ----------------
with st.container():
    st.markdown('<div class="reco-card"><h3>📂 Data Ingestion</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        gst_file = st.file_uploader("Upload GSTR-2B (Excel)", type=["xlsx"], key="gst")
    with col2:
        pur_file = st.file_uploader("Purchase Register (Excel)", type=["xlsx"], key="pur")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- MAIN PROCESSOR ----------------
if gst_file and pur_file:
    try:
        df_2b = pd.read_excel(gst_file)
        df_books = pd.read_excel(pur_file)
        
        # Simple cleanup
        df_2b.columns = df_2b.columns.str.strip()
        df_books.columns = df_books.columns.str.strip()

        # Action Area
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀 INITIATE RECONCILIATION", use_container_width=True)

        if run_btn:
            with st.spinner("🧠 Analyzing discrepancies..."):
                result_df = reco_logic.process_reco(df_2b, df_books)

            # --- ANALYTICS DASHBOARD ---
            st.markdown("### 📊 Executive Summary")
            
            total = len(result_df)
            matched = result_df["Match_Status"].str.contains("Match", case=False, na=False).sum()
            unmatched = total - matched
            match_rate = (matched / total) * 100 if total > 0 else 0

            # Metric Tiles
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Records", total)
            m2.metric("Fully Matched", matched, f"{match_rate:.1f}%")
            m3.metric("Mismatches", unmatched, f"-{100-match_rate:.1f}%", delta_color="inverse")
            m4.metric("Risk Score", "Low" if match_rate > 90 else "Medium")

            # --- DATA VIEW ---
            st.markdown('<div class="reco-card">', unsafe_allow_html=True)
            st.subheader("📋 Reconciliation Ledger")
            
            # Style the dataframe (Blue headers)
            st.dataframe(
                result_df.style.set_properties(**{'background-color': '#ffffff', 'color': '#1E3A8A'})
                .highlight_null(color='#f8d7da'),
                use_container_width=True,
                height=400
            )
            st.markdown('</div>', unsafe_allow_html=True)

            # --- EXPORT SECTION ---
            st.markdown("### 📥 Export Result")
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                result_df.to_excel(writer, index=False)
            
            st.download_button(
                label="✨ DOWNLOAD RECONCILIATION REPORT (EXCEL)",
                data=output.getvalue(),
                file_name="GST_Reco_Smart_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"🚨 Process Error: {str(e)}")

else:
    # Modern Empty State
    st.markdown("""
        <div style="text-align: center; padding: 50px; border: 2px dashed #3B82F6; border-radius: 15px; color: #1E3A8A;">
            <h3>Waiting for Data Input...</h3>
            <p>Please upload both excel files in the section above to begin the automated audit.</p>
        </div>
    """, unsafe_allow_html=True)
