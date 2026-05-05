import streamlit as st
import pandas as pd
import io
import time
import reconciliation_logic as reco_logic

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="GST Reco Pro | Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- ULTRA-MODERN CSS INJECTION ----------------
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #F0F4F8; }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulseButton {
        0% { box-shadow: 0 0 0 0 rgba(249, 115, 22, 0.7); transform: scale(1); }
        50% { box-shadow: 0 0 0 15px rgba(249, 115, 22, 0); transform: scale(1.02); }
        100% { box-shadow: 0 0 0 0 rgba(249, 115, 22, 0); transform: scale(1); }
    }

    @keyframes floatIcon {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    /* Animated Gradient Background for Hero */
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .animate-fade { animation: fadeIn 0.6s ease-out forwards; }
    .floating-icon { display: inline-block; animation: floatIcon 3s ease-in-out infinite; }

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
        position: relative;
        overflow: hidden;
    }
    
    /* Custom KPI Cards */
    .kpi-container {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        flex: 1;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-bottom: 4px solid #3B82F6;
        text-align: center;
        transition: transform 0.3s ease;
        animation: fadeIn 0.8s ease-out forwards;
    }
    .kpi-card:hover { transform: translateY(-5px); }
    .kpi-card.orange { border-bottom: 4px solid #F97316; }
    .kpi-value { font-size: 2.2rem; font-weight: 800; color: #0F172A; margin: 0.5rem 0; }
    .kpi-label { font-size: 0.9rem; color: #64748B; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }

    /* Modern Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #F97316 0%, #EA580C 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 50px; /* Pill shape */
        font-weight: bold;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        animation: pulseButton 2s infinite; 
    }
    .stButton>button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 6px 20px rgba(249, 115, 22, 0.5);
        color: white;
        animation: none; 
    }
    
    /* Empty State */
    .empty-state {
        background: white;
        padding: 4rem 2rem;
        text-align: center;
        border-radius: 16px;
        border: 2px dashed #CBD5E1;
        color: #64748B;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER SECTION ----------------
st.markdown("""
    <div class="hero-header">
        <h1 style='margin:0; font-size: 3rem; font-weight: 800; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
            <span class="floating-icon">⚡</span> GST Intelligence Hub
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

# ---------------- MAIN PROCESSOR ----------------
if gst_file and pur_file:
    try:
        df_2b = pd.read_excel(gst_file)
        df_books = pd.read_excel(pur_file)
        
        df_2b.columns = df_2b.columns.str.strip()
        df_books.columns = df_books.columns.str.strip()

        # Center the button
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            run_btn = st.button("🚀 INITIATE PROCESS", use_container_width=True)

        if run_btn:
            # --- DYNAMIC PROCESSING ANIMATION ---
            with st.status("⚡ Initiating Intelligence Engine...", expanded=True) as status:
                st.write("📥 Ingesting GSTR-2B and Purchase Data...")
                time.sleep(0.5)  # Visual pause for effect
                
                st.write("🔍 Running Fuzzy Logic & Exact Match Algorithms...")
                # The actual heavy lifting
                result_df = reco_logic.process_reco(df_2b, df_books)
                
                st.write("📊 Finalizing Discrepancy Analytics...")
                time.sleep(0.5)  # Visual pause for effect
                
                status.update(label="✅ Reconciliation Complete!", state="complete", expanded=False)
            
            st.balloons() 

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
                        <div class="kpi-label">Discrepancies</div>
                        <div class="kpi-value" style="color: #F97316;">{unmatched:,}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # --- DETAILED LEDGER (Direct View) ---
            st.markdown("### 📋 Detailed ")
            
            st.dataframe(
                result_df.style.map(
                    lambda x: "background-color: #FFEDD5" if x == "Mismatch" else "", 
                    subset=["Match_Status"]
                ),
                use_container_width=True, 
                height=400
            )

            # --- EXPORT SECTION ---
            st.markdown("<br>", unsafe_allow_html=True)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                result_df.to_excel(writer, index=False)
            
            st.download_button(
                label="📥 DOWNLOAD FINAL REPORT (EXCEL)",
                data=output.getvalue(),
                file_name="GST_Reco_Smart_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"🚨 Process Error: {str(e)}")

else:
    st.markdown("""
        <div class="empty-state animate-fade">
            <h2 style="margin-bottom: 10px;"><span class="floating-icon">🚀</span> Awaiting Data Injection</h2>
            <p>Upload your <b>GSTR-2B</b> and <b>Purchase Register</b> files above to trigger the reconciliation engine.</p>
        </div>
    """, unsafe_allow_html=True)
