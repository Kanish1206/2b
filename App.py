import streamlit as st
import pandas as pd
import io
import plotly.express as px
import reconciliation_logic as reco_logic

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="GST Reco Pro | Dashboard",
    page_icon="⚡",
    layout="wide"
)

# ---------------- PREMIUM CSS ----------------
st.markdown("""
<style>
    /* Main background and font */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Custom Card Design */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Header Styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    
    .status-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        border-left: 5px solid #3b82f6;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
col_header, col_logo = st.columns([4, 1])
with col_header:
    st.markdown('<p class="main-title">⚡ GST Reco Pro</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; font-size: 1.2rem;">Automated Smart Reconciliation Engine</p>', unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2621/2621846.png", width=80)
    st.title("Control Center")
    st.divider()
    
    filter_option = st.selectbox(
        "🎯 Filter View",
        ["All Records", "Perfect Match", "Mismatched / Missing"]
    )
    
    search_text = st.text_input("🔍 Quick Search", placeholder="GSTIN or Invoice No...")
    
    st.divider()
    st.caption("Developed for Omkar Enterprises v2.0")

# ---------------- FILE UPLOAD (Clean UI) ----------------
st.write("### 📂 Data Import")
with st.container():
    c1, c2 = st.columns(2)
    with c1:
        gst_file = st.file_uploader("📥 GSTR-2B (Excel)", type=["xlsx"], help="Upload the portal generated 2B")
    with c2:
        pur_file = st.file_uploader("📥 Purchase Register (Excel)", type=["xlsx"], help="Upload your Tally/ERP export")

# ---------------- CACHE & UTILS ----------------
@st.cache_data
def load_file(file):
    df = pd.read_excel(file)
    df.columns = df.columns.str.strip()
    return df

def explain_mismatch(row):
    reasons = []
    try:
        if pd.notna(row.get("Invoice_Value_2B")) and pd.notna(row.get("Invoice_Value_Books")):
            diff = abs(row["Invoice_Value_2B"] - row["Invoice_Value_Books"])
            if diff > 1:
                reasons.append(f"Value Diff: ₹{round(diff,2)}")

        if str(row.get("GSTIN_x")) != str(row.get("GSTIN_y")) and pd.notna(row.get("GSTIN_y")):
            reasons.append("GSTIN Mismatch")

        if pd.isna(row.get("Invoice_No")):
            reasons.append("Missing in Books")
        
        return " | ".join(reasons) if reasons else "Matched"
    except:
        return "Analysis Error"

# ---------------- PROCESSING ----------------
if gst_file and pur_file:
    try:
        df_2b = load_file(gst_file)
        df_books = load_file(pur_file)
        
        st.markdown("---")
        if st.button("🚀 Execute Smart Reconciliation", use_container_width=True, type="primary"):
            
            with st.spinner("Processing large datasets..."):
                result_df = reco_logic.process_reco(df_2b, df_books)
                result_df["Explanation"] = result_df.apply(explain_mismatch, axis=1)

            # --- KPI METRICS ---
            total = len(result_df)
            matched = result_df["Explanation"].str.contains("Matched").sum()
            unmatched = total - matched
            accuracy = (matched / total) if total else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Invoices", f"{total:,}")
            m2.metric("Matched", f"{matched:,}", f"{accuracy:.1%}")
            m3.metric("Mismatched", f"{unmatched:,}", f"-{1-accuracy:.1%}", delta_color="inverse")
            m4.metric("Reco Health", f"{accuracy:.1%}")

            # --- REPLACING PIE CHART WITH HEALTH BAR ---
            st.write("### 📊 Reconciliation Health")
            health_color = "#22c55e" if accuracy > 0.8 else "#f59e0b" if accuracy > 0.5 else "#ef4444"
            st.markdown(f"""
                <div style="width: 100%; background-color: #e2e8f0; border-radius: 20px; height: 25px;">
                    <div style="width: {accuracy*100}%; background-color: {health_color}; height: 25px; border-radius: 20px; text-align: center; color: white; font-weight: bold;">
                        {accuracy:.1%} Match Rate
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # --- DATA DISTRIBUTION (BETTER THAN PIE) ---
            st.write("")
            c_left, c_right = st.columns([2, 1])
            
            with c_left:
                st.subheader("📋 Reconciliation Ledger")
                # Apply Filters
                display_df = result_df.copy()
                if filter_option == "Perfect Match":
                    display_df = display_df[display_df["Explanation"] == "Matched"]
                elif filter_option == "Mismatched / Missing":
                    display_df = display_df[display_df["Explanation"] != "Matched"]
                
                if search_text:
                    display_df = display_df[display_df.astype(str).apply(lambda x: x.str.contains(search_text, case=False)).any(axis=1)]

                st.dataframe(display_df, use_container_width=True, height=400)

            with c_right:
                st.subheader("🧾 Audit Summary")
                issue_counts = result_df[result_df["Explanation"] != "Matched"]["Explanation"].value_counts()
                if not issue_counts.empty:
                    st.write("Top Issues Found:")
                    st.bar_chart(issue_counts)
                else:
                    st.success("No critical issues found!")

            # --- EXPORT ---
            st.divider()
            st.write("### 📥 Download Reports")
            ec1, ec2 = st.columns(2)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                result_df.to_excel(writer, sheet_name="Full_Report", index=False)
            
            ec1.download_button(
                label="📁 Download Complete Excel",
                data=output.getvalue(),
                file_name="GST_Reco_Final.xlsx",
                mime="application/vnd.ms-excel",
                use_container_width=True
            )
            
            if ec2.button("📋 Copy Summary to Clipboard", use_container_width=True):
                st.toast("Summary copied! (Feature logic pending)")

    except Exception as e:
        st.error(f"Analysis Interrupted: {e}")
else:
    # Landing Page State
    st.info("💡 Please upload GSTR-2B and Purchase Register to activate the engine.")
    st.image("https://img.freepik.com/free-vector/data-extraction-concept-illustration_114360-4766.jpg", width=400)
