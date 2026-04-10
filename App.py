import streamlit as st
import pandas as pd
import io
import plotly.express as px
import reconciliation_logic as reco_logic

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="GST Reco Pro",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
        .main-title {
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .sub-text {
            color: #6c757d;
            margin-bottom: 25px;
        }
        .card {
            padding: 20px;
            border-radius: 12px;
            background-color: #f8f9fa;
            margin-bottom: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">📘 GST Reconciliation Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Compare GSTR-2B with Purchase Register in seconds</div>', unsafe_allow_html=True)

# ---------------- FILE UPLOAD SECTION ----------------
with st.container():
    st.markdown("### 📂 Upload Files")

    col1, col2 = st.columns(2)

    with col1:
        gst_file = st.file_uploader("Upload GSTR-2B Excel", type=["xlsx"])

    with col2:
        pur_file = st.file_uploader("Upload Purchase Register Excel", type=["xlsx"])

# ---------------- MAIN PROCESS ----------------
if gst_file and pur_file:

    try:
        df_2b = pd.read_excel(gst_file)
        df_books = pd.read_excel(pur_file)

        df_2b.columns = df_2b.columns.str.strip()
        df_books.columns = df_books.columns.str.strip()

        st.success("✅ Files uploaded successfully")

        if st.button("🚀 Run Reconciliation", use_container_width=True):

            with st.spinner("Processing... Please wait ⏳"):
                result_df = reco_logic.process_reco(df_2b, df_books)

            # ---------------- SUMMARY ----------------
            st.markdown("## 📊 Summary")

            total = len(result_df)
            matched = result_df["Match_Status"].str.contains("Match", case=False, na=False).sum()
            unmatched = total - matched

            c1, c2, c3 = st.columns(3)

            c1.metric("📄 Total Records", total)
            c2.metric("✅ Matched", matched)
            c3.metric("❌ Unmatched", unmatched)

            # ---------------- CHART ----------------
            st.markdown("## 📈 Match Distribution")

            pie_df = result_df["Match_Status"].value_counts().reset_index()
            pie_df.columns = ["Status", "Count"]

            fig = px.pie(
                pie_df,
                names="Status",
                values="Count",
                hole=0.4,
            )

            st.plotly_chart(fig, use_container_width=True)

            # ---------------- DATA TABLE ----------------
            st.markdown("## 📋 Detailed Results")

            def highlight_status(val):
                if "Exact Match" in str(val):
                    return "background-color: #d4edda"
                elif "Mismatch" in str(val):
                    return "background-color: #fff3cd"
                elif "Open" in str(val):
                    return "background-color: #f8d7da"
                elif "Fuzzy" in str(val):
                    return "background-color: #d1ecf1"
                elif "PAN" in str(val):
                    return "background-color: #e2e3ff"
                return ""

            styled_df = result_df.style.applymap(highlight_status, subset=["Match_Status"])

            st.dataframe(styled_df, use_container_width=True)

            # ---------------- DOWNLOAD ----------------
            st.markdown("## 📥 Export Results")

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                result_df.to_excel(writer, index=False)

            st.download_button(
                "⬇️ Download Excel Report",
                data=output.getvalue(),
                file_name="GST_Reconciliation.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

else:
    st.info("👆 Upload both files to start reconciliation.")
