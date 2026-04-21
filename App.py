import streamlit as st
import pandas as pd
import io
import plotly.express as px
import reconciliation_logic as reco_logic

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="GST Reco Pro",
    page_icon="📘",
    layout="wide"
)

# ---------------- CLEAN CSS ----------------
st.markdown("""
<style>
.main-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: #1E3A8A;
}
.sub-text {
    color: #64748B;
    margin-bottom: 20px;
}
.card {
    background: white;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">📘 GST Reconciliation Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Smart comparison of GSTR-2B vs Purchase Register</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")

show_only_unmatched = st.sidebar.checkbox("Show only unmatched", value=False)

# ---------------- FILE UPLOAD ----------------
st.subheader("📂 Upload Files")

col1, col2 = st.columns(2)

with col1:
    gst_file = st.file_uploader("GSTR-2B File", type=["xlsx"])

with col2:
    pur_file = st.file_uploader("Purchase Register", type=["xlsx"])

# ---------------- CACHE ----------------
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    df.columns = df.columns.str.strip()
    return df

# ---------------- MAIN ----------------
if gst_file and pur_file:

    try:
        df_2b = load_data(gst_file)
        df_books = load_data(pur_file)

        st.success("Files loaded successfully ✔️")

        if st.button("🚀 Run Reconciliation", use_container_width=True):

            with st.spinner("Running reconciliation..."):
                result_df = reco_logic.process_reco(df_2b, df_books)

            # ---------------- METRICS ----------------
            total = len(result_df)
            matched = result_df["Match_Status"].str.contains("Match", case=False, na=False).sum()
            unmatched = total - matched

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Total Records", f"{total:,}")
            c2.metric("Matched", f"{matched:,}")
            c3.metric("Unmatched", f"{unmatched:,}")
            c4.metric("Accuracy %", f"{(matched/total*100):.1f}%" if total else "0%")

            st.divider()

            # ---------------- CHART ----------------
            st.subheader("📊 Reconciliation Overview")

            chart_df = pd.DataFrame({
                "Status": ["Matched", "Unmatched"],
                "Count": [matched, unmatched]
            })

            fig = px.pie(chart_df, names="Status", values="Count",
                         color="Status",
                         color_discrete_map={
                             "Matched": "#10B981",
                             "Unmatched": "#EF4444"
                         })

            st.plotly_chart(fig, use_container_width=True)

            # ---------------- FILTER ----------------
            st.subheader("📋 Detailed Results")

            if show_only_unmatched:
                result_df = result_df[~result_df["Match_Status"].str.contains("Match", case=False, na=False)]

            st.dataframe(result_df, use_container_width=True, height=450)

            # ---------------- DOWNLOAD ----------------
            st.subheader("📥 Export")

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                result_df.to_excel(writer, index=False)

            st.download_button(
                "⬇️ Download Report",
                data=output.getvalue(),
                file_name="GST_Reco_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    st.info("Upload both files to begin")
