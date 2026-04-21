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

# ---------------- CSS ----------------
st.markdown("""
<style>
body {
    background-color: #F6F8FB;
}
.hero {
    padding: 20px 0px 10px 0px;
}
.title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #1E3A8A;
}
.subtitle {
    color: #64748B;
    margin-bottom: 20px;
}
.card {
    background: white;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="hero">', unsafe_allow_html=True)
st.markdown('<div class="title">📘 GST Reco Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered GST 2B vs Purchase Register Reconciliation</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")

filter_option = st.sidebar.selectbox(
    "Filter Data",
    ["All", "Matched", "Unmatched"]
)

search_text = st.sidebar.text_input("🔍 Search Vendor / Invoice")

# ---------------- FILE UPLOAD ----------------
st.subheader("📂 Upload Files")

col1, col2 = st.columns(2)

with col1:
    gst_file = st.file_uploader("Upload GSTR-2B", type=["xlsx"])

with col2:
    pur_file = st.file_uploader("Upload Purchase Register", type=["xlsx"])

# ---------------- CACHE ----------------
@st.cache_data
def load_file(file):
    df = pd.read_excel(file)
    df.columns = df.columns.str.strip()
    return df

# ---------------- MAIN ----------------
if gst_file and pur_file:

    try:
        df_2b = load_file(gst_file)
        df_books = load_file(pur_file)

        st.success("Files uploaded successfully ✔️")

        if st.button("🚀 Run Reconciliation", use_container_width=True):

            with st.spinner("Running smart reconciliation..."):
                result_df = reco_logic.process_reco(df_2b, df_books)

            # ---------------- KPIs ----------------
            total = len(result_df)
            matched = result_df["Match_Status"].str.contains("Match", case=False, na=False).sum()
            unmatched = total - matched
            accuracy = (matched / total * 100) if total else 0

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("📄 Total", f"{total:,}")
            c2.metric("✅ Matched", f"{matched:,}")
            c3.metric("❌ Unmatched", f"{unmatched:,}")
            c4.metric("🎯 Accuracy", f"{accuracy:.1f}%")

            st.divider()

            # ---------------- CHART ----------------
            st.subheader("📊 Overview")

            chart_df = pd.DataFrame({
                "Status": ["Matched", "Unmatched"],
                "Count": [matched, unmatched]
            })

            

            st.plotly_chart(fig, use_container_width=True)

            # ---------------- FILTER LOGIC ----------------
            if filter_option == "Matched":
                result_df = result_df[result_df["Match_Status"].str.contains("Match", case=False, na=False)]

            elif filter_option == "Unmatched":
                result_df = result_df[~result_df["Match_Status"].str.contains("Match", case=False, na=False)]

            # ---------------- SEARCH ----------------
            if search_text:
                result_df = result_df[
                    result_df.astype(str)
                    .apply(lambda row: row.str.contains(search_text, case=False).any(), axis=1)
                ]

            # ---------------- TABLE ----------------
            st.subheader("📋 Detailed Results")
            st.dataframe(result_df, use_container_width=True, height=500)

            # ---------------- DOWNLOAD ----------------
            st.subheader("📥 Export Report")

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                result_df.to_excel(writer, index=False)

            st.download_button(
                "⬇️ Download Excel",
                data=output.getvalue(),
                file_name="GST_Reco_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

else:
    st.info("👆 Upload both files to start reconciliation")
