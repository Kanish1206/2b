import streamlit as st
import pandas as pd
import io
import reconciliation_logic as reco_logic

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="GST Reco Pro",
    page_icon="📘",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
        .main-title {
            font-size: 32px;
            font-weight: 700;
        }
        .sub-text {
            color: #6c757d;
            margin-bottom: 20px;
        }
        .section {
            margin-top: 25px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">📘 GST Reconciliation Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Compare GSTR-2B with Purchase Register</div>', unsafe_allow_html=True)

# ---------------- FILE UPLOAD ----------------
st.markdown("### 📂 Upload Files")

col1, col2 = st.columns(2)

with col1:
    gst_file = st.file_uploader("Upload GSTR-2B Excel", type=["xlsx"])

with col2:
    pur_file = st.file_uploader("Upload Purchase Register Excel", type=["xlsx"])

# ---------------- MAIN LOGIC ----------------
if gst_file and pur_file:

    try:
        df_2b = pd.read_excel(gst_file)
        df_books = pd.read_excel(pur_file)

        df_2b.columns = df_2b.columns.str.strip()
        df_books.columns = df_books.columns.str.strip()

        st.success("✅ Files uploaded successfully")

        if st.button("🚀 Run Reconciliation", use_container_width=True):

            with st.spinner("Processing... ⏳"):
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

            # ---------------- FILTER ----------------
            #st.markdown("## 🔍 Filter Data")

            #status_options = ["All"] + sorted(result_df["Match_Status"].dropna().unique().tolist())
            #selected_status = st.selectbox("Filter by Match Status", status_options)

            #if selected_status != "All":
                #filtered_df = result_df[result_df["Match_Status"] == selected_status]
            #else:
                #filtered_df = result_df

            # ---------------- TABLE ----------------
            st.markdown("## 📋 Detailed Results")

            #st.dataframe( use_container_width=True)

            # ---------------- DOWNLOAD ----------------
            st.markdown("## 📥 Export Results")

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                filtered_df.to_excel(writer, index=False)

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
