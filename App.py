import streamlit as st
import pandas as pd
import io
import plotly.express as px
import reconciliation_logic as reco_logic

st.set_page_config(page_title="GST Reco Pro", page_icon="📘", layout="wide")

st.title("GST 2B vs Purchase Register Reconciliation")

gst_file = st.file_uploader("Upload GSTR-2B Excel", type=["xlsx"])
pur_file = st.file_uploader("Upload Purchase Register Excel", type=["xlsx"])

if gst_file and pur_file:

    try:
        df_2b = pd.read_excel(gst_file)
        df_books = pd.read_excel(pur_file)

        df_2b.columns = df_2b.columns.str.strip()
        df_books.columns = df_books.columns.str.strip()

        if st.button("Run Reconciliation"):

            result_df = reco_logic.process_reco(df_2b, df_books)

            total = len(result_df)
            matched = result_df["Match_Status"].str.contains(
                "Match", case=False, na=False
            ).sum()

            st.subheader("Summary")

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Records", total)
            c2.metric("Matched", matched)
            c3.metric("Unmatched", total - matched)

            pie_df = result_df["Match_Status"].value_counts().reset_index()
            pie_df.columns = ["Status", "Count"]

            fig = px.pie(
                pie_df,
                names="Status",
                values="Count",
                hole=0.5,
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Detailed Output")
            st.dataframe(result_df, use_container_width=True)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                result_df.to_excel(writer, index=False)

            st.download_button(
                "Download Excel",
                data=output.getvalue(),
                file_name="GST_Reconciliation.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    except Exception as e:
        st.error(str(e))

else:
    st.info("Upload both files to start reconciliation.")
