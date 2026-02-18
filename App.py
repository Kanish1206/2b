import streamlit as st
import pandas as pd
import io
import plotly.express as px
import reconciliation_logic as reco_logic

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="GST Reco Pro",
    page_icon="📘",
    layout="wide"
)

# --------------------------------------------------
# BLUE FINANCE THEME CSS
# --------------------------------------------------
st.markdown("""
<style>

/* Base */
html, body, [class*="css"] {
    font-family: "Segoe UI", system-ui, sans-serif;
}

.main {
    background-color: #eef7f1; /* LIGHT GREEN BACKGROUND */
    padding: 1.5rem;
}

/* Header */
.header {
    background: #d1fae5; /* SOFT GREEN */
    padding: 24px 30px;
    border-radius: 14px;
    border: 1px solid #a7f3d0;
    margin-bottom: 24px;
}

/* Section Cards */
.section {
    background: #f0fdf4; /* VERY LIGHT GREEN */
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #a7f3d0;
    margin-bottom: 22px;
}

/* Buttons */
.stButton>button {
    background-color: #15803d; /* PRIMARY GREEN */
    color: #ffffff;
    font-weight: 600;
    border-radius: 10px;
    height: 44px;
    border: none;
}
.stButton>button:hover {
    background-color: #166534;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #dcfce7;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #a7f3d0;
}
[data-testid="stMetricValue"] {
    color: #064e3b;
    font-weight: 700;
}
[data-testid="stMetricLabel"] {
    color: #065f46;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #064e3b; /* DARK GREEN */
}
section[data-testid="stSidebar"] * {
    color: #d1fae5;
}

/* Dataframe */
.stDataFrame {
    background: #f0fdf4;
    border-radius: 12px;
    border: 1px solid #a7f3d0;
}

/* Accent text */
.text-success { color: #16a34a; }
.text-warning { color: #d97706; }

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.title("📘 GST Reco Pro")
    st.caption("Finance Edition")

    st.divider()
    st.markdown("""
    **View File Requirement**
    
    → **Books:** Should contain columns like 'GSTIN', 'Reference Document No.', 'Document Date', and 'Invoice Value'.

    → **2B:** Should be the standard Excel export from the GST Portal.
    → **Format:** .xlsx
    """)

    st.divider()
    st.caption("Version 4.0")

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="header">
    <h2 style="color:#0f172a;">GST 2B vs Purchase Register</h2>
    <p style="color:#1e293b;">
        Finance-grade reconciliation with a blue trust-first interface.
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("📤 Upload Source Files")

c1, c2 = st.columns(2)
with c1:
    gst_file = st.file_uploader("GSTR-2B Excel (.xlsx)", type=["xlsx"])
with c2:
    pur_file = st.file_uploader("Purchase Register Excel (.xlsx)", type=["xlsx"])

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# PROCESSING
# --------------------------------------------------
if gst_file and pur_file:
    try:
        df_books = pd.read_excel(pur_file)
        df_2b = pd.read_excel(gst_file)

        df_books.columns = df_books.columns.str.strip()
        df_2b.columns = df_2b.columns.str.strip()

        if st.button("Run Reconciliation"):
            result_df = reco_logic.process_reco(df_books, df_2b)

            matched = result_df['Status'].str.contains("match", case=False, na=False).sum()
            total = len(result_df)
            diff = total - matched

            # SUMMARY
            st.markdown('<div class="section">', unsafe_allow_html=True)
            st.subheader("📊 Reconciliation Summary")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Books Records", len(df_books))
            m2.metric("2B Records", len(df_2b))
            m3.metric("Matched", matched)
            m4.metric("Differences", diff)

            st.markdown('</div>', unsafe_allow_html=True)

            # VISUAL (CLEAR BLUE + GREEN + WARM)
            st.markdown('<div class="section">', unsafe_allow_html=True)
            st.subheader("📈 Match Overview")

            pie_df = pd.DataFrame({
                "Category": ["Matched", "Differences"],
                "Count": [matched, diff]
            })

            fig = px.pie(
                pie_df,
                names="Category",
                values="Count",
                hole=0.6,
                color="Category",
                color_discrete_map={
                    "Matched": "#16a34a",     # GREEN
                    "Differences": "#d97706"  # WARM
                }
            )
            fig.update_layout(
                paper_bgcolor="#f5f9ff",
                plot_bgcolor="#f5f9ff"
            )

            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # TABLE
            st.markdown('<div class="section">', unsafe_allow_html=True)
            st.subheader("📄 Reconciliation Details")
            st.dataframe(result_df, use_container_width=True, height=480)
            st.markdown('</div>', unsafe_allow_html=True)

            # EXPORT
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                result_df.to_excel(writer, index=False)

            st.download_button(
                "⬇️ Download Excel Report",
                data=output.getvalue(),
                file_name="GST_Reconciliation.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(str(e))

else:
    st.info("Upload both files to proceed.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.caption("Trust • Compliance • Audit")
