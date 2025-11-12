import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
from fpdf import FPDF

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Universal CSV Analyzer", layout="wide")
st.title("📊 Universal Data Insights Dashboard (Pro Edition)")

# ------------------- FILE UPLOAD -------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
    df = df.apply(lambda col: pd.to_numeric(col, errors='ignore') if col.dtype == 'object' else col)

    # ------------------- SIDEBAR CLEANING -------------------
    st.sidebar.header("🧹 Data Cleaning & Filters")
    if st.sidebar.checkbox("Remove Duplicates"):
        df.drop_duplicates(inplace=True)

    if st.sidebar.checkbox("Fill Missing Values"):
        fill_method = st.sidebar.selectbox("Choose fill method", ["Mean", "Median", "Mode", "Zero", "None"])
        for col in df.select_dtypes(include='number'):
            if fill_method == "Mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif fill_method == "Median":
                df[col].fillna(df[col].median(), inplace=True)
            elif fill_method == "Mode":
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif fill_method == "Zero":
                df[col].fillna(0, inplace=True)

    # ------------------- COLUMN TYPE IDENTIFICATION -------------------
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    # ------------------- NUMERIC RANGE FILTER -------------------
    st.sidebar.subheader("📐 Numeric Range Filters")
    for col in numeric_cols:
        min_val, max_val = float(df[col].min()), float(df[col].max())
        selected_range = st.sidebar.slider(f"{col} range", min_val, max_val, (min_val, max_val))
        df = df[df[col].between(*selected_range)]

    # ------------------- FILTERS -------------------
    st.sidebar.subheader("🔍 Interactive Filters")
    filter_col = st.sidebar.multiselect("Select columns to filter", df.columns)
    for col in filter_col:
        unique_vals = df[col].dropna().unique().tolist()
        selected_vals = st.sidebar.multiselect(f"Filter {col}", unique_vals)
        if selected_vals:
            df = df[df[col].isin(selected_vals)]

    # ------------------- SUMMARY METRICS -------------------
    st.subheader("📈 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", df.isnull().sum().sum())
    c4.metric("Duplicate Rows", df.duplicated().sum())
    st.dataframe(df.head())

    # ------------------- HELPER FUNCTION TO SAVE FIGURES -------------------
    def get_fig_download(fig, filename):
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        st.download_button(f"⬇ Download {filename}", data=buf, file_name=filename, mime="image/png")

    # ------------------- HISTOGRAM -------------------
    st.subheader("📊 Histogram")
    if numeric_cols:
        hist_col = st.selectbox("Select numeric column", numeric_cols)
        fig1, ax1 = plt.subplots()
        sns.histplot(df[hist_col].dropna(), kde=True, ax=ax1)
        st.pyplot(fig1)
        get_fig_download(fig1, "histogram.png")
        plt.close(fig1)

    # ------------------- HEATMAP -------------------
    if len(numeric_cols) >= 2:
        st.subheader("🔥 Correlation Heatmap")
        fig2, ax2 = plt.subplots()
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)
        get_fig_download(fig2, "heatmap.png")
        plt.close(fig2)

    # ------------------- PIE CHART -------------------
    st.subheader("🥧 Pie Chart")
    if categorical_cols:
        pie_col = st.selectbox("Select categorical column", categorical_cols)
        pie_data = df[pie_col].value_counts().head(10)
        fig3, ax3 = plt.subplots()
        ax3.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
        ax3.axis('equal')
        st.pyplot(fig3)
        get_fig_download(fig3, "piechart.png")
        plt.close(fig3)

    # ------------------- BOX PLOT -------------------
    st.subheader("📦 Box Plot")
    if categorical_cols and numeric_cols:
        cat_col = st.selectbox("X-axis (categorical)", categorical_cols)
        num_col = st.selectbox("Y-axis (numeric)", numeric_cols)
        fig4, ax4 = plt.subplots()
        sns.boxplot(x=cat_col, y=num_col, data=df, ax=ax4)
        plt.xticks(rotation=45)
        st.pyplot(fig4)
        get_fig_download(fig4, "boxplot.png")
        plt.close(fig4)

    # ------------------- SCATTER PLOT -------------------
    st.subheader("📌 Scatter Plot")
    if len(numeric_cols) >= 2:
        x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
        y_col = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
        fig5, ax5 = plt.subplots()
        sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax5)
        st.pyplot(fig5)
        get_fig_download(fig5, "scatterplot.png")
        plt.close(fig5)

    # ------------------- BAR CHART -------------------
    st.subheader("📊 Bar Chart")
    if categorical_cols:
        bar_col = st.selectbox("Select categorical column for bar chart", categorical_cols)
        bar_data = df[bar_col].value_counts().head(10)
        fig6, ax6 = plt.subplots()
        sns.barplot(x=bar_data.index, y=bar_data.values, ax=ax6)
        plt.xticks(rotation=45)
        st.pyplot(fig6)
        get_fig_download(fig6, "barchart.png")
        plt.close(fig6)

    # ------------------- LINE CHART -------------------
    st.subheader("📉 Line Chart (Trends)")
    date_cols = df.select_dtypes(include=['datetime', 'object']).columns
    time_col = st.selectbox("Select X-axis column", date_cols)
    line_y = st.selectbox("Select Y-axis column", numeric_cols)
    if time_col and line_y:
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except:
            pass
        df_sorted = df.sort_values(by=time_col)
        st.line_chart(df_sorted[[time_col, line_y]].set_index(time_col))
        # Download line chart
        fig7, ax7 = plt.subplots()
        ax7.plot(df_sorted[time_col], df_sorted[line_y])
        ax7.set_xlabel(time_col)
        ax7.set_ylabel(line_y)
        plt.xticks(rotation=45)
        st.pyplot(fig7)
        get_fig_download(fig7, "linechart.png")
        plt.close(fig7)

    # ------------------- DOWNLOAD PROCESSED DATA -------------------
    st.subheader("💾 Download Processed Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Download CSV", data=csv, file_name="cleaned_data.csv", mime="text/csv")

    excel_buffer = BytesIO()
    df.to_excel(excel_buffer, index=False, sheet_name="CleanedData")
    excel_data = excel_buffer.getvalue()
    st.download_button("⬇ Download Excel", data=excel_data, file_name="cleaned_data.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ------------------- EXPORT TO PDF -------------------
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="Universal Data Insights Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Dataset Summary:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8, f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}\nMissing Values: {df.isnull().sum().sum()}\nDuplicate Rows: {df.duplicated().sum()}")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Columns:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8, ", ".join(df.columns))
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Statistical Summary:", ln=True)
    pdf.set_font("Arial", size=9)
    desc = df.describe(include='all').fillna("").astype(str)
    for i, row in desc.iterrows():
        row_text = f"{i}: " + " | ".join([str(x) for x in row.values])
        pdf.multi_cell(0, 6, row_text)
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    st.download_button("📄 Download Summary as PDF", data=pdf_bytes, file_name="data_summary.pdf", mime="application/pdf")
