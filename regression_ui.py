import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from scipy.stats import norm
import plotly.express as px

# ===== 1. PAGE CONFIG =====
st.set_page_config(
    page_title="📈 Regression Analysis",
    layout="wide"
)


# ===== 2. BACKGROUND & TEXT STYLING =====
def set_app_style():
    BG_IMAGE = "https://images.unsplash.com/photo-1639762681057-408e52192e55"  # Dark tech background

    st.markdown(
        f"""
        <style>
        /* Main background */
        .stApp {{
            background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.7)), 
                        url('{BG_IMAGE}');
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}

        /* Text styling */
        h1, h2, h3, h4, h5, h6 {{
            color: #E8EE0A !important;
            font-family: 'Arial', sans-serif;
            text-shadow: 1px 1px 3px #000000;
        }}

        h1 {{
            font-size: 3.5rem !important;
            margin-bottom: 1rem !important;
        }}

        h2 {{
            font-size: 2.5rem !important;
            margin-top: 1.5rem !important;
        }}

        /* Body text */
        p, .stMarkdown, .stText, .stAlert {{
            color: #e0e0e0 !important;
            font-size: 1.2rem !important;
            line-height: 1.6 !important;
        }}

        /* Containers */
        .main .block-container, 
        .stDataFrame, 
        .stAlert {{
            background-color: rgba(30, 30, 30, 0.5) !important;
            border-radius: 10px;
            padding: 2rem;
            border: 1px solid #444;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}

        /* Input fields */
        .stTextInput>div>div>input,
        .stSelectbox>div>div>select,
        .stNumberInput>div>div>input {{
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            border: 1px solid #555 !important;
        }}

        /* Tables */
        .stDataFrame {{
            color: #ffffff !important;
        }}

        /* Buttons */
        .stButton>button {{
            background-color: #4f46e5 !important;
            color: white !important;
            border-radius: 8px !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
        }}

        /* Expanders */
        .st-expander {{
            border: 1px solid #444 !important;
        }}

        .st-expanderHeader {{
            color: #ffffff !important;
            font-weight: bold !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


set_app_style()

# ===== 3. MAIN APP CONTENT =====
st.title("📈 Regression Analysis")

# [Rest of your existing regression analysis code remains exactly the same]
# ======================
# MODIFIED DATA UPLOAD SECTION
# ======================
uploaded_file = st.file_uploader("Upload your data file (CSV, Excel, or Text)",
                                 type=["csv", "xlsx", "txt"])

if uploaded_file is not None:
    # Read file based on extension
    try:
        if uploaded_file.name.endswith('.xlsx'):
            raw_data = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            # Try common delimiters
            try:
                raw_data = pd.read_csv(uploaded_file, sep='\t')  # Try tab first
            except:
                try:
                    raw_data = pd.read_csv(uploaded_file, sep=',')  # Then try comma
                except:
                    raw_data = pd.read_csv(uploaded_file, delim_whitespace=True)  # Fallback
        else:  # Default to CSV
            raw_data = pd.read_csv(uploaded_file)

    except Exception as e:
        st.error(f"Error reading file: {str(e)}\n\nSupported formats:\n"
                 "1. CSV - Comma separated values\n"
                 "2. Excel - .xlsx files\n"
                 "3. Text - Tab, comma, or space delimited")
        st.stop()

    # Display raw data
    st.subheader("📋 Raw Uploaded Data")
    st.write(f"Total rows uploaded: {len(raw_data)}")
    st.dataframe(raw_data, height=250)

    # ======================
    # DATA CLEANING SECTION (unchanged)
    # ======================
    st.subheader("🧹 Data Cleaning")

    numeric_cols = raw_data.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns for regression analysis!")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("Select X-axis column", numeric_cols, index=0)
    with col2:
        y_col = st.selectbox("Select Y-axis column", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

    # Clean data
    clean_data = raw_data[[x_col, y_col]].copy()
    clean_data.columns = ['x', 'y']
    clean_data = clean_data.apply(pd.to_numeric, errors='coerce').dropna()

    st.write(f"Rows after cleaning: {len(clean_data)} (Removed {len(raw_data) - len(clean_data)} invalid rows)")

    if len(clean_data) < 3:
        st.error("Need at least 3 valid data points for regression!")
        st.stop()

    with st.expander("View cleaned data"):
        st.dataframe(clean_data)

    # ======================
    # REGRESSION ANALYSIS WITH BAND WIDTH CONTROL
    # ======================
    st.subheader("📊 Regression Results")

    with st.expander("🔧 Interval Settings", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ci_level = st.slider(
                "Confidence Level (%)",
                min_value=5, max_value=99, value=95, step=1
            )
        with col2:
            ci_width = st.slider(
                "CI Width Multiplier",
                min_value=0.1, max_value=3.0, value=1.0, step=0.1,
                help="Adjust CI band width (1.0 = standard)"
            )
        with col3:
            pi_level = st.slider(
                "Prediction Level (%)",
                min_value=5, max_value=99, value=90, step=1
            )
        with col4:
            pi_width = st.slider(
                "PI Width Multiplier",
                min_value=0.1, max_value=3.0, value=1.0, step=0.1,
                help="Adjust PI band width (1.0 = standard)"
            )

    # Regression calculation with width control
    clean_data['const'] = 1
    model = sm.OLS(clean_data['y'], clean_data[['const', 'x']]).fit()

    # Get prediction and standard errors
    predictions = model.get_prediction(clean_data[['const', 'x']])
    mean_se = predictions.se_mean
    obs_se = np.sqrt(mean_se ** 2 + model.scale)  # For prediction intervals

    # Calculate critical value based on selected confidence level
    z = norm.ppf(1 - (1 - ci_level / 100) / 2)

    # Apply width multipliers
    results_df = clean_data.copy()
    results_df["Predicted"] = predictions.predicted_mean
    results_df["CI Lower"] = results_df["Predicted"] - z * mean_se * ci_width
    results_df["CI Upper"] = results_df["Predicted"] + z * mean_se * ci_width
    results_df["PI Lower"] = results_df["Predicted"] - z * obs_se * pi_width
    results_df["PI Upper"] = results_df["Predicted"] + z * obs_se * pi_width
    results_df_sorted = results_df.sort_values(by='x')

    # ======================
    # PLOT CUSTOMIZATION (MODIFIED)
    # ======================
    with st.expander("🎨 Plot Customization"):
        col1, col2, col3 = st.columns(3)
        with col1:
            point_color = st.color_picker("Data points color", "#1f77b4")
            point_size = st.slider("Point size", 1, 100, 30)
        with col2:
            line_color = st.color_picker("Regression line color", "#ff7f0e")
            line_width = st.slider("Line width", 1, 5, 2)
        with col3:
            ci_color = st.color_picker("Confidence interval", "#1f77b4")
            pi_color = st.color_picker("Prediction interval", "#ff7f0e")

        line_style = st.selectbox("Line style", ['solid', 'dashed', 'dotted', 'dashdot'])
        line_styles = {'solid': '-', 'dashed': '--', 'dotted': ':', 'dashdot': '-.'}

        x_label = st.text_input("X-axis label", x_col)
        y_label = st.text_input("Y-axis label", y_col)

        # Info box customization
        st.markdown("**Info Box Settings**")
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            eq_pos_x = st.slider("Box X position", 0.0, 1.0, 0.05)
            eq_pos_y = st.slider("Box Y position", 0.0, 1.0, 0.95)
        with col_info2:
            show_equation = st.checkbox("Show equation", value=True)
            show_rsquared = st.checkbox("Show R²", value=True)
        with col_info3:
            show_pvalue = st.checkbox("Show p-value", value=True)
            show_n = st.checkbox("Show sample size", value=False)

        st.markdown("**Legend Settings**")
        legend_pos = st.selectbox(
            "Legend Position",
            options=["best", "upper right", "upper left", "lower left", "lower right",
                     "right", "center left", "center right", "lower center",
                     "upper center", "center"],
            index=0
        )

    # ======================
    # PLOTTING (FIXED VERSION)
    # ======================
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(results_df_sorted['x'],
                    results_df_sorted['PI Lower'],
                    results_df_sorted['PI Upper'],
                    color=pi_color, alpha=0.2,
                    label=f'{pi_level}% Prediction Interval')

    ax.fill_between(results_df_sorted['x'],
                    results_df_sorted['CI Lower'],
                    results_df_sorted['CI Upper'],
                    color=ci_color, alpha=0.3,
                    label=f'{ci_level}% Confidence Interval')

    ax.scatter(results_df_sorted['x'], results_df_sorted['y'],
               color=point_color, s=point_size, alpha=0.7,
               edgecolor='black', linewidth=0.5, label='Observed Data')

    ax.plot(results_df_sorted['x'], results_df_sorted['Predicted'],
            color=line_color, linewidth=line_width,
            linestyle=line_styles[line_style], label='Regression Line')

    # Get all required metrics from the model
    intercept = model.params['const']
    slope = model.params['x']
    r_squared = model.rsquared
    p_value = model.f_pvalue
    sample_size = len(clean_data)

    # Dynamic info box content
    info_lines = []
    if show_equation:
        operator = "+" if slope >= 0 else "-"
        abs_slope = abs(slope)
        info_lines.append(f"y = {intercept:.4f} {operator} {abs_slope:.4f}x")
    if show_rsquared:
        info_lines.append(f"R² = {r_squared:.4f}")
    if show_pvalue:
        info_lines.append(f"p = {p_value:.2g}")
    if show_n:
        info_lines.append(f"N = {sample_size}")

    if info_lines:  # Only add box if at least one option is selected
        ax.annotate('\n'.join(info_lines),
                    xy=(eq_pos_x, eq_pos_y),
                    xycoords='axes fraction',
                    fontsize=12,
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Regression Analysis", pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc=legend_pos)  # Uses the user-selected position

    st.pyplot(fig)

    # ======================
    # STATISTICAL RESULTS (unchanged)
    # ======================
    st.subheader("📝 Statistical Summary")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.text(model.summary().as_text())
    with col2:
        st.markdown("**Key Metrics**")
        st.metric("R-squared", f"{r_squared:.4f}")
        st.metric("Adjusted R-squared", f"{model.rsquared_adj:.4f}")
        st.metric("F-statistic", f"{model.fvalue:.2f}")
        st.metric("Prob (F-statistic)", f"{p_value:.4g}")

    # ======================
    # DOWNLOAD SECTION (unchanged)
    # ======================
    st.subheader("💾 Export Results")

    col1, col2 = st.columns(2)
    with col1:
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button("Download Plot (PNG)", buf.getvalue(),
                           file_name="regression_plot.png",
                           mime="image/png")
    with col2:
        results_df_sorted['CI_Level'] = f"{ci_level}%"
        results_df_sorted['PI_Level'] = f"{pi_level}%"
        csv = results_df_sorted.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results (CSV)", csv,
                           file_name="regression_results.csv",
                           mime="text/csv")
