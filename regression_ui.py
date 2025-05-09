import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.stats import norm

st.set_page_config(layout="wide")
st.title("📈 Advanced Regression Analysis")

# ======================
# DATA UPLOAD SECTION
# ======================
uploaded_file = st.file_uploader("Upload your data file (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read file based on extension
    try:
        if uploaded_file.name.endswith('.xlsx'):
            raw_data = pd.read_excel(uploaded_file)
        else:
            raw_data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.stop()

    # Display raw data
    st.subheader("📋 Raw Uploaded Data")
    st.write(f"Total rows uploaded: {len(raw_data)}")
    st.dataframe(raw_data, height=250)

    # ======================
    # DATA CLEANING SECTION
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
    # REGRESSION ANALYSIS
    # ======================
    st.subheader("📊 Regression Results")

    # ======================
    # NEW: INTERVAL SETTINGS
    # ======================
    with st.expander("🔧 Interval Settings", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ci_level = st.slider(
                "Confidence Level (%)",
                min_value=50, max_value=99, value=95, step=1
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
                min_value=50, max_value=99, value=90, step=1
            )
        with col4:
            pi_width = st.slider(
                "PI Width Multiplier",
                min_value=0.1, max_value=3.0, value=1.0, step=0.1,
                help="Adjust PI band width (1.0 = standard)"
            )

    # ======================
    # UPDATED REGRESSION CALCULATION
    # ======================
    clean_data['const'] = 1
    model = sm.OLS(clean_data['y'], clean_data[['const', 'x']]).fit()

    # Get prediction and standard errors
    predictions = model.get_prediction(clean_data[['const', 'x']])
    mean_ci = predictions.conf_int(alpha=1 - ci_level / 100)
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
    # PLOT CUSTOMIZATION SECTION
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

        # NEW: Band thickness controls
        col4, col5 = st.columns(2)
        with col4:
            ci_thickness = st.slider(
                "CI Band Thickness",
                min_value=0.1, max_value=1.0, value=0.3, step=0.1,
                help="Transparency of confidence band"
            )
        with col5:
            pi_thickness = st.slider(
                "PI Band Thickness",
                min_value=0.1, max_value=1.0, value=0.2, step=0.1,
                help="Transparency of prediction band"
            )

        line_style = st.selectbox("Line style", ['solid', 'dashed', 'dotted', 'dashdot'])
        line_styles = {'solid': '-', 'dashed': '--', 'dotted': ':', 'dashdot': '-.'}

        x_label = st.text_input("X-axis label", x_col)
        y_label = st.text_input("Y-axis label", y_col)
        eq_pos_x = st.slider("Equation X position", 0.0, 1.0, 0.05)
        eq_pos_y = st.slider("Equation Y position", 0.0, 1.0, 0.95)

    # ======================
    # UPDATED PLOTTING SECTION
    # ======================
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot intervals with adjustable thickness
    ax.fill_between(
        results_df_sorted['x'],
        results_df_sorted['PI Lower'],
        results_df_sorted['PI Upper'],
        color=pi_color,
        alpha=pi_thickness,  # Updated to use slider value
        label=f'{pi_level}% Prediction Interval'
    )

    ax.fill_between(
        results_df_sorted['x'],
        results_df_sorted['CI Lower'],
        results_df_sorted['CI Upper'],
        color=ci_color,
        alpha=ci_thickness,  # Updated to use slider value
        label=f'{ci_level}% Confidence Interval'
    )

    # Rest of your plotting code remains the same...
    ax.scatter(results_df_sorted['x'], results_df_sorted['y'],
               color=point_color, s=point_size, alpha=0.7,
               edgecolor='black', linewidth=0.5, label='Observed Data')

    ax.plot(results_df_sorted['x'], results_df_sorted['Predicted'],
            color=line_color, linewidth=line_width,
            linestyle=line_styles[line_style], label='Regression Line')

    # ======================
    # PLOTTING
    # ======================
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot intervals first (UPDATED with dynamic labels)
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

    # Plot points and line
    ax.scatter(results_df_sorted['x'], results_df_sorted['y'],
               color=point_color, s=point_size, alpha=0.7,
               edgecolor='black', linewidth=0.5, label='Observed Data')

    ax.plot(results_df_sorted['x'], results_df_sorted['Predicted'],
            color=line_color, linewidth=line_width,
            linestyle=line_styles[line_style], label='Regression Line')

    # Add regression equation
    intercept = model.params['const']
    slope = model.params['x']
    r_squared = model.rsquared
    p_value = model.f_pvalue

    operator = "+" if slope >= 0 else "-"
    abs_slope = abs(slope)
    eq_text = (f"y = {intercept:.4f} {operator} {abs_slope:.4f}x\n"
               f"R² = {r_squared:.4f}\n"
               f"p-value = {p_value:.2g}")

    ax.annotate(eq_text, xy=(eq_pos_x, eq_pos_y), xycoords='axes fraction',
                fontsize=12, color='black',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Regression Analysis", pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')

    st.pyplot(fig)

    # ======================
    # STATISTICAL RESULTS
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
    # DOWNLOAD SECTION
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
        # UPDATED: Include interval levels in exported data
        results_df_sorted['CI_Level'] = f"{ci_level}%"
        results_df_sorted['PI_Level'] = f"{pi_level}%"
        csv = results_df_sorted.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results (CSV)", csv,
                           file_name="regression_results.csv",
                           mime="text/csv")
      
