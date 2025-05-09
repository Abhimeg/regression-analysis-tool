import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.stats import norm, f
import plotly.graph_objects as go  # NEW for interactive F-dist plot

st.set_page_config(layout="wide")
st.title("📈 Advanced Regression Analysis with F-Distribution")

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
    # ENHANCED STATISTICAL RESULTS
    # ======================
    st.subheader("📝 Statistical Summary")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.text(model.summary().as_text())
    with col2:
        st.markdown("**Key Metrics**")
        st.metric("R-squared", f"{model.rsquared:.4f}")
        st.metric("Adjusted R-squared", f"{model.rsquared_adj:.4f}")
        st.metric("F-statistic", f"{model.fvalue:.2f}")
        st.metric("Prob (F-statistic)", f"{model.f_pvalue:.4g}")

        # NEW F-DISTRIBUTION VISUALIZATION
        if st.toggle("📊 Show F-distribution", help="Visualize where your F-statistic falls on the distribution"):
            dfn = model.df_model  # Numerator df (number of predictors)
            dfd = model.df_resid  # Denominator df (n - p - 1)

            # Create F-distribution curve
            x = np.linspace(0, f.ppf(0.999, dfn, dfd), 500)
            y = f.pdf(x, dfn, dfd)

            # Create critical value for alpha=0.05
            critical_value = f.ppf(0.95, dfn, dfd)

            fig_f = go.Figure()

            # Main F-distribution curve
            fig_f.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                name=f'F({dfn:.0f},{dfd:.0f})',
                line=dict(color='blue', width=2)
            ))

            # Critical region shading
            mask = x > critical_value
            fig_f.add_trace(go.Scatter(
                x=x[mask], y=y[mask],
                fill='tozeroy',
                name='Critical Region (α=0.05)',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(width=0)
            ))

            # Add observed F-statistic
            fig_f.add_vline(
                x=model.fvalue,
                line=dict(color='red', dash='dash', width=2),
                annotation=dict(text=f"Your F = {model.fvalue:.2f}",
                                font=dict(color="red"))
            )

            # Add critical value line
            fig_f.add_vline(
                x=critical_value,
                line=dict(color='green', dash='dot', width=2),
                annotation=dict(text=f"Critical F = {critical_value:.2f}",
                                font=dict(color="green"))
            )

            fig_f.update_layout(
                title=f"F-Distribution (dfn={dfn:.0f}, dfd={dfd:.0f})",
                xaxis_title="F Value",
                yaxis_title="Probability Density",
                hovermode="x unified",
                showlegend=True
            )

            st.plotly_chart(fig_f, use_container_width=True)

            # Interpretation help
            st.info(f"""
            **Interpretation Guide:**
            - The red dashed line shows your model's F-statistic ({model.fvalue:.2f})
            - The green dotted line shows the critical F-value ({critical_value:.2f}) at α=0.05
            - If red line is in the shaded area, your model is statistically significant
            - Current p-value: {model.f_pvalue:.4g}
            """)
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
    # PLOT CUSTOMIZATION (unchanged)
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
        eq_pos_x = st.slider("Equation X position", 0.0, 1.0, 0.05)
        eq_pos_y = st.slider("Equation Y position", 0.0, 1.0, 0.95)

    # ======================
    # PLOTTING (unchanged)
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
