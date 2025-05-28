import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data, train_and_forecast_random_forest, train_and_forecast_auto_arima

st.set_page_config(
    page_title="Rainfall Analysis & Forecasting",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)


DATA_PATH = "data.csv"

try:
    rainfall_df_full = load_data(DATA_PATH)
except FileNotFoundError as e:
    st.error(f"🔴 ERROR: {e}. Not Found")
except Exception as e:
    st.error(f"🔴 ERROR: An error occurred while loading data: {e}")


st.title("🌧️ Indian Subdivisions Rainfall Analysis & Forecasting")
st.markdown("""
    An interactive tool to explore historical monsoon rainfall (June-September) 
    for various Indian subdivisions and to generate future forecasts using AI/ML.
""")

if rainfall_df_full is not None:

    st.sidebar.header("⚙️ User Controls")
    
    all_subdivisions = sorted(rainfall_df_full['subdivision'].unique())
    default_subdivision_index = 0
    if "COASTAL KARNATAKA" in all_subdivisions:
        default_subdivision_index = all_subdivisions.index("COASTAL KARNATAKA")
        
    selected_subdivision = st.sidebar.selectbox(
        "Select Subdivision:",
        options=all_subdivisions,
        index=default_subdivision_index,
        help="Choose the meteorological subdivision for analysis and forecasting."
    )

    min_year_overall = int(rainfall_df_full['YEAR'].min())
    max_year_overall = int(rainfall_df_full['YEAR'].max())
    
    default_start_year = max(min_year_overall, max_year_overall - 29)
    selected_year_range_analysis = st.sidebar.slider(
        "Select Year Range for Historical Analysis:",
        min_value=min_year_overall,
        max_value=max_year_overall,
        value=(default_start_year, max_year_overall),
        help="Filter the historical data displayed in 'Trend Analysis' and 'Distribution & Statistics' tabs."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔮 Forecasting Settings")

    selected_model_type = st.sidebar.radio(
        "Select Forecasting Model:",
        options=["Random Forest", "Auto-ARIMA"],
        index=0, 
        help="Choose the algorithm for rainfall forecasting."
    )

    num_forecast_years_input = st.sidebar.number_input(
        "Years to Forecast (JUN-SEP):",
        min_value=1, max_value=10, value=3,
        help="How many years into the future to predict total monsoon rainfall."
    )
    n_lags_input = 3
    if selected_model_type == "Random Forest":
        n_lags_input = st.sidebar.slider(
            "Past Years (Lags) for RF Prediction:",
            min_value=1, max_value=10, value=3,
            help="For Random Forest: number of previous years' rainfall to use as predictors."
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed by: **Yash Tiwari**")
    st.sidebar.markdown("Inspired by the ISRO RESPOND Programme.")
    st.sidebar.markdown("---")

    subdivision_data_analysis = rainfall_df_full[
        (rainfall_df_full['subdivision'] == selected_subdivision) &
        (rainfall_df_full['YEAR'] >= selected_year_range_analysis[0]) &
        (rainfall_df_full['YEAR'] <= selected_year_range_analysis[1])
    ].copy()
    
    subdivision_data_for_model_training = rainfall_df_full[
        rainfall_df_full['subdivision'] == selected_subdivision
    ].copy()

    st.header(f"📍 Analysis & Forecast for: {selected_subdivision}")
    
    tab_trend, tab_dist, tab_forecast, tab_data_view, tab_about = st.tabs([
        "📈 Trend Analysis", 
        "📊 Distribution & Statistics", 
        "🌦️ Rainfall Forecasting", 
        "📄 Data View",
        "ℹ️ About this App"
    ])


    with tab_trend:
        st.subheader("Historical Rainfall Trends")
        st.markdown(f"Displaying data from **{selected_year_range_analysis[0]}** to **{selected_year_range_analysis[1]}**.")
        if subdivision_data_analysis.empty:
            st.warning("No historical data available.")
        else:
            st.markdown("#### Total Monsoon Rainfall (JUN-SEP) Over Years")
            fig_monsoon_trend = px.line(subdivision_data_analysis, x='YEAR', y='JUN-SEP', labels={'YEAR': 'Year', 'JUN-SEP': 'Total Rainfall (mm)'}, markers=True)
            st.plotly_chart(fig_monsoon_trend, use_container_width=True)
            st.markdown("#### Monthly Rainfall (JUN, JUL, AUG, SEP) Over Years")
            monthly_data_filtered = subdivision_data_analysis.melt(id_vars=['YEAR', 'subdivision'], value_vars=['JUN', 'JUL', 'AUG', 'SEP'], var_name='Month', value_name='Rainfall_mm')
            fig_monthly_trend = px.line(monthly_data_filtered, x='YEAR', y='Rainfall_mm', color='Month', labels={'YEAR': 'Year', 'Rainfall_mm': 'Monthly Rainfall (mm)'}, markers=True)
            st.plotly_chart(fig_monthly_trend, use_container_width=True)

    with tab_dist:
        st.subheader("Rainfall Distribution & Key Statistics")
        st.markdown(f"Based on data from **{selected_year_range_analysis[0]}** to **{selected_year_range_analysis[1]}**.")
        if subdivision_data_analysis.empty:
            st.warning("No historical data available.")
        else:
            st.markdown("#### Summary Statistics")
            stats_cols = ['JUN', 'JUL', 'AUG', 'SEP', 'JUN-SEP']
            summary_stats = subdivision_data_analysis[stats_cols].agg(['mean', 'median', 'std', 'min', 'max']).T; summary_stats.columns = ['Mean', 'Median', 'Std. Dev.', 'Min', 'Max']
            st.dataframe(summary_stats.style.format("{:.2f} mm"))
            st.markdown("#### Rainfall Distribution (Box Plots)")
            st.markdown("Box plots show spread & central tendency. Box=IQR, Line=Median, Whiskers=1.5xIQR.")
            boxplot_data = subdivision_data_analysis.melt(id_vars=['YEAR'], value_vars=['JUN', 'JUL', 'AUG', 'SEP', 'JUN-SEP'], var_name='Period', value_name='Rainfall_mm')
            fig_boxplots = px.box(boxplot_data, x='Period', y='Rainfall_mm', color='Period', title='Distribution of Rainfall by Month and Total Monsoon Season', labels={'Period': 'Month/Season', 'Rainfall_mm': 'Rainfall (mm)'}, points="outliers")
            st.plotly_chart(fig_boxplots, use_container_width=True)
            st.markdown("#### Frequency Distribution of Total Monsoon Rainfall (JUN-SEP)")
            fig_hist_total_monsoon = px.histogram(subdivision_data_analysis, x='JUN-SEP', nbins=20, marginal="box", title='Histogram of Total Monsoon Rainfall (JUN-SEP)', labels={'JUN-SEP': 'Total Monsoon Rainfall (mm)'})
            st.plotly_chart(fig_hist_total_monsoon, use_container_width=True)


    with tab_forecast:
        st.subheader(f"Forecasting Total Monsoon Rainfall (JUN-SEP) using {selected_model_type}")
        target_col_to_forecast = 'JUN-SEP'
        forecast_col_name = f'Forecasted_{target_col_to_forecast}'

        st.markdown(f"Model trained on historical data for **{selected_subdivision}** (all available years up to **{max_year_overall}**).")
        if selected_model_type == "Random Forest":
            st.markdown(f"Random Forest uses **{n_lags_input}** past year(s) of {target_col_to_forecast} rainfall as predictors.")
        
        st.markdown(f"Forecasting for the next **{num_forecast_years_input}** year(s).")

        trained_model = None
        forecast_df = None
        error_msg = None

        if subdivision_data_for_model_training.empty:
            st.warning(f"No historical data available for {selected_subdivision} to train any model.")
        else:
            if selected_model_type == "Random Forest":
                if len(subdivision_data_for_model_training) < n_lags_input + 10:
                    st.warning(f"Insufficient data for Random Forest (need at least {n_lags_input + 10} years).")
                else:
                    trained_model, forecast_df, error_msg = train_and_forecast_random_forest(
                        historical_data_df=subdivision_data_for_model_training[['YEAR', target_col_to_forecast]], 
                        target_col_name=target_col_to_forecast, 
                        num_lags=n_lags_input, 
                        num_future_years=num_forecast_years_input
                    )
            elif selected_model_type == "Auto-ARIMA":
                if len(subdivision_data_for_model_training) < 10: 
                    st.warning("Insufficient data for Auto-ARIMA (need at least 10 years).")
                else:
                    trained_model, forecast_df, error_msg = train_and_forecast_auto_arima(
                        historical_data_df=subdivision_data_for_model_training[['YEAR', target_col_to_forecast]], 
                        target_col_name=target_col_to_forecast,
                        num_future_years=num_forecast_years_input
                    )
        
        if error_msg:
            st.error(f"Forecasting Error ({selected_model_type}): {error_msg}")
        elif forecast_df is not None and not forecast_df.empty:
            st.markdown(f"#### Forecasted {target_col_to_forecast} Rainfall ({selected_model_type}):")
            display_forecast_df = forecast_df[['YEAR', forecast_col_name]].copy()
            if 'Conf_Int_Lower' in forecast_df.columns and 'Conf_Int_Upper' in forecast_df.columns:
                 display_forecast_df['Confidence Interval'] = forecast_df.apply(lambda row: f"[{row['Conf_Int_Lower']:.2f} - {row['Conf_Int_Upper']:.2f}]", axis=1)
            
            st.dataframe(display_forecast_df.style.format({'YEAR': '{}', forecast_col_name: "{:.2f} mm"}))

            fig_forecast_plot = go.Figure()
            hist_data_to_plot_fc = subdivision_data_analysis 
            
            fig_forecast_plot.add_trace(go.Scatter(
                x=hist_data_to_plot_fc['YEAR'], y=hist_data_to_plot_fc[target_col_to_forecast], 
                mode='lines+markers', name='Historical Rainfall', line=dict(color='royalblue')
            ))
            
            fig_forecast_plot.add_trace(go.Scatter(
                x=forecast_df['YEAR'], y=forecast_df[forecast_col_name], 
                mode='lines+markers', name=f'{selected_model_type} Forecast', line=dict(color='darkorange', dash='dash')
            ))
            if selected_model_type == "Auto-ARIMA" and 'Conf_Int_Lower' in forecast_df.columns:
                fig_forecast_plot.add_trace(go.Scatter(
                    x=forecast_df['YEAR'], y=forecast_df['Conf_Int_Upper'],
                    mode='lines', line=dict(width=0), name='Upper CI Bound', showlegend=False
                ))
                fig_forecast_plot.add_trace(go.Scatter(
                    x=forecast_df['YEAR'], y=forecast_df['Conf_Int_Lower'],
                    mode='lines', line=dict(width=0), name='Lower CI Bound', fillcolor='rgba(255,165,0,0.2)',
                    fill='tonexty', showlegend=False 
                ))
            
            fig_forecast_plot.update_layout(
                title=f'Historical and Forecasted {target_col_to_forecast} Rainfall ({selected_model_type}) for {selected_subdivision}',
                xaxis_title='Year', yaxis_title='Rainfall (mm)', legend_title_text='Data Type'
            )
            st.plotly_chart(fig_forecast_plot, use_container_width=True)
            
            with st.expander(f"View Model & Training Details ({selected_model_type} - Optional)"):
                st.markdown(f"""
                    - **Model Used:** {selected_model_type}
                    - **Training Data Range:** Historical {target_col_to_forecast} rainfall for {selected_subdivision} from {subdivision_data_for_model_training['YEAR'].min()} to {subdivision_data_for_model_training['YEAR'].max()}.
                """)
                if selected_model_type == "Random Forest" and trained_model:
                    st.markdown(f"- **Number of Estimators (Trees):** {trained_model.n_estimators}")
                    st.markdown(f"- **Features Used (Lags):** {n_lags_input}")
                    if hasattr(trained_model, 'oob_score_') and trained_model.oob_score_:
                         st.write(f"- **Model Out-of-Bag (OOB) Score:** {trained_model.oob_score_:.4f}")
                    if hasattr(trained_model, 'feature_importances_'):
                        st.write(f"- **Feature Importances:**")
                        importances = pd.Series(trained_model.feature_importances_, index=[f'Lag {i+1}' for i in range(n_lags_input)])
                        st.bar_chart(importances.sort_values(ascending=False))
                elif selected_model_type == "Auto-ARIMA" and trained_model:
                    st.markdown(f"- **Selected ARIMA Order (p,d,q):** {trained_model.order}")
                    st.markdown(f"- **Selected Seasonal Order (P,D,Q,s):** {trained_model.seasonal_order}") 
                    st.markdown(f"- **AIC (Akaike Info Criterion):** {trained_model.aic():.2f} (Lower is generally better)")
                    st.markdown(f"**Summary:**") # check
                    st.text(trained_model.summary()) 
        else:
            st.error(f"Forecasting with {selected_model_type} did not produce results. Check data and model parameters.")

    # Data View and About tabs remain the same
    with tab_data_view:
        st.subheader(f"Raw Data Snippet for {selected_subdivision}")
        st.markdown(f"Displaying data from **{selected_year_range_analysis[0]}** to **{selected_year_range_analysis[1]}**.")
        display_cols = ['YEAR', 'JUN', 'JUL', 'AUG', 'SEP', 'JUN-SEP']
        if st.checkbox(f"Show full data table (filtered by year range)", value=True):
            st.dataframe(subdivision_data_analysis[display_cols].reset_index(drop=True))
        if st.checkbox(f"Show descriptive statistics for this data table", value=False):
            st.dataframe(subdivision_data_analysis[display_cols].describe().style.format("{:.2f}"))

    with tab_about:
        st.subheader("About This Application")
        st.markdown(f"""
            This interactive application is designed to analyze historical monsoon rainfall data for various meteorological subdivisions of India and to provide future rainfall forecasts using machine learning and statistical techniques.
            
            **Developed by:** Yash Tiwari 
            ---
            #### **Purpose & Linkage to ISRO Project**
            This tool serves as a practical demonstration aligned with the objectives of the ISRO RESPOND project titled: 
            
            *"Advances in Numerical Weather Prediction (NWP) Modelling using AI/ML Techniques"*. 
            It showcases how AI/ML and statistical models can enhance weather parameter prediction.
            ---
            #### **Data Source**
            - Sub-divisional monsoon rainfall data (1901-2021) from IMD archives.
            ---
            #### **Methodology**
            - **Historical Analysis:** Pandas, Plotly Express.
            - **Forecasting Models:**
                - **Random Forest Regressor:** A machine learning ensemble model from `scikit-learn`. Uses lagged rainfall values as features.
                - **Auto-ARIMA:** An automated approach to fit Autoregressive Integrated Moving Average models using `pmdarima`. Identifies optimal (p,d,q) orders.
            ---
            #### **Technologies Used**
            - Python, Streamlit, Pandas, NumPy, Scikit-learn, Plotly, Pmdarima.
            """)
        st.markdown("---")
        st.markdown("For any queries, contact me at yashtiwari9182@gmail.com")
else:
    st.error("🔴 CRITICAL ERROR: Rainfall dataset could not be loaded.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>A Proof-of-Concept Application for ISRO RESPOND</div>", unsafe_allow_html=True)