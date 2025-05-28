import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import streamlit as st 
import pmdarima as pm 

@st.cache_data
def load_data(file_path):
    """Loads and prepares the rainfall data."""
    try:
        df = pd.read_csv(file_path)
        df['YEAR'] = df['YEAR'].astype(int)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The data file {file_path} was not found.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the data: {e}")

def create_lagged_features(df, target_column, n_lags=3):
    """Creates lagged features for a time series DataFrame."""
    df_lagged = df.copy()
    for lag in range(1, n_lags + 1):
        df_lagged[f'{target_column}_lag_{lag}'] = df_lagged[target_column].shift(lag)
    df_lagged = df_lagged.dropna() 
    return df_lagged


@st.cache_data(show_spinner="⏳ Training Random Forest model and generating predictions...")
def train_and_forecast_random_forest(historical_data_df, target_col_name, num_lags, num_future_years): # Renamed for clarity
    """
    Trains a RandomForestRegressor model and performs autoregressive forecasting.   
    """
    lagged_df = create_lagged_features(historical_data_df[['YEAR', target_col_name]], target_col_name, num_lags)
    
    if lagged_df.empty or len(lagged_df) < num_lags + 5:
        return None, None, "Not enough historical data (after creating lags) for Random Forest model."

    feature_cols = [f'{target_col_name}_lag_{lag}' for lag in range(1, num_lags + 1)]
    X = lagged_df[feature_cols]
    y = lagged_df[target_col_name]

    model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1, oob_score=True)
    model.fit(X, y)
    
    forecast_predictions = []
    last_known_lags_array = list(historical_data_df[target_col_name].iloc[-num_lags:].astype(float).values)
    current_forecast_year = historical_data_df['YEAR'].max()
    forecasted_years_list = []

    for i in range(num_future_years):
        input_features_for_pred = np.array(last_known_lags_array[::-1]).reshape(1, -1) 
        predicted_value = model.predict(input_features_for_pred)[0]
        forecast_predictions.append(predicted_value)
        current_forecast_year += 1
        forecasted_years_list.append(current_forecast_year)
        last_known_lags_array.pop(0)
        last_known_lags_array.append(predicted_value) 
        
    forecast_results_df = pd.DataFrame({'YEAR': forecasted_years_list, f'Forecasted_{target_col_name}': forecast_predictions})
    return model, forecast_results_df, None


@st.cache_data(show_spinner="⏳ Fitting Auto-ARIMA model and generating predictions...")
def train_and_forecast_auto_arima(historical_data_df, target_col_name, num_future_years):

    if historical_data_df.empty or len(historical_data_df) < 10: 
        return None, None, "Not enough historical data"



    train_series = historical_data_df[target_col_name]

    try:
        auto_arima_model = pm.auto_arima(
            train_series,
            start_p=1, start_q=1,
            test='adf',      
            max_p=5, max_q=5, 
            m=1,              
            d=None,          
            seasonal=False,   
            start_P=0, D=0,   
            trace=False,      
            error_action='ignore',  
            suppress_warnings=True, 
            stepwise=True    
        )
        forecast_values, conf_int = auto_arima_model.predict(n_periods=num_future_years, return_conf_int=True)
        
        last_year = historical_data_df['YEAR'].max()
        forecast_years = [last_year + i + 1 for i in range(num_future_years)]
        
        forecast_df = pd.DataFrame({
            'YEAR': forecast_years, 
            f'Forecasted_{target_col_name}': forecast_values,
            'Conf_Int_Lower': conf_int[:, 0],
            'Conf_Int_Upper': conf_int[:, 1]
        })
        
        return auto_arima_model, forecast_df, None
    except Exception as e:
        return None, None, f"Error during Auto-ARIMA model fitting or forecasting: {str(e)}"