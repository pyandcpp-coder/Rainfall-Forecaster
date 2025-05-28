# Indian Monsoon Rainfall Analysis and Forecasting Tool 🌧️

This interactive web application provides analysis of historical Indian monsoon rainfall data (June-September) for various meteorological subdivisions and generates future rainfall forecasts using machine learning (Random Forest) and statistical (Auto-ARIMA) models.

**Live Application:** [https://rainfall-forecaster.streamlit.app/](https://rainfall-forecaster.streamlit.app/)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Data Source](#data-source)
- [Methodology](#methodology)
  - [Historical Analysis](#historical-analysis)
  - [Forecasting Models](#forecasting-models)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Setup and Usage](#setup-and-usage)
- [Screenshots](#screenshots)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

The Indian monsoon is a critical climatic phenomenon, and understanding its patterns and predictability is vital. This tool was developed as a personal learning project to explore the application of data analysis and AI/ML techniques to meteorological datasets. It allows users to:
*   Visualize historical rainfall trends for different Indian subdivisions.
*   Examine rainfall distributions and key statistics.
*   Generate short-term forecasts for total monsoon rainfall using different predictive models.

This project was inspired by the challenges and advancements in Numerical Weather Prediction (NWP) and the potential of AI/ML to enhance forecasting capabilities.

## Features

*   **Interactive Data Exploration:**
    *   Select from various Indian meteorological subdivisions.
    *   Filter historical data by a specific year range.
    *   View time series plots for total monsoon (JUN-SEP) and monthly (JUN, JUL, AUG, SEP) rainfall.
    *   Analyze rainfall distributions using box plots and histograms.
    *   Access summary statistics (mean, median, std. dev., min, max) for selected data.
*   **Rainfall Forecasting:**
    *   Choose between two forecasting models:
        1.  **Random Forest Regressor:** A machine learning model using lagged rainfall values as features.
        2.  **Auto-ARIMA:** A statistical model that automatically determines the best ARIMA(p,d,q) orders.
    *   Specify the number of future years to forecast.
    *   For Random Forest, configure the number of past years (lags) to use as predictors.
    *   View forecasted values in a table and plotted alongside historical data.
    *   For Auto-ARIMA, view confidence intervals for forecasts.
*   **Model Details:**
    *   View key parameters and metrics for the selected forecasting model (e.g., OOB score for Random Forest, ARIMA order for Auto-ARIMA).
*   **User-Friendly Interface:** Built with Streamlit for easy interaction.

## Data Source

*   **Dataset:** Sub-divisional monsoon rainfall data for India.
*   **Source:** Aggregated from India Meteorological Department (IMD) historical data archives.
*   **Period Covered:** 1901-2021.
*   **Parameters Used:** Monthly rainfall for June, July, August, September, and the total June-September (JUN-SEP) monsoon rainfall for 36 meteorological subdivisions.
*   The data file (`data.csv`) is included in this repository.

## Methodology

### Historical Analysis
Historical rainfall data is analyzed using Pandas for data manipulation. Interactive visualizations are generated using Plotly Express, including:
*   Line charts for rainfall trends over time.
*   Box plots to show the distribution and variability of rainfall for each monsoon month and the total season.
*   Histograms to display the frequency distribution of total monsoon rainfall.

### Forecasting Models

#### 1. Random Forest Regressor
*   An ensemble machine learning model from `scikit-learn`.
*   **Features:** Lagged values of past 'JUN-SEP' total monsoon rainfall (e.g., rainfall from year T-1, T-2, ..., T-n_lags). The number of lags is user-configurable.
*   **Process:** The model is trained on the complete historical data of the selected subdivision. It then predicts future rainfall autoregressively (using its own past predictions as input for subsequent years' forecasts).

#### 2. Auto-ARIMA
*   An automated approach to fit Autoregressive Integrated Moving Average (ARIMA) models using the `pmdarima` library.
*   **Process:** `pmdarima.auto_arima` automatically searches for the optimal non-seasonal ARIMA(p,d,q) order based on information criteria (e.g., AIC).
*   The model is fitted to the historical 'JUN-SEP' rainfall series for the selected subdivision.
*   Forecasts are generated along with 95% confidence intervals.

## Technologies Used

*   **Python:** Core programming language.
*   **Streamlit:** For building the interactive web application.
*   **Pandas:** For data manipulation and analysis.
*   **NumPy:** For numerical operations.
*   **Scikit-learn:** For the Random Forest machine learning model.
*   **Pmdarima:** For the Auto-ARIMA forecasting model.
*   **Plotly & Plotly Express:** For creating interactive charts and graphs.
*   **Git & GitHub:** For version control and hosting.

## Project Structure

## Setup and Usage

To run this application locally:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/pyandcpp-coder/Rainfall-Forecaster.git
    cd Rainfall-Forecaster
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    The application should open in your default web browser.

## Future Work

Potential enhancements for this project could include:
*   Incorporating more sophisticated time series models (e.g., SARIMAX, Prophet with exogenous variables).
*   Adding more detailed statistical tests for stationarity, seasonality, etc.
*   Allowing users to upload their own time series data.
*   Implementing model evaluation metrics (e.g., RMSE, MAE) on a hold-out set.
*   Exploring the integration of climate indices (e.g., ENSO) as exogenous variables for forecasting.
*   Improving UI/UX and adding more customization options for plots.

## Contributing
This is a personal project, but feedback and suggestions are welcome. Please feel free to open an issue or submit a pull request if you have ideas for improvements.

## Contact
Yash Tiwari
*   Email: yashtiwari9182@gmail.com
*   LinkedIn: [https://www.linkedin.com/in/yrevash/](https://www.linkedin.com/in/yrevash/)
*   GitHub: [https://github.com/pyandcpp-coder](https://github.com/pyandcpp-coder)
