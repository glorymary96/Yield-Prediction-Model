# U.S. Corn Yield Prediction

 **Note**: This is a ongoing project. The progress will be updated here. 

## 1. Project Summary & Objective
This repository contains the code and analysis for a machine learning model designed to predict **U.S. corn yield** at the state and county level. The model utilizes comprehensive meteorological and soil features to forecast yields, with a focus on capturing the impact of growing conditions.

The primary objective of this project is to develop and evaluate a robust machine learning model capable of accurately predicting **U.S. corn yield per acre** at the state/county level for the upcoming season.
The model leverages monthly/weekly aggregated features derived from daily weather data to capture critical agricultural stressors and growth drivers.


## 2. Methodology

### Data Acquisition & Strategy
- Historical yield data sourced from the USDA NASS API.
- Succesfully implemented national, state and county level yield data.
- Daily county-level weather data loaded from `hist_wx_df.parquet`.

### Assumptions
- GDD, and Heat Stress were simulated due to absence in the original weather file.
- These simulations are necessary but introduce limitations.

### Feature Engineering
Monthly, state-level features aggregated from daily weather data:
- Growing Degree Days (`GDD`)
- Precipitation (`precip`)
- Cumulative precipitation  for 7, 15, and 30 days (`CP7D`, `CP15D`, `CP30D`)
- Square of 7 days cumulative precipitation (`CP7DS`)
- Max/Min Temperatures (`tmax`, `tmin`)
- Average Temperature (`TAVG`)
- Heat Stress (days with TMAX > 32°C)
- Soil Moisture (`swvl1`, `swvl2`)
- Average Soil Moisture of `swvl1` and `swvl2`

### Model Development & Evaluation
- Models: 
  - `RandomForestRegressor()`
  - `LSTM Model`
  - A USDA model from [this report](https://ers.usda.gov/sites/default/files/_laserfiche/outlooks/36651/39297_fds-13g-01.pdf?v=99616).

- Training period: 2000–2023
- Validation period: 2024

## 3. Results & Predictions

### Final Model Performance from the best performing model (Validation: 2024)
- **RMSE:** 12.42 bushels/acre 
- **MAE:** 10.20 bushels/acre 
- **R²:** 0.69 (explains approx. 70% of yield variance)


## 4. Assumptions & Limitations
- No major geological changes assumed.
- Simulated features introduce uncertainty.
- Feature engineering focused on linear relationships.

## 5. Potential Improvements
- Feature Importance: Use SHAP or SHAPIQ for interpretability.
- Advanced Models: Try Gradient Boosting.
- Additional Features: Include drought indices and crop-stage weather impacts.

---

Note: The USDA model from [FDS-13g-01](https://ers.usda.gov/sites/default/files/_laserfiche/outlooks/36651/39297_fds-13g-01.pdf?v=99616) was implemented for comparison. While conceptually strong, the Random Forest model demonstrated superior predictive performance.

---
## 🚀 Getting Started

### Prerequisites

You need **Python** installed. All necessary libraries are listed in `requirements.txt`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/glorymary96/Yield-Prediction-Model.git](https://github.com/glorymary96/Yield-Prediction-Model.git)
    cd Yield-Prediction-Model
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # .\venv\Scripts\activate # On Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

The core project workflow is contained within the following files:

| File Name | Description |
| :--- | :--- |
| `main.py` | Main script to run the entire data fetching, preparation, and modeling pipeline. |
| `Fetch_Yield.py` | Handles the acquisition of historical yield data from the USDA NASS API. |
| `prep_data.py` | Contains scripts for cleaning, transforming, and engineering features from raw data. |
| `corn_yield_prediction.ipynb` | Jupyter Notebook containing the main exploratory data analysis, feature engineering, and model training workflow at the state level. |
| `corn_yield_pred_counties.ipynb` | Jupyter Notebook focusing on county-level yield prediction analysis. |
| `rnn_modeling.py` | Script for exploring Recurrent Neural Network (RNN) models for time series prediction. |

