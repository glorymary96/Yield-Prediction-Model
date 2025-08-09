# 2025 U.S. Corn Yield Prediction

## 1. Project Summary & Objective
The objective of this project was to predict the 2025 U.S. corn yield at the state level using a comprehensive set of monthly features, including:
- Growing Degree Days (GDD)
- Precipitation
- Temperature extremes
- Soil moisture

## 2. Methodology

### Data Acquisition & Strategy
- Historical yield data sourced from the USDA NASS API.
- Succesfully implemented national, state and county level yield data.
- Daily county-level weather data loaded from `hist_wx_df.parquet`.

**Challenge:** API record limits made county-level data fetching difficult. 
**Solution:** An iterative strategy was implemented to fetch data per state and county, with automatic retries for failed requests. This code is commented in the python notebook. But to check, please uncomment and run it. While implementing it, its highly advisable to save this data into a database, to avoid multiple API fetches in the future.

### Assumptions
- GDD, and Heat Stress were simulated due to absence in the original weather file.
- These simulations are necessary but introduce limitations.

### Feature Engineering
Monthly, state-level features aggregated from daily weather data:
- Growing Degree Days (GDD)
- Precipitation (`precip`)
- Max/Min Temperatures (`tmax`, `tmin`)
- Heat Stress (days with TMAX > 32°C)
- Soil Moisture (`swvl1`, `swvl2`)

### Model Development & Evaluation
- Final model: `RandomForestRegressor(n_estimators=50)`
- Training period: 2000–2022
- Validation period: 2023–2024
- A USDA model from [this report](https://ers.usda.gov/sites/default/files/_laserfiche/outlooks/36651/39297_fds-13g-01.pdf?v=99616) was also tested, but Random Forest performed better.

## 3. Results & Predictions

### Final Model Performance (Validation: 2023–2024)
- **RMSE:** 15.61 bushels/acre 
- **MAE:** 12.95 bushels/acre 
- **R²:** 0.54 (explains 54% of yield variance)

### 2024 Yield Predictions vs Actuals

| State        | Predicted | Actual |
|--------------|-----------|--------|
| Illinois     | 191.66    | 217.00 |
| Indiana      | 188.30    | 198.00 |
| Iowa         | 189.14    | 211.00 |
| Kansas       | 134.92    | 129.00 |
| Kentucky     | 173.98    | 178.00 |
| Michigan     | 171.72    | 181.00 |
| Minnesota    | 172.76    | 174.00 |
| Missouri     | 176.84    | 183.00 |
| Nebraska     | 156.78    | 188.00 |
| Ohio         | 187.72    | 177.00 |
| South Dakota | 178.68    | 164.00 |
| Tennessee    | 170.26    | 152.00 |
| Wisconsin    | 175.20    | 174.00 |

## 4. Assumptions & Limitations
- No major geological changes assumed.
- Simulated features introduce uncertainty.
- Feature engineering focused on linear relationships.
- State-level granularity; county-level data could improve accuracy.

## 5. Potential Improvements
- Feature Importance: Use SHAP or SHAPIQ for interpretability.
- Advanced Models: Try Gradient Boosting or LSTM.
- Additional Features: Include drought indices and crop-stage weather impacts.

---

Note: The USDA model from [FDS-13g-01](https://ers.usda.gov/sites/default/files/_laserfiche/outlooks/36651/39297_fds-13g-01.pdf?v=99616) was implemented for comparison. While conceptually strong, the Random Forest model demonstrated superior predictive performance.
