import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm  # For prediction intervals
import requests
import time
from tqdm import tqdm  # For progress bars
import json  # Import json for pretty printing params in fetch_data
from sklearn.ensemble import RandomForestRegressor  # Import Random Forest

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


import requests
import time
import datetime as dt
from config import Config

# Set plot style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 7)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["figure.titlesize"] = 16

NASS_API_KEY = Config.USDA.PARAMS["key"]
BASE_URL = Config.USDA.BASE_URL
COMMON_PARAMS = Config.USDA.PARAMS

print("All necessary libraries imported successfully!")

def get_weather():
    # Load given weather data (hist_wx_df.parquet)
    weather = pd.read_parquet("./data/hist_wx_df.parquet")

    # Rename certain columns for better understanding
    weather = weather.rename(
        columns={"date": "Date", "adm2_name": "County", "adm1_name": "State"}
    )

    # Create a 'Year' column on weather data to do an aggregation
    weather["Year"] = weather["Date"].dt.year

    # --- Create the State-County Dictionary ---
    # Initialize an empty dictionary to store the results
    state_counties_dict = {}

    # Iterate through each unique state in the 'adm1_name' column
    for state in weather["State"].unique():
        # Filter the DataFrame for the current state
        state_df = weather[weather["State"] == state]
        # Get the unique county names for this state
        unique_counties = state_df["County"].unique().tolist()
        # Add the state and its list of counties to the dictionary
        state_counties_dict[state] = unique_counties

    print("\n--- Generated State-County Dictionary ---")
    # Print first three counties in the generated dictionary for each state

    for state, counties in state_counties_dict.items():
        short_list = counties[:3] + ["..."]
        print(f"'{state}': {short_list}")

    # Filter weather data for the growing season
    wx_gs_df = weather[
        (weather["Date"].dt.month >= Config.Crop.GROWING_SEASON_START_MONTH)
        & (weather["Date"].dt.month <= Config.Crop.GROWING_SEASON_END_MONTH)
        ].copy()  # Use .copy() to avoid SettingWithCopyWarning

    # Calculate daily GDD and Heat Stress indicators
    wx_gs_df["GDD"] = calculate_gdd(wx_gs_df["tmax"], wx_gs_df["tmin"])
    wx_gs_df["Heat_Stress"] = calculate_heat_stress_days(wx_gs_df["tmax"])

    # Calculate per state features
    wx_gs_df = wx_gs_df.groupby(["Date", "State"]).mean(numeric_only=True).reset_index()
    wx_gs_df["Month"] = wx_gs_df["Date"].dt.month

    wx_gs_df["Year"] = wx_gs_df["Date"].dt.year

    modeling_df = pd.melt(
        wx_gs_df,
        id_vars=["Date", "State", "Year", "Month"],  # keep these as identifiers
        value_vars=[
            "tmax",
            "tmin",
            "precip",
            "swvl1",
            "swvl2",
            "GDD",
        ],  # melt these into 'Type'
        var_name="Type",
        value_name="Value",
    )
    modeling_df = (
        modeling_df.groupby(["State", "Year", "Month", "Type"])["Value"]
        .mean()
        .reset_index()
    )

    modeling_df["Feature_Month"] = (
            modeling_df["Type"] + "_" + modeling_df["Month"].astype(str)
    )

    modeling_df = modeling_df[["State", "Year", "Feature_Month", "Value"]]

    modeling_df = modeling_df.pivot(
        index=["State", "Year"], columns="Feature_Month", values="Value"
    ).reset_index()

    modeling_df.columns.name = None

    return modeling_df, state_counties_dict


def fetch_data(params, max_retries=5, backoff_factor=3.0):
    """Fetches data from NASS API with retry logic."""
    if not NASS_API_KEY:
        print("API key not provided. Skipping API call.")
        return []

    for attempt in range(max_retries):
        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            time.sleep(2)
            data = response.json()
            if "data" in data:
                return data["data"]
            else:
                print(
                    f"Warning: 'data' key not found in response for params: {params}. Response: {data}"
                )
                return []
        except requests.exceptions.HTTPError as e:
            # ONLY print the error message if it's a 403, to avoid spamming for other errors
            if e.response.status_code == 403:
                print(
                    f"HTTP Error: {e} for url: {response.url} for params: {params}"
                )  # Added for debugging 403
                sleep_time = backoff_factor * (2**attempt)
                time.sleep(sleep_time)
            if e.response.status_code == 429:  # Too Many Requests
                sleep_time = backoff_factor * (2**attempt)
                # print(f"Rate limit hit. Retrying in {sleep_time:.2f} seconds (attempt {attempt + 1}/{max_retries})...")
                time.sleep(sleep_time)
            else:
                # print(f"Non-retryable HTTP error. Skipping: {e}")
                pass  # Suppress other HTTP errors from printing repeatedly
            return []  # Return empty list on non-retryable or max retries
        except requests.exceptions.ConnectionError as e:
            print(f"Connection Error: {e} for params: {params}")
            sleep_time = backoff_factor * (2**attempt)
            print(
                f"Connection error. Retrying in {sleep_time:.2f} seconds (attempt {attempt + 1}/{max_retries})..."
            )
            time.sleep(sleep_time)
        except requests.exceptions.Timeout as e:
            print(f"Timeout Error: {e} for params: {params}")
            sleep_time = backoff_factor * (2**attempt)
            print(
                f"Timeout. Retrying in {sleep_time:.2f} seconds (attempt {attempt + 1}/{max_retries})..."
            )
            time.sleep(sleep_time)
        except Exception as e:
            print(f"An unexpected error occurred: {e} for params: {params}")
            return []
    print(f"Failed to fetch data after {max_retries} attempts for params: {params}")
    return []

def fetch_state_yield(state_counties_dict, DATA_YEARS, fig_plot=True):
    print("\nAttempting to fetch State Level Corn Yield Data from USDA NASS API...")
    state_data_list = []

    if NASS_API_KEY:  # Only attempt if API key is provided
        for state in tqdm(state_counties_dict.keys(), desc="Fetching State Data by State"):
            state_params = COMMON_PARAMS.copy()
            state_params["agg_level_desc"] = "STATE"
            state_params["state_name"] = state.upper()  # Specify state explicitly
            state_params["year__GE"] = min(DATA_YEARS)
            state_params["year__LE"] = max(DATA_YEARS)

            state_data = fetch_data(state_params)
            if state_data:
                state_data_list.extend(state_data)
            time.sleep(0.8)  # Be polite to the API
    else:
        print("API key not provided. Skipping state-level data fetch from API.")

    us_state_yield_df = pd.DataFrame(state_data_list)
    if not us_state_yield_df.empty:
        # Select relevant columns and rename for consistency
        us_state_yield_df = us_state_yield_df[
            ["year", "state_name", "Value", "reference_period_desc"]
        ].rename(columns={"year": "Year", "state_name": "State", "Value": "Yield_bu_acre"})
        us_state_yield_df["Year"] = pd.to_numeric(us_state_yield_df["Year"])
        us_state_yield_df["Yield_bu_acre"] = pd.to_numeric(
            us_state_yield_df["Yield_bu_acre"], errors="coerce"
        )
        us_state_yield_df.dropna(subset=["Yield_bu_acre"], inplace=True)
        us_state_yield_df["State"] = us_state_yield_df["State"].str.title()
        print("Successfully fetched US State Yield Data from API.")
        print("US State Yield Data (first 5 rows):")
        print(us_state_yield_df.head())
        print("US State Yield Data (last 5 rows):")
        print(us_state_yield_df.tail())
    else:
        print(
            "No state level data fetched from API (API key missing/invalid or error occurred). State yield dataframe will be empty."
        )

    if fig_plot:
        plot_yield(us_state_yield_df, DATA_YEARS)
    return us_state_yield_df

def plot_yield(us_state_yield_df, DATA_YEARS):
    # --- Yield Trends Over Time (State Level) ---
    if not us_state_yield_df.empty:
        plt.figure(figsize=(12, 7))
        sns.lineplot(
            data=us_state_yield_df[us_state_yield_df["reference_period_desc"] == "YEAR"],
            x="Year",
            y="Yield_bu_acre",
            hue="State",
            marker="o",
        )
        plt.title(f"U.S. State Corn Yield Over Time ({min(DATA_YEARS)}-{max(DATA_YEARS)})")
        plt.xlabel("Year")
        plt.ylabel("Yield (bushels/acre)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("No state yield data available for plotting trends.")

def calculate_gdd(tmax, tmin, t_base=10, t_cap=30):
    # Cap TMAX at T_CAP
    tmax_capped = np.minimum(tmax, t_cap)
    # Calculate average daily temperature
    t_avg = (tmax_capped + tmin) / 2
    # Set T_AVG to T_BASE if it falls below T_BASE
    gdd = np.maximum(0, t_avg - t_base)
    return gdd

# Function to calculate Heat Stress Days
def calculate_heat_stress_days(tmax, threshold=32):  # Days above 32C
    return (tmax > threshold).astype(int)

def plot_correlation_matrix(data):
    correlation_matrix = data.corr(numeric_only=True)
    plt.figure(figsize=(15, 12))
    sns.heatmap(correlation_matrix, cmap="coolwarm", square=True)
    plt.title("Correlation Matrix of Features", fontsize=14)
    plt.tight_layout()
    plt.show()

    return correlation_matrix

def fill_NaN(modeling_df):
    # --- To fill NaNs with historical averages ---
    print(
        "Filling in missing 2024 monthly weather data with historical averages for 'modeling_df'..."
    )

    # Identify all columns that represent monthly features
    monthly_feature = [
        col
        for col in modeling_df.columns
        if (
                "_" in col
                and col.split("_")[0] in ["precip", "tmax", "tmin", "GDD", "swvl1", "swvl2"]
        )
    ]

    # Calculate the historical average for each feature, grouped by state
    historical_averages = (
        modeling_df[modeling_df["Year"] < 2024].groupby("State")[monthly_feature].mean()
    )

    # Iterate through each state and fill the missing 2024 values
    states_with_data = modeling_df["State"].unique()
    for state in states_with_data:
        # Get the index for the 2024 row for the current state
        idx_2024 = modeling_df[
            (modeling_df["Year"] == 2024) & (modeling_df["State"] == state)
            ].index

        if not idx_2024.empty:
            # Get the historical average for this state
            state_avg = historical_averages.loc[state]

            # Use .fillna() to replace NaN values in the 2024 row with the historical averages
            modeling_df.loc[idx_2024, monthly_feature] = modeling_df.loc[
                idx_2024, monthly_feature
            ].fillna(state_avg)

    print("Missing 2024 weather data filled successfully with historical averages.")
    return modeling_df

def model(modeling_df, train_years_end, top_corr):
    features = []

    # Select the most correlated features to the yield
    monthly_features = top_corr.index.tolist()[0:10]
    features.extend(monthly_features)

    target = "Yield_bu_acre"

    X = modeling_df[features]
    y = modeling_df[target]


    X_train = X[X["Year"] <= train_years_end]
    y_train = y[X["Year"] <= train_years_end]

    X_val = X[X["Year"] > train_years_end]
    y_val = y[X["Year"] > train_years_end]

    print(f"Training data years: {X_train['Year'].min()} - {X_train['Year'].max()}")
    print(f"Validation data years: {X_val['Year'].min()} - {X_val['Year'].max()}")
    print(f"Number of training records: {len(X_train)}")
    print(f"Number of validation records: {len(X_val)}")

    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)

    print("\nModel trained successfully!")

    y_pred_val = model.predict(X_val)

    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    mae_val = mean_absolute_error(y_val, y_pred_val)
    r2_val = r2_score(y_val, y_pred_val)

    print(
        f"\nModel Performance on Validation Set (Years {X_val['Year'].min()}-{X_val['Year'].max()}):"
    )
    print(f"RMSE: {rmse_val:.2f} bushels/acre")
    print(f"MAE: {mae_val:.2f} bushels/acre")
    print(f"R-squared: {r2_val:.2f}")

    plt.figure(figsize=(15, 6))

    # Plot training data
    plt.plot(X_train["Year"], y_train, label="Actual (Train)", color="blue", marker="o")
    plt.plot(
        X_train["Year"],
        model.predict(X_train),
        "--",
        label="Predicted (Train)",
        color="skyblue",
    )

    # Plot validation/test data
    plt.plot(X_val["Year"], y_val, label="Actual (Test)", color="green", marker="s")
    plt.plot(
        X_val["Year"],
        model.predict(X_val),
        "--",
        label="Predicted (Test)",
        color="limegreen",
    )

    # Labels and title
    plt.title("Actual vs Predicted Crop Yields Over Time", fontsize=16)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Yield (bushels/acre)", fontsize=12)

    # Grid, legend, and layout
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best", fontsize=11)
    plt.tight_layout()
    plt.show()

    # --- Feature Importance Analysis (Random Forest) ---
    print("\n--- Feature Importance Analysis (Random Forest) ---")
    # Feature importance is a powerful tool to understand which features contribute most to the model's predictions.
    # For tree-based models like Random Forest, a feature's importance is calculated as the total reduction in the
    # criterion (e.g., mean squared error) brought by that feature across all trees in the forest.

    # Get feature importances from the best Random Forest model
    importances = model.feature_importances_
    feature_names = X.columns
    feature_importances_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    )

    # Sort the features by importance in descending order
    feature_importances_df = feature_importances_df.sort_values(
        by="Importance", ascending=False
    )

    # Plot the top 20 most important features
    plt.figure(figsize=(12, 5))
    sns.barplot(x="Importance", y="Feature", data=feature_importances_df.head(20))
    plt.title("Top 20 Feature Importances (Random Forest)")
    plt.xlabel("Relative Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    print("Using the Random Forest model to predict corn yield for the year 2024.")

    # Filter the validation data for the year 2024
    X_2024_val = X_val[X_val["Year"] == 2024]
    y_2024_val = y_val[X_val["Year"] == 2024]

    # Predict on the 2024 data
    y_pred_2024 = model.predict(X_2024_val)

    # Create a DataFrame to show actual vs. predicted for 2024
    results_2024 = X_2024_val.copy()
    results_2024["Predicted_Yield"] = y_pred_2024
    results_2024["Actual_Yield"] = y_2024_val.values
    results_2024["State"] = modeling_df[modeling_df["Year"] == 2024]["State"]

    print("\n2024 U.S. Corn Yield Predictions vs. Actual (State Level):")
    print(results_2024[["State", "Actual_Yield", "Predicted_Yield"]].to_string(index=False))

    rmse_2024 = np.sqrt(mean_squared_error(y_2024_val, y_pred_2024))
    mae_2024 = mean_absolute_error(y_2024_val, y_pred_2024)
    r2_2024 = r2_score(y_2024_val, y_pred_2024)

    print(f"\nModel Performance on 2024 Data:")
    print(f"RMSE: {rmse_2024:.2f} bushels/acre")
    print(f"MAE: {mae_2024:.2f} bushels/acre")
    print(f"R-squared: {r2_2024:.2f}")

    plt.figure(figsize=(12, 6))

    # Plot actual and predicted yields
    plt.plot(
        results_2024["State"],
        results_2024["Actual_Yield"],
        marker="o",
        label="Actual Yield",
        color="darkgreen",
    )
    plt.plot(
        results_2024["State"],
        results_2024["Predicted_Yield"],
        marker="s",
        label="Predicted Yield",
        color="orange",
    )

    # Title and labels
    plt.title("Actual vs Predicted Corn Yield by State (2024)", fontsize=14)
    plt.xlabel("States", fontsize=12)
    plt.ylabel("Yield (bushels/acre)", fontsize=12)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha="right")

    # Add grid and legend
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", fontsize=11)

    # Tight layout for better spacing
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # Get the weather data
    weather_df, state_counties_dict  = get_weather()

    DATA_YEARS = list(range(weather_df["Year"].min(), weather_df["Year"].max()))

    print(DATA_YEARS)
    # Get yield data
    yield_df = fetch_state_yield(state_counties_dict, DATA_YEARS)

    modeling_df = pd.merge(
        weather_df,
        yield_df[yield_df["reference_period_desc"] == "YEAR"][
            ["Year", "State", "Yield_bu_acre"]
        ],
        on=["Year", "State"],
        how="inner",
    )

    print(
        "\nEngineered Features (first 5 rows of state-level modeling data with monthly features):"
    )
    print(modeling_df.head())

    print(
        f"The feature name indicate the average feature of xth month in the format :'feature_month'"
    )
    print(f"Shape of modeling dataframe: {modeling_df.shape}")
    print(
        "State-level feature engineering with monthly aggregates complete. Merged with state yield data."
    )

    # Plot correlation matrix
    correlation_matrix = plot_correlation_matrix(modeling_df)

    # Extract correlations with 'Yield_bu_acre'
    yield_corr = correlation_matrix["Yield_bu_acre"].drop("Yield_bu_acre")

    # Sort by absolute correlation
    top_corr = yield_corr.abs().sort_values(ascending=False)

    # Filling the missing NaN with average
    modeling_df = fill_NaN(modeling_df)

    # Model development
    train_years_end = Config.Model.train_years_end
    model(modeling_df, train_years_end, top_corr)








