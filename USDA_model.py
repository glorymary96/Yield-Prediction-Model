import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from prep_data import process_weather_data
from config import Config

def USDA_model_preprocessing(data_path:str):

    try:
        df = pd.read_parquet(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Parquet file not found at {data_path}. Please provide the dataset or adjust the path."
        )

    wx_hist_df = process_weather_data(df)

    # Filter weather data for the growing season
    wx_gs_df = wx_hist_df[
        (wx_hist_df["Date"].dt.month >= Config.Crop.GROWING_SEASON_START_MONTH)
        & (wx_hist_df["Date"].dt.month <= Config.Crop.GROWING_SEASON_END_MONTH)
        ].copy()  # Use .copy() to avoid SettingWithCopyWarning

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

    modeling_df = pd.merge(
        modeling_df,
        wx_hist_df[["Year", "State", "Yield_bu_acre"]],
        on=[ "State", "Year"],
        how="inner",
    )

    print(
        "\nEngineered Features (first 5 rows of state-level modeling data with monthly features):"
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

    print(modeling_df.head())

    return modeling_df

def USDA_model(
        data_path:str
):

    df = USDA_model_preprocessing(data_path)
    print(df.head())
    # Step 1: Create model inputs
    df["Trend"] = df["Year"] - 2000
    df["JulyTemp"] = df["tmax_7"]  # average July temperature
    df["JulyPrecip"] = df["precip_7"]  # total July precipitation
    df["JulyPrecipSq"] = df["JulyPrecip"] ** 2

    # Step 2: Calculate June precipitation shortfall
    june_avg_precip = 4.33  # historical average from paper
    df["JunePrecipShortfall"] = np.where(
        df["precip_6"] < 2.51, june_avg_precip - df["precip_6"], 0
    )

    # Step 3: Estimate planting progress (if not available, assume 80%)
    df["PlantingProgress"] = 80  # percent planted by mid-May

    train_years_end = 2022
    features = [
        "Year",
        "PlantingProgress",
        "JunePrecipShortfall",
        "JulyTemp",
        "JulyPrecip",
        "JulyPrecipSq",
    ]

    X_usda = df[features]
    y_usda = df["Yield_bu_acre"]

    X_train_usda = X_usda[X_usda["Year"] <= train_years_end]
    y_train_usda = y_usda[X_usda["Year"] <= train_years_end]

    X_val_usda = X_usda[X_usda["Year"] > train_years_end]
    y_val_usda = y_usda[X_usda["Year"] > train_years_end]

    model_usda = LinearRegression()
    model_usda.fit(X_train_usda, y_train_usda)

    print("\nModel trained successfully!")

    y_pred_val_usda = model_usda.predict(X_val_usda)

    rmse_val_usda = np.sqrt(mean_squared_error(y_val_usda, y_pred_val_usda))
    mae_val_usda = mean_absolute_error(y_val_usda, y_pred_val_usda)
    r2_val_usda = r2_score(y_val_usda, y_pred_val_usda)

    print(
        f"\nModel Performance on Validation Set (Years {X_val_usda['Year'].min()}-{X_val_usda['Year'].max()}):"
    )
    print(f"RMSE: {rmse_val_usda:.2f} bushels/acre")
    print(f"MAE: {mae_val_usda:.2f} bushels/acre")
    print(f"R-squared: {r2_val_usda:.2f}")

    plt.figure(figsize=(15, 6))

    # Plot training data
    plt.plot(
        X_train_usda["Year"], y_train_usda, label="Actual (Train)", color="blue", marker="o"
    )
    plt.plot(
        X_train_usda["Year"],
        model_usda.predict(X_train_usda),
        "--",
        label="Predicted (Train)",
        color="skyblue",
    )

    # Plot validation/test data
    plt.plot(
        X_val_usda["Year"], y_val_usda, label="Actual (Test)", color="green", marker="s"
    )
    plt.plot(
        X_val_usda["Year"],
        model_usda.predict(X_val_usda),
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

if __name__ == "__main__":
    #USDA_model_preprocessing("./data/final_modeling_data.parquet")
    USDA_model("./data/final_modeling_data.parquet")