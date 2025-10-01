import pandas as pd
import requests
import time
from tqdm import tqdm
import datetime as dt
import numpy as np

from functools import lru_cache
from config import Config

import logging

# Get a logger instance for this module
# This logger automatically inherits the configuration from main.py
logger = logging.getLogger(__name__)


# --- Data Loading Stage ---
def load_data(filepath: str) -> pd.DataFrame:
    """Loads data from a given file path, handling file-not-found errors."""
    try:
        df = pd.read_parquet(filepath)
        logger.info(f"Successfully loaded data from {filepath}.")
        return df
    except FileNotFoundError:
        logger.error(f"File not found at {filepath}. Please check the path.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"An error occurred while loading data: {e}")
        return pd.DataFrame()


# --- Feature Engineering Stage ---
def _calculate_gdd(
    tmax: pd.Series, tmin: pd.Series, t_base: int = 10, t_cap: int = 30
) -> pd.Series:
    """Calculates Growing Degree Days (GDD) for a series."""
    tmax_capped = tmax.clip(upper=t_cap)
    t_avg = (tmax_capped + tmin) / 2
    gdd = (t_avg - t_base).clip(lower=0)
    return gdd


def _calculate_heat_stress_days(tmax: pd.Series, threshold: int = 32) -> pd.Series:
    """Calculates Heat Stress Days for a series."""
    return (tmax > threshold).astype(int)


def process_weather_data(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw daily weather data into a monthly, state-level feature set.
    This function encapsulates all weather-related feature engineering.
    """
    if weather_df.empty:
        logger.warning("Input weather dataframe is empty. Skipping processing.")
        return pd.DataFrame()

    logger.info("Starting weather data processing and feature engineering...")

    # Standardize column names
    df = weather_df.rename(
        columns={"date": "Date", "adm2_name": "County", "adm1_name": "State"}
    ).copy()
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    # --- Find max date ---
    max_date = df["Date"].max()
    max_year = df["Date"].dt.year.max()  # scalar integer

    # --- Get states and counties present on max date ---
    latest_states_counties = df[df["Date"] == max_date][["State", "County"]].drop_duplicates()

    full_dates = pd.date_range(start=max_date + pd.Timedelta(days=1),
                               end=pd.Timestamp(year=max_year, month=12, day=31))

    DF = []
    for date in full_dates:
        temp_df = latest_states_counties.reset_index(drop=True)
        temp_df['Date'] = date
        DF.append(temp_df)
    DF = pd.concat(DF).reset_index(drop=True)

    DF['Month'] = DF['Date'].dt.month
    DF['Day'] = DF['Date'].dt.day

    # --- Merge average values by State, County, Month, Day ---
    # Compute averages
    avg_df = df.groupby(['State', 'County', 'Month', 'Day'], as_index=False)[
        ['tmax', 'tmin', 'precip', 'swvl1', 'swvl2']
    ].mean()

    # Merge averages into DF
    DF = DF.merge(avg_df, on=['State', 'County', 'Month', 'Day'], how='inner')

    df = pd.concat([df, DF[["Date", "State", "County", "tmax", "tmin", "precip", "swvl1", "swvl2"]]])

    # Calculate GDD and Heat Stress
    df["GDD"] = _calculate_gdd(df["tmax"], df["tmin"])
    df["Heat_Stress"] = _calculate_heat_stress_days(df["tmax"])

    logger.info("Weather data processing complete. Created a wide-format feature set.")
    return df


# --- External Data Fetching Stage (API) ---
def _fetch_data_with_retries(
    params: dict, max_retries: int = 5, backoff_factor: float = 3.0
) -> list:
    """Robust function to fetch data from an API with retry logic and logging."""
    if not Config.USDA.PARAMS["key"]:
        logger.warning("API key not provided. Skipping API call.")
        return []

    for attempt in range(max_retries):
        try:
            response = requests.get(Config.USDA.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            if "data" in data:
                return data["data"]
            else:
                logger.warning(
                    f"'data' key not found in response. Response keys: {data.keys()}"
                )
                return []
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            if status_code in [429, 403, 500, 502, 503, 504]:
                sleep_time = backoff_factor * (2**attempt)
                logger.warning(
                    f"HTTP Error {status_code}. Retrying in {sleep_time:.2f}s (attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(sleep_time)
            else:
                logger.error(f"Non-retryable HTTP Error: {e} for params: {params}")
                return []
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            sleep_time = backoff_factor * (2**attempt)
            logger.warning(
                f"{type(e).__name__}. Retrying in {sleep_time:.2f}s (attempt {attempt + 1}/{max_retries})..."
            )
            time.sleep(sleep_time)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e} for params: {params}")
            return []

    logger.error(f"Failed to fetch data after {max_retries} attempts.")
    return []


@lru_cache(maxsize=32)
def get_state_yield_data(states: tuple, years: tuple) -> pd.DataFrame:
    """Fetches state-level corn yield data from the USDA NASS API with caching."""
    logger.info("Attempting to fetch state-level corn yield data from USDA NASS API...")

    if not Config.USDA.PARAMS["key"]:
        logger.warning("API key not provided. Skipping API data fetch.")
        return pd.DataFrame()

    all_data = []

    for state in tqdm(states, desc="Fetching State Data"):
        params = Config.USDA.PARAMS.copy()
        params.update(
            {
                "agg_level_desc": "STATE",
                "state_name": state.upper(),
                "year__GE": min(years),
                "year__LE": max(years),
            }
        )

        state_data = _fetch_data_with_retries(params)
        if state_data:
            all_data.extend(state_data)
        time.sleep(0.5)

    if not all_data:
        logger.warning("No state-level data fetched from API.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df = df[["year", "state_name", "Value"]].rename(
        columns={"year": "Year", "state_name": "State", "Value": "Yield_bu_acre"}
    )

    df["Year"] = pd.to_numeric(df["Year"])
    df["Yield_bu_acre"] = pd.to_numeric(df["Yield_bu_acre"], errors="coerce")
    df.dropna(subset=["Yield_bu_acre"], inplace=True)
    df["State"] = df["State"].str.title()

    logger.info("Successfully fetched and processed state yield data.")

    return df


# --- Main Orchestration Function (The Pipeline) ---
def create_modeling_dataset() -> pd.DataFrame:
    """
    Main pipeline function that orchestrates the data loading, processing,
    and merging to create a final modeling-ready dataset.
    """
    logger.info("Starting the data preparation pipeline...")

    # Stage 1: Load weather data
    weather_df = load_data(Config.Paths.WEATHER_DATA)

    if weather_df.empty:
        return pd.DataFrame()

    # Get unique states and years from weather data to use for API calls
    unique_states = tuple(weather_df["adm1_name"].unique())
    unique_years = tuple(weather_df["date"].dt.year.unique())

    # Stage 2: Process weather data to create features
    weather_features_df = process_weather_data(weather_df)
    if weather_features_df.empty:
        return pd.DataFrame()

    weather_features_df["Year"] = pd.to_datetime(weather_features_df["Date"]).dt.year
    weather_features_df.drop(["County", "aoi_id"], axis=1, inplace=True)
    weather_features_df = (
        weather_features_df.groupby(["Date", "State"])
        .mean(numeric_only=True)
        .reset_index()
    )

    # Stage 3: Fetch yield data from USDA NASS API
    yield_df = get_state_yield_data(unique_states, unique_years)
    if yield_df.empty:
        return pd.DataFrame()

    # Stage 4: Merge weather features and yield data
    final_df = pd.merge(weather_features_df, yield_df, on=["Year", "State"], how="left")

    missing_yield_count = final_df["Yield_bu_acre"].isnull().sum()

    print(f"\nNumber of rows with missing yield data: {missing_yield_count}")
    logger.info(f"Final modeling dataset created with shape: {final_df.shape}")
    logger.info("Head of the final dataset:")
    logger.info(final_df.head())

    if not final_df.empty:
        # Save the final dataset for later use
        try:
            final_df.to_parquet(Config.Paths.OUTPUT_DATA)
            logger.info(f"Final dataset saved to {Config.Paths.OUTPUT_DATA}")
        except Exception as e:
            logger.error(f"Failed to save final dataset: {e}")

    return final_df
