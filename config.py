with open("usda_key.txt", "r") as file:
    key = file.read().strip()
    print(f"The key is: {key}")


# --- Configuration (Centralized Settings) ---
class Config:
    """Centralized configuration for the entire application."""

    class USDA:
        # KEY = '088D5072-1953-3D8C-91E3-62A84E0C49C4'
        BASE_URL = "http://quickstats.nass.usda.gov/api/api_GET/"
        PARAMS = {
            "key": key,
            "commodity_desc": "CORN",
            "short_desc": "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE",
            "freq_desc": "ANNUAL",
            "statisticcat_desc": "YIELD",
            "source_desc": "SURVEY",  # 'SURVEY' for annual estimates, 'CENSUS' for census years
            "format": "JSON",
        }

    class Crop:
        GROWING_SEASON_START_MONTH = 4
        GROWING_SEASON_END_MONTH = 11

    class Paths:
        WEATHER_DATA = "./data/hist_wx_df.parquet"
        WEATHER_DF_DATA = "./data//weather_df.parquet"
        OUTPUT_DATA = "./data/final_modeling_data.parquet"
        YIELD_DATA = "./data//yield_data.parquet"

    class Model:
        features = [
            "tmax",
            "tmin",
            "precip",
            "swvl1",
            "swvl2",
            "GDD",
            "Heat_Stress",
            "Year",
        ]
        target_col = "Yield_bu_acre"
