import requests
import pandas as pd
import numpy as np
from config import Config
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("Fetch_Yield").getOrCreate()

# Replace with your own API key from https://quickstats.nass.usda.gov/api
API_KEY = Config.USDA.PARAMS["key"]

# Define API endpoint
BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/"

def get_yield_data(state, year_GE=2000):
    params = {
        "key": API_KEY,
        "year__GE": 2000,
        "state_name": state,
        "agg_level_desc": "COUNTY",        # AGRICULTURAL DISTRICT
        "commodity_desc": "CORN",          # <-- change crop here
        "statisticcat_desc": "YIELD",
        "unit_desc": "BU / ACRE",          # <-- crop-specific unit
        "format": "JSON"
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json().get("data", [])

    return data

if __name__ == "__main__":
    sc = spark.sparkContext

    years = np.arange(2000, 2024, 1)
    States =['Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Michigan', 'Minnesota',
            'Missouri', 'Nebraska', 'Ohio', 'South Dakota', 'Tennessee', 'Wisconsin']

    rdd_data = sc.parallelize(States).flatMap(get_yield_data)

    data = spark.createDataFrame(rdd_data)

    data = data.select(
        "Value", "year", "state_name", "county_name",
    )

    data.toPandas().to_parquet("./data/yield_data.parquet", index=False)




