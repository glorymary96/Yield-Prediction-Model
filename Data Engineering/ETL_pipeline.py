import os
import requests
import numpy as np
from netCDF4 import Dataset
from pyspark import SparkContext
from pyspark.sql import SparkSession, Row


output_dir = "../data/"

years = list(range(2015, 2021))  # synthetic years

def download_nc(year):
    local_file = os.path.join(output_dir, f"precip.{year}.nc")
    if not os.path.exists(local_file):
        url = f"https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/precip.{year}.nc"
        print(f"Downloading {year} ...")
        try:
            r = requests.get(url)
            with open(local_file, "wb") as f:
                f.write(r.content)
            print(f"Download complete for {year}.")
        except Exception as e:
            print(f"Failed to download {year}: {e}")
    else:
        print(f"File already exists for {year}.")


def process_year(year):
    local_file = os.path.join(output_dir, f"precip.{year}.nc")
    nc = Dataset(local_file, "r")
    precip = nc.variables["precip"][:]
    lat_vals = nc.variables["lat"][:]
    lon_vals = nc.variables["lon"][:]
    time_vals = nc.variables["time"][:]
    nc.close()
    rows = []
    for t_idx, t_val in enumerate(time_vals):
        for lat_idx, lat_val in enumerate(lat_vals):
            for lon_idx, lon_val in enumerate(lon_vals):
                val = float(precip[t_idx, lat_idx, lon_idx])
                if np.isnan(val):
                    val = 0.0
                rows.append(Row(year=year,
                                day_of_year=int(t_val),
                                lat=float(lat_val),
                                lon=float(lon_val),
                                precip=val))
    return rows


if __name__ == "__main__":
    # Create spark session
    spark = SparkSession.builder \
        .appName("ETL_pipeline") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()

    sc = spark.sparkContext

    sc.parallelize(years).map(download_nc).collect()

    sc.stop()