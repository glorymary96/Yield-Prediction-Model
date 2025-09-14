from pyspark.sql import functions as F, SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, ArrayType
import json
import requests
import os

spark = SparkSession.builder.appName("Weather").getOrCreate()

def download_url(url, local_file):

    if not os.path.exists(local_file):
        try:
            r = requests.get(url)
            with open(local_file, "wb") as f:
                f.write(r.content)
        except Exception as e:
            print(f"Download error: {e}")

def US_regions():
# --- Join and preserve structs as before ---
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    local_file = "../data/US_counties.json"
    download_url(url, local_file)
    counties_df = spark.read.option("multiline", "true").json(local_file)
    counties_exploded = counties_df.select(F.explode("features").alias("feature"))

    state_df = spark.read.option("multiline", "true").json("../data/gz_2010_us_040_00_20m.json")
    state_exploded = state_df.select(F.explode("features").alias("feature"))
    state_flat = state_exploded.select(
        F.col("feature.properties.STATE").alias("State_FIPS"),
        F.col("feature.properties.NAME").alias("State_Name")
    )

    joined_df = counties_exploded.join(
        state_flat,
        counties_exploded["feature.properties.STATE"] == state_flat["State_FIPS"],
        "left"
    )

    # --- Collect rows and build GeoJSON with structs preserved ---
    rows = joined_df.collect()
    features = []

    for row in rows:
        feature = {
            "type": row.feature.type,
            "id": row.feature.id,
            "geometry": {
                "type": row.feature.geometry.type,
                "coordinates": row.feature.geometry.coordinates
            },
            "properties": {
                "CENSUSAREA": row.feature.properties.CENSUSAREA,
                "COUNTY": row.feature.properties.COUNTY,
                "GEO_ID": row.feature.properties.GEO_ID,
                "LSAD": row.feature.properties.LSAD,
                "NAME": row.feature.properties.NAME,
                "STATE": row.feature.properties.STATE,
                "State_Name": row.State_Name  # new field
            }
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    # --- Save as single GeoJSON file ---
    output_path = "../data/US.geojson"
    with open(output_path, "w") as f:
        json.dump(geojson, f)

    print(f"GeoJSON saved as {output_path}")

    df = spark.read.json(output_path)

    df.printSchema()

if __name__ == "__main__":
    US_regions()