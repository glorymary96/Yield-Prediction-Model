
from WeatherTransform import TransformWeatherRegions
from pyspark.sql.functions import *
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
GET_COUNTRIES = ["US"]

regions = []

# Define month_start and month_end for each crop type
crop_seasons = {
    "HRW": (8, 7),
    "SRW": (8, 7),
    "HRS": (3, 10),
    "corn": (10, 9),
}

# Define regions for different countries as tuples
country_regions = {
    "US": {
        "HRW": ["Colorado", "Kansas", "Montana", "Nebraska",
                "Oklahoma", "South Dakota", "Texas"],
        "SRW": ["Alabama", "Arkansas", "Georgia", "Illinois",
                "Indiana", "Kentucky", "Michigan", "Mississippi",
                "Missouri", "North Carolina", "Ohio", "South Carolina", "Tennessee", "Virginia"],
        "HRS": ["Minnesota", "Montana", "North Dakota", "South Dakota"],
        "corn": ["Illinois", "Indiana", "Iowa", "Minnesota",
                 "Missouri", "Nebraska", "North Dakota",
                 "Ohio", "South Dakota",]
    }
}

# Populate regions list for selected countries
for country in GET_COUNTRIES:
    if country in country_regions:
        for crop_type, locations in country_regions[country].items():
            month_start, month_end = crop_seasons[crop_type]  # Retrieve predefined months
            #for state, county in locations:
            if country == "US":
                output_path = "../data/US.geojson"
                df = spark.read.json(output_path)
                df = df.select(explode("features").alias("features"))
                df = df.select(
                    col("features.properties.State_Name").alias("State"),
                    col("features.properties.NAME").alias("County"),
                )
                df = df.filter(col("State").isin(locations))

                weather_regions = [(st[0], count[0]) for st, count in
                          zip(df.select("State").collect(), df.select("County").collect())]

                for state, count in weather_regions:
                    regions.append(TransformWeatherRegions(country, state, count, crop_type, month_start, month_end))


