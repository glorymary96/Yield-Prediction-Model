import json
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder.appName("WeatherTransform").getOrCreate()

class TransformWeatherRegions:
    def __init__(self,
                 country:str,
                 state:str,
                 county:str,
                 crop:str,
                 month_start:int,
                 month_end:int
                ):
        self.country = country
        self.state = state
        self.county = county
        self.crop = crop
        self.month_start = month_start
        self.month_end = month_end
        self.polygon_region = self.getWeatherRegion()


    def getWeatherRegion(self):

        #Construct file path
        file_name = f"../data/{self.country}.geojson"

        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File {file_name} not found.")
        try:
            df = spark.read.json(file_name)
            df = df.select(explode("features").alias("features"))

            df = df.select(
                col("features.properties.State_Name").alias("State"),
                col("features.properties.NAME").alias("County"),
                col("features.geometry.coordinates").alias("Coordinates")
            )

            polygon = df.filter((col("State")==self.state) & (col("County")==self.county)).select("Coordinates").collect()[0][0]

            print(len(polygon))

        except json.decoder.JSONDecodeError:
            raise ValueError(f"Error decoding JSON in file: {file_name}")
        except Exception as e:
            raise RuntimeError(f"An error occurred while processing the file: {file_name} - {str(e)}")

