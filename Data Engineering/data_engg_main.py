from pyspark.sql import SparkSession
import os
import requests
import json
import Regions
from pyspark.sql.functions import cast, lpad, col, concat, explode

spark = SparkSession.builder.appName("Weather").getOrCreate()

def download_url(url, local_file):
    if not os.path.exists(local_file):
        try:
            r = requests.get(url)
            with open(local_file, "wb") as f:
                f.write(r.content)
        except Exception as e:
            print(f"Download error: {e}")




if __name__ == "__main__":

    print("Done")