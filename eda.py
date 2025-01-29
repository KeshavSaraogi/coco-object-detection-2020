from pyspark.sql import SparkSession
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Spark
spark = SparkSession.builder.appName("COCO_EDA").getOrCreate()

# S3 Paths for Processed Data
s3_bucket = "images-data-coco"
train_path = f"s3://{s3_bucket}/processed/train_annotations.parquet"
val_path = f"s3://{s3_bucket}/processed/val_annotations.parquet"

# Load Data
train_df = spark.read.parquet(train_path).toPandas()
val_df = spark.read.parquet(val_path).toPandas()

# Combine Train and Val for Analysis
df = pd.concat([train_df, val_df])

# Plot Image Size Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["width"] * df["height"], bins=30, kde=True)
plt.xlabel("Image Size (Pixels)")
plt.ylabel("Count")
plt.title("Distribution of Image Sizes in COCO Dataset")
plt.savefig("image_size_distribution.png")

# Plot Object Categories
plt.figure(figsize=(12, 6))
sns.countplot(y=df["category_id"], order=df["category_id"].value_counts().index)
plt.xlabel("Count")
plt.ylabel("Category ID")
plt.title("Distribution of Object Categories")
plt.savefig("category_distribution.png")

print("EDA Completed. Saved plots to local storage.")

# Stop Spark
spark.stop()
