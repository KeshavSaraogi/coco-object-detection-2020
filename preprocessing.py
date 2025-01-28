from pyspark.sql import SparkSession
import boto3
import json

def read_annotations(file_path):
    """
    Reads a JSON file from S3 and returns its content as a dictionary.
    :param file_path: Path to the JSON file in S3.
    :return: Parsed JSON data.
    """
    s3 = boto3.client('s3')
    bucket, key = file_path.replace("s3://", "").split("/", 1)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj['Body'].read().decode('utf-8'))

def preprocess_annotations(file_path, dataset_type, processed_output):
    """
    Preprocesses COCO annotations and saves them as Parquet files.
    :param file_path: Path to the JSON file in S3.
    :param dataset_type: The type of dataset (e.g., "train", "val").
    :param processed_output: Path to save the processed files in S3.
    """
    annotations = read_annotations(file_path)
    images = annotations["images"]
    annotations_data = annotations["annotations"]

    # Flatten and prepare RDDs
    image_rdd = spark.sparkContext.parallelize(images).map(lambda x: (x["id"], x["file_name"], x["height"], x["width"]))
    annotations_rdd = spark.sparkContext.parallelize(annotations_data).map(lambda x: (x["image_id"], x["category_id"], x["bbox"], x["area"], x["iscrowd"]))

    # Create DataFrames
    images_df = image_rdd.toDF(["image_id", "file_name", "height", "width"])
    annotations_df = annotations_rdd.toDF(["image_id", "category_id", "bbox", "area", "iscrowd"])

    # Join and save as Parquet
    joined_df = images_df.join(annotations_df, on="image_id", how="inner")
    output_path = f"{processed_output}{dataset_type}_annotations.parquet"
    joined_df.write.parquet(output_path, mode="overwrite")
    print(f"Preprocessed {dataset_type} annotations saved to {output_path}.")

# Initialize Spark session
spark = SparkSession.builder.appName("COCO_Preprocessing").getOrCreate()

# Define S3 paths
s3_bucket = "images-data-coco"
train_annotations = f"s3://{s3_bucket}/extracted/annotations_trainval2017/instances_train2017.json"
val_annotations = f"s3://{s3_bucket}/extracted/annotations_trainval2017/instances_val2017.json"
processed_output = f"s3://{s3_bucket}/processed/"

# Process train and validation annotations
preprocess_annotations(train_annotations, "train", processed_output)
preprocess_annotations(val_annotations, "val", processed_output)

# Stop Spark session
spark.stop()