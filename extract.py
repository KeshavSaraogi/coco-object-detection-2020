from pyspark.sql import SparkSession
import boto3
import zipfile
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# S3 bucket details
S3_BUCKET_NAME = "images-data-coco"
DATASET_FILES = [
    "train2017.zip",
    "val2017.zip",
    "test2017.zip",
    "annotations_trainval2017.zip"
]
EXTRACTED_FOLDER = "extracted/"

# Initialize Spark session
spark = SparkSession.builder \
    .appName("COCO Dataset Extraction") \
    .getOrCreate()

# Boto3 client for S3 interactions
s3 = boto3.client('s3')

def download_and_extract(file_key):
    """Download a ZIP file from S3, extract it, and return the local extraction path."""
    local_zip_path = f"/mnt/tmp/{file_key}"
    extracted_dir = f"/mnt/tmp/{EXTRACTED_FOLDER}{file_key.replace('.zip', '')}/"

    # Ensure directories exist
    os.makedirs(os.path.dirname(local_zip_path), exist_ok=True)
    os.makedirs(extracted_dir, exist_ok=True)

    # Download file from S3
    logging.info(f"Downloading {file_key} from S3...")
    s3.download_file(S3_BUCKET_NAME, file_key, local_zip_path)

    # Extract ZIP contents
    logging.info(f"Extracting {file_key}...")
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_dir)

    logging.info(f"Extraction completed for: {file_key}")

    return extracted_dir

def upload_to_s3(local_extracted_dir, s3_folder):
    """Uploads extracted files back to S3."""
    for root, _, files in os.walk(local_extracted_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            s3_key = os.path.join(EXTRACTED_FOLDER, s3_folder, file).replace("\\", "/")

            logging.info(f"Uploading {file} to s3://{S3_BUCKET_NAME}/{s3_key}...")
            s3.upload_file(local_file_path, S3_BUCKET_NAME, s3_key)

    logging.info(f"Uploaded extracted data to S3: {s3_folder}")

def process_coco_dataset():
    """Main processing function to handle extraction and upload."""
    logging.info("Starting COCO dataset processing on EMR cluster...")

    for file_key in DATASET_FILES:
        extracted_dir = download_and_extract(file_key)
        upload_to_s3(extracted_dir, file_key.replace('.zip', ''))

    logging.info("All dataset files have been successfully processed.")

# Run the extraction and upload process
process_coco_dataset()

# Stop Spark session
spark.stop()
