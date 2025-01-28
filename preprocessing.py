from pyspark.sql import SparkSession
import boto3
import json
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Spark session
spark = SparkSession.builder.appName("COCO Data Preprocessing").getOrCreate()

# Define S3 bucket details
S3_BUCKET = "images-data-coco"
TRAIN_IMAGES_PREFIX = "extracted/train2017/"
VAL_IMAGES_PREFIX = "extracted/val2017/"
TEST_IMAGES_PREFIX = "extracted/test2017/"
ANNOTATIONS_PATH = "extracted/annotations_trainval2017/instances_train2017.json"
OUTPUT_PREFIX = "processed-images/"

# Initialize S3 client
s3_client = boto3.client('s3')

def load_annotations():
    """Load COCO annotations from S3."""
    response = s3_client.get_object(Bucket=S3_BUCKET, Key=ANNOTATIONS_PATH)
    return json.loads(response['Body'].read().decode('utf-8'))

def read_image_from_s3(image_key):
    """Read image from S3."""
    response = s3_client.get_object(Bucket=S3_BUCKET, Key=image_key)
    image_data = response['Body'].read()
    image = Image.open(BytesIO(image_data))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def preprocess_image(image_key):
    """Resize and grayscale image, then upload to S3."""
    try:
        img = read_image_from_s3(image_key)
        resized = cv2.resize(img, (512, 512))
        processed_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Convert processed image to bytes
        _, buffer = cv2.imencode(".jpg", processed_img)
        s3_client.put_object(Bucket=S3_BUCKET, Key=OUTPUT_PREFIX + image_key.split("/")[-1], Body=buffer.tobytes())
        
        return image_key, processed_img.shape
    except Exception as e:
        return image_key, str(e)

def analyze_class_distribution(annotations):
    """Analyze and plot class distribution from COCO annotations."""
    category_counts = {}
    for ann in annotations['annotations']:
        category_id = ann['category_id']
        category_counts[category_id] = category_counts.get(category_id, 0) + 1

    # Plot distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(category_counts.keys()), y=list(category_counts.values()))
    plt.title("COCO Class Distribution")
    plt.xlabel("Class ID")
    plt.ylabel("Frequency")
    
    # Save figure as bytes and upload to S3
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    s3_client.put_object(Bucket=S3_BUCKET, Key="eda/class_distribution.png", Body=buf)

def main():
    """Main function to execute preprocessing."""
    annotations = load_annotations()
    analyze_class_distribution(annotations)

    # Get all image keys
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=TRAIN_IMAGES_PREFIX)
    image_keys = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.jpg')]

    # Process images using Spark
    images_rdd = spark.sparkContext.parallelize(image_keys)
    results = images_rdd.map(preprocess_image).collect()

    # Print sample results
    for res in results[:10]:
        print(res)

    print("Preprocessing completed and saved to S3.")

if __name__ == "__main__":
    main()
