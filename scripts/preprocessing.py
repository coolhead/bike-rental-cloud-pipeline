import pandas as pd
import boto3
import os
import yaml
from joblib import dump

def load_data_from_s3(bucket_name, key, local_path, s3_client):
    s3_client.download_file(bucket_name, key, local_path)
    return pd.read_csv(local_path)

def save_processed_data(X_train, X_test, y_train, y_test, scaler, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dump((X_train, X_test, y_train, y_test), os.path.join(output_dir, "train_test_split.joblib"))
    dump(scaler, os.path.join(output_dir, "feature_scaler.joblib"))

def preprocess_and_split_data(config_path):
    # Load configs
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    s3_bucket = config['s3_bucket']
    s3_data_key = config['s3_data_key']

    s3_client = boto3.client('s3')

    # Load data from S3
    df = load_data_from_s3(s3_bucket, s3_data_key, "bike_rental.csv", s3_client)

    # Preprocessing (very simple example)
    df = df.drop(columns=["instant", "dteday", "casual", "registered"])
    X = df.drop(columns=["cnt"])
    y = df["cnt"]

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Save processed data locally
    save_processed_data(X_train, X_test, y_train, y_test, scaler, "processed")

if __name__ == "__main__":
    preprocess_and_split_data("configs/config.yaml")

