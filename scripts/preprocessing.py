import boto3
import yaml
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump

def load_data_from_s3(bucket_name, key, local_path, s3_client):
    s3_client.download_file(bucket_name, key, local_path)

def preprocess_and_split_data(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    s3_client = boto3.client('s3')

    # Download data
    if not os.path.exists("data"):
        os.makedirs("data")
    load_data_from_s3(config['s3_bucket'], config['s3_data_key'], "data/bike_rental.csv", s3_client)

    df = pd.read_csv("data/bike_rental.csv")

    # Feature engineering
    df.drop(columns=['instant', 'dteday', 'casual', 'registered'], inplace=True)

    X = df.drop('cnt', axis=1)
    y = df['cnt']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    if not os.path.exists("processed"):
        os.makedirs("processed")

    dump((X_train, X_test, y_train, y_test), "processed/train_test_split.joblib")
    dump(scaler, "processed/feature_scaler.joblib")

    print("✅ Data preprocessing completed!")

if __name__ == "__main__":
    preprocess_and_split_data("configs/config.yaml")
