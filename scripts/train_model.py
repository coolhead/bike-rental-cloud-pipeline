import boto3
import yaml
import os
import numpy as np
from joblib import load, dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

def upload_model_to_s3(local_path, bucket_name, key, s3_client):
    s3_client.upload_file(local_path, bucket_name, key)

def train_and_tune_model(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    s3_client = boto3.client('s3')

    X_train, X_test, y_train, y_test = load("processed/train_test_split.joblib")

    model = RandomForestRegressor(**config['model_params'], random_state=42)

    tuner = RandomizedSearchCV(
        model,
        config['param_distributions'],
        n_iter=config['tuner']['n_iter'],
        cv=config['tuner']['cv'],
        random_state=42,
        n_jobs=-1
    )

    tuner.fit(X_train, y_train)

    best_model = tuner.best_estimator_

    if not os.path.exists("models"):
        os.makedirs("models")

    dump(best_model, "models/best_model.joblib")
    upload_model_to_s3("models/best_model.joblib", config['s3_bucket'], config['s3_model_key'], s3_client)

    print("✅ Model training and tuning completed!")

if __name__ == "__main__":
    train_and_tune_model("configs/config.yaml")

