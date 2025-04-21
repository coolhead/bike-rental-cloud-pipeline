import boto3
import yaml
import os
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

def upload_model_to_s3(local_path, bucket_name, s3_key, s3_client):
    s3_client.upload_file(local_path, bucket_name, s3_key)

def train_model(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    X_train, X_test, y_train, y_test = load("processed/train_test_split.joblib")
    
    base_model = RandomForestRegressor(**config['model_params'])

    search = RandomizedSearchCV(
        base_model,
        config['param_distributions'],
        n_iter=config['tuner']['n_iter'],
        cv=config['tuner']['cv'],
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    # Save model locally first
    if not os.path.exists("models"):
        os.makedirs("models")
    dump(best_model, "models/best_model.joblib")

    # Upload to S3
    s3_client = boto3.client('s3')
    upload_model_to_s3("models/best_model.joblib", config['s3_bucket'], config['s3_model_key'], s3_client)

if __name__ == "__main__":
    train_model("configs/config.yaml")

