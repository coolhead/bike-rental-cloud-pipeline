import boto3
import yaml
import os
import pandas as pd
from joblib import load
from sklearn.metrics import mean_squared_error

def download_model_from_s3(bucket_name, key, local_path, s3_client):
    s3_client.download_file(bucket_name, key, local_path)

def upload_predictions_to_s3(local_path, bucket_name, s3_key, s3_client):
    s3_client.upload_file(local_path, bucket_name, s3_key)

def model_inference(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    s3_client = boto3.client('s3')

    # Download best model
    if not os.path.exists("models"):
        os.makedirs("models")
    download_model_from_s3(config['s3_bucket'], config['s3_model_key'], "models/best_model.joblib", s3_client)

    model = load("models/best_model.joblib")
    X_train, X_test, y_train, y_test = load("processed/train_test_split.joblib")

    y_pred = model.predict(X_test)

    # Save predictions locally
    predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    if not os.path.exists("predictions"):
        os.makedirs("predictions")
    predictions_df.to_csv("predictions/predictions.csv", index=False)

    # Upload to S3
    upload_predictions_to_s3("predictions/predictions.csv", config['s3_bucket'], config['s3_predictions_key'], s3_client)

    # Optional: print RMSE
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"Root Mean Squared Error on Test Set: {rmse:.2f}")

if __name__ == "__main__":
    model_inference("configs/config.yaml")

