from fastapi import FastAPI
import pandas as pd
from joblib import load
import uvicorn

app = FastAPI()

# Load model at startup
model = load("models/best_model.joblib")
scaler = load("processed/feature_scaler.joblib")

@app.get("/")
def home():
    return {"message": "Bike Rental Prediction API"}

@app.post("/predict")
def predict(features: dict):
    # Features should be sent as JSON
    input_df = pd.DataFrame([features])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return {"predicted_bike_rentals": int(prediction[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

