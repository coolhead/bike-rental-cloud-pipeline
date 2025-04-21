<p align="center">
  <img src="Banner.jpg" alt="Bike Rental MLOps Project Banner" width="80%" />
</p>


# Bike Rental Prediction MLOps Pipeline 🚲

## Overview
This project implements a full Machine Learning Operations (MLOps) pipeline for predicting bike rentals using AWS Cloud and a local setup. It includes:
- Data Preprocessing
- Hyperparameter Tuning
- Model Training
- Model Inference
- Saving Models and Predictions

Designed to be cloud-friendly, simple, modular, and production-ready!

---

## 📁 Project Structure

```
├── data/
│   └── bike_rental.csv (Input dataset)
├── processed/
│   ├── train_test_split.joblib (Saved train/test data)
│   └── feature_scaler.joblib (Scaler object)
├── models/
│   └── best_model.joblib (Saved best trained model)
├── predictions/
│   └── predictions.csv (Test set predictions)
├── scripts/
│   ├── preprocessing.py (Data preprocessing)
│   ├── train_model.py (Model training and tuning)
│   └── model_inference.py (Model inference)
├── requirements.txt (Required Python packages)
└── README.md
```

---

## 🧪 Environment Setup

```bash
# Install required packages
pip install -r requirements.txt
```

---

## 🔥 How to Run

### Step 1: Preprocess Data

```bash
python scripts/preprocessing.py
```

- Reads `bike_rental.csv`
- Cleans, scales, splits into train/test
- Saves processed files into `processed/`

### Step 2: Train and Tune Model

```bash
python scripts/train_model.py
```

- Loads train/test splits
- Runs RandomizedSearchCV for hyperparameter tuning
- Saves the best model to `models/`

### Step 3: Model Inference

```bash
python scripts/model_inference.py
```

- Loads the trained model and test set
- Generates predictions
- Saves results to `predictions/`

---

## 📚 Key Features

- Clean train-test split
- Feature scaling (without scaling the target)
- Randomized Search for hyperparameter tuning
- Joblib-based model persistence
- Modular and extensible structure

---

## 💡 Future Enhancements
- Add FastAPI server for real-time inference
- Integrate MLflow for tracking experiments
- Setup Airflow for end-to-end automation
- Expand to multiple models (XGBoost, LightGBM)

---

Happy Coding! 🎯
