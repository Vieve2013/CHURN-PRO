import streamlit as st
import pandas as pd
import numpy as np
import joblib


import warnings 
warnings.filterwarnings('ignore')

## Data Import
data = pd.read_csv(r"data.csv")

# Data Preprocessing
data = pd.get_dummies(data, columns=['Gender'], drop_first=True)
data = pd.get_dummies(data, columns=['Geography'], drop_first=True)
data["Tenure_to_product"] = data["Tenure"] / data["Num Of Products"]

# Prediction
model = joblib.load(r"logistic_model.pkl")

preds = model.predict(data)
probs = model.predict_proba(data)[:, 1]
print(f"Predictions for last 10 samples: {preds[-10:]}")
print(f"Probabilities for last 10 samples: {probs[-10:]}")

