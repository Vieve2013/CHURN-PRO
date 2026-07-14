
import streamlit as st
import pandas as pd
import numpy as np
import joblib

import warnings 
warnings.filterwarnings('ignore')

## Page title
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Customer Churn Prediction")

st.write(
    "Enter customer information below and click Predict."
)

## Create the input form
### Credit Score 
credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

### Geography
geography = st.selectbox(
    "Geography",
    [
        'Christchurch', 
        'Dunedin',
        'Hamilton', 
        'Nelson', 
        'New Plymouth',
        'Palmerston North', 
        'Rotorua', 
        'Tauranga',
        'Wellington'
    ]
)

### Gender
gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

gender = 1 if gender == "Male" else 0

### Age
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

### Tenure
tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=20,
    value=5
)

### Balance
balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

### Number of Products
products = st.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=2
)

### Has Credit Card
credit_card = st.selectbox(
    "Has Credit Card",
    ["Yes", "No"]
)
credit_card = 1 if credit_card == "Yes" else 0

### Is Active Member
active = st.selectbox(
    "Is Active Member",
    ["Yes", "No"]
)

active = 1 if active == "Yes" else 0

### Estimated Salary
salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)

## Prediction button
if st.button("Predict"):
    input_df = pd.DataFrame({
        "CreditScore":[credit_score],
        "Geography_Christchurch": [1 if geography == "Christchurch" else 0],
        "Geography_Dunedin": [1 if geography == "Dunedin" else 0],
        "Geography_Hamilton": [1 if geography == "Hamilton" else 0],
        "Geography_Nelson": [1 if geography == "Nelson" else 0],
        "Geography_New Plymouth": [1 if geography == "New Plymouth" else 0],
        "Geography_Palmerston North": [1 if geography == "Palmerston North" else 0],
        "Geography_Rotorua": [1 if geography == "Rotorua" else 0],
        "Geography_Tauranga": [1 if geography == "Tauranga" else 0],
        "Geography_Wellington": [1 if geography == "Wellington" else 0],
        "Gender_Male":[gender],
        "Age":[age],
        "Tenure":[tenure],
        "Balance":[balance],
        "Num Of Products":[products],
        "Has Credit Card":[credit_card],
        "Is Active Member":[active],
        "Estimated Salary":[salary]
    })
        
    # Apply exactly the same preprocessing

    input_df["Tenure_to_product"] = (
        input_df["Tenure"] /
        input_df["Num Of Products"]
    )

    ## Match the training columns
    expected_columns = [
        'CreditScore',
        'Age',
        'Tenure',
        'Balance',
        'Num Of Products',
        'Has Credit Card',
        'Is Active Member',
        'Estimated Salary',
        'Gender_Male',
        'Geography_Christchurch',
        'Geography_Dunedin',
        'Geography_Hamilton',
        'Geography_Nelson',
        'Geography_New Plymouth',
        'Geography_Palmerston North',
        'Geography_Rotorua',
        'Geography_Tauranga',
        'Geography_Wellington',
        'Tenure_to_product'
    ]

    input_df = input_df.reindex(
        columns=expected_columns,
        fill_value=0
    )

    st.dataframe(input_df)
    
    # Load the model
    model = joblib.load("logistic_model.pkl")

    # Predict the churn
    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    ## Display the result
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is unlikely to churn")
        
    st.metric(
        "Probability of Churn",
        f"{probability:.2%}"
    )

    st.metric(
        "Probability of Not Churning",
        f"{1 - probability:.2%}"
    )
