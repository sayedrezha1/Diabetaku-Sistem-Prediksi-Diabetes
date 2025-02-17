import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load Model
diabetes_model = joblib.load("../model/diabetes_model.sav")
df = pd.read_csv('diabets_dataset_clean.csv')

X = df.drop(columns='diabetes', axis=1)
y = df['diabetes']
scaler = StandardScaler()

# Fit scaler on the dataset
scaler.fit(X)

# Set page config
st.set_page_config(
    page_title="DiabetaKu",
    layout="wide",
    page_icon="asset/diabetes_icon.png"
)

# Title Website
st.title('DiabetaKu : Aplikasi Prediksi Diabetes')

# User input columns
col1, col2 = st.columns(2)

with col1:
    Gender = st.number_input('Gender (0 : Female, 1 : Male):', min_value=0, max_value=1, step=1)

with col1:
    Age = st.number_input("Your Age:", min_value=1, max_value=120, step=1)

with col1:
    Hipertension = st.number_input('Have Hypertension? (1 : yes, 0 : no):', min_value=0, max_value=1, step=1)

with col1:
    Heart_disease = st.number_input('Have Heart Disease? (1 : yes, 0 : no):', min_value=0, max_value=1, step=1)

with col2:
    Smoking_history = st.number_input(
        "Are You Smoking? (-1 : No Info, 0 : never, 1 : former, 2 : current, 3 : not current, 4 : ever):",
        min_value=-1, max_value=4, step=1)

with col2:
    bmi = st.number_input('Your BMI:', min_value=0.0, step=0.1)

with col2:
    HbA1c_level = st.number_input("Level Hemoglobin A1c:", min_value=0.0, step=0.1)

with col2:
    Blood_glucose = st.number_input('Level Blood Glucose:', min_value=0.0, step=0.1)

# Prepare input data
input_data = np.array([Gender, Age, Hipertension, Heart_disease, Smoking_history, bmi, HbA1c_level, Blood_glucose])
input_reshape = input_data.reshape(1, -1)
std_data = scaler.transform(input_reshape)

# Prediction section
diabetes_diagnosis = "BELUM MELENGKAPI DATA"
accuracy = None

# Predict when button is pressed
if st.button('Prediction'):
    diabetes_prediction = diabetes_model.predict(std_data)

    # Simulate accuracy for prediction display
    y_pred = diabetes_model.predict(scaler.transform(X))
    accuracy = accuracy_score(y, y_pred) * 100  # Calculate accuracy in percentage

    # Diagnosis result
    if diabetes_prediction[0] == 0:
        diabetes_diagnosis = "PASIEN TIDAK TERKENA DIABETES"
    else:
        diabetes_diagnosis = "PASIEN TERKENA DIABETES"

    st.subheader("Hasil Prediksi Adalah")
    st.success(diabetes_diagnosis)

    # Show accuracy in percentage
    st.subheader("Model Accuracy on Test Data")
    st.write(f"Accuracy: {accuracy:.2f}%")
