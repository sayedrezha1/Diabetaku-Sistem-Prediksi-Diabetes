import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

# Function to load the model lazily
def load_model():
    return joblib.load("../model/diabetes_model.sav")

# Function to load the dataset lazily
def load_data():
    return pd.read_csv('diabets_dataset_clean.csv')

# Function to handle prediction and accuracy calculation
def predict_diabetes(model, scaler, input_data, X, y):
    input_data_array = np.array(input_data).reshape(1, -1)
    std_data = scaler.transform(input_data_array)

    # Make the prediction
    prediction = model.predict(std_data)

    # Simulate accuracy for prediction display
    y_pred = model.predict(scaler.transform(X))
    accuracy = accuracy_score(y, y_pred) * 100  # Calculate accuracy in percentage

    return prediction, accuracy

# Set page config
st.set_page_config(
    page_title="DiabetaKu",
    layout="wide",
    page_icon="asset/diabetes_icon.png"
)

# Title Website
st.title('DiabetaKu : Aplikasi Prediksi Diabetes')

# Initialize lazy-loaded variables
model = None
data = None
scaler = StandardScaler()

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
input_data = [Gender, Age, Hipertension, Heart_disease, Smoking_history, bmi, HbA1c_level, Blood_glucose]

# Prediction section
diabetes_diagnosis = "BELUM MELENGKAPI DATA"
accuracy = None

# Use ThreadPoolExecutor for handling prediction request asynchronously
if st.button('Prediction'):
    # Lazy loading the model and dataset
    with ThreadPoolExecutor() as executor:
        futures = []
        futures.append(executor.submit(load_model))
        futures.append(executor.submit(load_data))

        # Wait for the results
        for future in as_completed(futures):
            result = future.result()
            if isinstance(result, pd.DataFrame):
                data = result
                X = data.drop(columns='diabetes', axis=1)
                y = data['diabetes']
                scaler.fit(X)  # Fit the scaler on the data
            else:
                model = result

        # Predict and calculate accuracy concurrently
        with ThreadPoolExecutor(max_workers=cpu_count()) as exec:
            prediction_future = exec.submit(predict_diabetes, model, scaler, input_data, X, y)
            prediction, accuracy = prediction_future.result()

            # Show results
            if prediction[0] == 0:
                diabetes_diagnosis = "PASIEN TIDAK TERKENA DIABETES"
            else:
                diabetes_diagnosis = "PASIEN TERKENA DIABETES"

    st.subheader("Hasil Prediksi Adalah")
    st.success(diabetes_diagnosis)

    st.subheader("Model Accuracy on Test Data")
    st.write(f"Accuracy: {accuracy:.2f}%")
