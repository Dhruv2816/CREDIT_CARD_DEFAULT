import os
import sys
import streamlit as st
import pandas as pd
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
sys.path.append(project_root)
from src.exception import CustomException
from src.utils import load_object
from src.pipeline.predict_pipeline import PredictPipeline
from src.custom_data import CustomData

# Function to load model and preprocessing pipeline
def load_model_and_preprocessor():
    try:
        model_path = "artifacts/model.pkl"
        preprocessor_path = "artifacts/preprocessor.pkl"
        model = load_object(model_path)
        preprocessor = load_object(preprocessor_path)
        return model, preprocessor
    except Exception as e:
        raise CustomException(f"Failed to load model and preprocessor: {e}")

# Function to make predictions
def main():
    st.title("Credit Card Default Prediction")

    # Load model and preprocessing pipeline
    model, preprocessor = load_model_and_preprocessor()

    # Input fields for user data
    st.header("Enter Customer Data")
    ID = 1
    LIMIT_BAL = st.number_input("Limit Balance", value=0.0)
    SEX = st.selectbox("Sex", ["Male", "Female"])
    EDUCATION = st.selectbox("Education", [1, 2, 3, 4])
    MARRIAGE = st.selectbox("Marriage", [1, 2, 3])
    AGE = st.number_input("Age", value=18)
    PAY_0 = st.number_input("PAY_0", value=0)
    PAY_2 = st.number_input("PAY_2", value=0)
    PAY_3 = st.number_input("PAY_3", value=0)
    PAY_4 = st.number_input("PAY_4", value=0)
    PAY_5 = st.number_input("PAY_5", value=0)
    PAY_6 = st.number_input("PAY_6", value=0)
    BILL_AMT1 = st.number_input("BILL_AMT1", value=0.0)
    BILL_AMT2 = st.number_input("BILL_AMT2", value=0.0)
    BILL_AMT3 = st.number_input("BILL_AMT3", value=0.0)
    BILL_AMT4 = st.number_input("BILL_AMT4", value=0.0)
    BILL_AMT5 = st.number_input("BILL_AMT5", value=0.0)
    BILL_AMT6 = st.number_input("BILL_AMT6", value=0.0)
    PAY_AMT1 = st.number_input("PAY_AMT1", value=0.0)
    PAY_AMT2 = st.number_input("PAY_AMT2", value=0.0)
    PAY_AMT3 = st.number_input("PAY_AMT3", value=0.0)
    PAY_AMT4 = st.number_input("PAY_AMT4", value=0.0)
    PAY_AMT5 = st.number_input("PAY_AMT5", value=0.0)
    PAY_AMT6 = st.number_input("PAY_AMT6", value=0.0)

    # Create CustomData object
    custom_data = CustomData(
        ID=ID, LIMIT_BAL=LIMIT_BAL, SEX=1 if SEX == "Male" else 2,
        EDUCATION=EDUCATION, MARRIAGE=MARRIAGE, AGE=AGE,
        PAY_0=PAY_0, PAY_2=PAY_2, PAY_3=PAY_3, PAY_4=PAY_4,
        PAY_5=PAY_5, PAY_6=PAY_6, BILL_AMT1=BILL_AMT1,
        BILL_AMT2=BILL_AMT2, BILL_AMT3=BILL_AMT3, BILL_AMT4=BILL_AMT4,
        BILL_AMT5=BILL_AMT5, BILL_AMT6=BILL_AMT6, PAY_AMT1=PAY_AMT1,
        PAY_AMT2=PAY_AMT2, PAY_AMT3=PAY_AMT3, PAY_AMT4=PAY_AMT4,
        PAY_AMT5=PAY_AMT5, PAY_AMT6=PAY_AMT6
    )

    # Predict button
    if st.button("Predict"):
        try:
            # Make predictions
            predictions = make_predictions(model, preprocessor, custom_data)
            st.success(f"Predicted Default Payment Next Month: {predictions[0]}")
        except CustomException as e:
            st.error(f"Prediction error: {str(e)}")

def make_predictions(model, preprocessor, custom_data):
    try:
        # Convert custom_data to DataFrame
        custom_data_df = custom_data.get_data_as_dataframe()

        # Preprocess the data
        processed_data = preprocessor.transform(custom_data_df)

        # Make predictions using the model
        predictions = model.predict(processed_data)

        return predictions
    except Exception as e:
        raise CustomException(f"Prediction error: {e}", str(e))

if __name__ == "__main__":
    main()