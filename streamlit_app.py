import streamlit as st
import pandas as pd
import joblib
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Medical Insurance Charges Predictor",
    page_icon="🏥",
    layout="centered"
)

# --- MODEL LOADING ---
# Using cache_resource to ensure the model is loaded only once
@st.cache_resource
def load_model():
    model_path = "best_model_pipeline.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"Model file '{model_path}' not found. Please ensure it is in the same directory.")
        return None

model = load_model()

# --- APP INTERFACE ---
st.title("🏥 Medical Insurance Cost Predictor")
st.markdown("""
Predict estimated health insurance charges based on demographic and health indicators.
Please fill in the details below and click **Predict**.
""")

# Create a form for user inputs
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30, help="Age of the primary beneficiary")
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, format="%.2f")
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

    with col2:
        sex = st.selectbox("Sex", options=["male", "female"], index=0)
        smoker = st.selectbox("Smoker Status", options=["yes", "no"], index=1)
        region = st.selectbox("Region", options=["southwest", "southeast", "northwest", "northeast"])

    submit_button = st.form_submit_button("Predict Charges")

# --- PREDICTION LOGIC ---
if submit_button:
    if model is not None:
        # Create a DataFrame matching the model's training feature names
        input_df = pd.DataFrame([{
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region
        }])

        try:
            # Generate prediction using the pipeline
            prediction = model.predict(input_df)[0]
            
            # Display the result
            st.success("### Prediction Result")
            st.metric(label="Estimated Annual Charges", value=f"${prediction:,.2f}")
            
            # Visual context
            st.info("Note: This estimate is based on the patterns identified in the training dataset.")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Prediction could not be performed because the model is missing.")

# --- FOOTER ---
st.markdown("---")
st.caption("Powered by Scikit-Learn and Streamlit")