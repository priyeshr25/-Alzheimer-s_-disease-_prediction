import streamlit as st
import pickle
import pandas as pd


# Load the trained model and dataset
model = pickle.load(open('model_pipeline.pkl', 'rb'))
df = pickle.load(open('data.pkl', 'rb'))





st.title("Alzheimer's (disease) Predictor")
st.write("""Alzheimer's disease is a progressive neurodegenerative disorder 
that primarily affects memory, thinking, and behavior. It happens due to the abnormal buildup of proteins in and around brain cells, leading to brain damage over time. The exact cause is not fully understood, 
but the disease is associated with several key processes""")

# Creating input fields based on dataset features
age = st.slider('Age', min_value=40, max_value=100, value=65)
gender = st.selectbox('Gender', df['gender'].unique())
physical_activity_level = st.selectbox('Physical Activity Level', df['physical activity level'].unique())
smoking_status = st.selectbox('Smoking Status', df['smoking status'].unique())
alcohol_consumption = st.selectbox('Alcohol Consumption', df['alcohol consumption'].unique())
bmi = st.slider('BMI', min_value=10, max_value=40, value=25)
diabetes = st.selectbox('Diabetes', df['diabetes'].unique())
hypertension = st.selectbox('Hypertension', df['hypertension'].unique())
cholesterol_level = st.selectbox('Cholesterol Level', df['cholesterol level'].unique())
family_history = st.selectbox("Family History of Alzheimer's", df['family history of alzheimer’s'].unique())
cognitive_test_score = st.selectbox('Cognitive Test Score', df['cognitive test score'].unique())
depression_level = st.selectbox('Depression Level', df['depression level'].unique())
sleep_quality = st.selectbox('Sleep Quality', df['sleep quality'].unique())
dietary_habits = st.selectbox('Dietary Habits', df['dietary habits'].unique())
air_pollution_exposure = st.selectbox('Air Pollution Exposure', df['air pollution exposure'].unique())
genetic_risk_factor = st.selectbox('Genetic Risk Factor (APOE-ε4)', df['genetic risk factor (apoe-ε4 allele)'].unique())
social_engagement_level = st.selectbox('Social Engagement Level', df['social engagement level'].unique())
stress_levels = st.selectbox('Stress Levels', df['stress levels'].unique())  # Removed the extra tab space
urban_vs_rural_living = st.selectbox('Urban vs Rural Living', df['urban vs rural living'].unique())

# False Negative Control Slider
fn_control = st.slider('False Negative Control (Lower = More Sensitive)', min_value=0.1, max_value=1.0, value=0.5, step=0.05)

# Prediction Button
if st.button("Predict Alzheimer's Risk"):
    # Prepare input data as per model requirements
    input_data = pd.DataFrame([[
        age, gender, physical_activity_level, smoking_status, alcohol_consumption, bmi,
        diabetes, hypertension, cholesterol_level, family_history, cognitive_test_score,
        depression_level, sleep_quality, dietary_habits, air_pollution_exposure,
        genetic_risk_factor, social_engagement_level, stress_levels, urban_vs_rural_living
    ]], columns=[
        'age', 'gender', 'physical activity level', 'smoking status', 'alcohol consumption', 'bmi',
        'diabetes', 'hypertension', 'cholesterol level', 'family history of alzheimer’s',
        'cognitive test score', 'depression level', 'sleep quality', 'dietary habits',
        'air pollution exposure', 'genetic risk factor (apoe-ε4 allele)',
        'social engagement level', 'stress levels', 'urban vs rural living'
    ])

    # Predict probability
    probability = model.predict_proba(input_data)[:, 1][0]  # Assuming class 1 means Alzheimer's positive

    # Adjust prediction threshold based on FN control
    threshold = 0.5 * fn_control
    prediction = 'Positive' if probability >= threshold else 'Negative'

    # Display Results
    st.subheader(f'Predicted Alzheimer\'s Status: {prediction}')
    st.write(f'Confidence Score: {probability:.2f}')
