import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the model
model = tf.keras.models.load_model(r'D:\Python Projects\DeepLearningProjects\BankChurnProject\model.h5')

# Load encoders and scaler
with open(r'D:\Python Projects\DeepLearningProjects\BankChurnProject\label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open(r'D:\Python Projects\DeepLearningProjects\BankChurnProject\onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open(r'D:\Python Projects\DeepLearningProjects\BankChurnProject\scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit UI
st.title("Customer Churn Prediction Using ANN")

# Collect user input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare input features
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine all features
input_df = pd.concat([input_data, geo_encoded_df], axis=1)

# Scale the data
input_scaled = scaler.transform(input_df)

# Make prediction
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

# Show result
st.write(f"### Churn Probability: `{prediction_proba:.2f}`")

if prediction_proba > 0.5:
    st.error("⚠️ The Person is likely going to churn.")
else:
    st.success("✅ The person is not going to churn.")
