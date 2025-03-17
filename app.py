import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load the pre-trained churn prediction model
model = tf.keras.models.load_model('churn_model.h5')

# Load the saved scaler for feature normalization
with open('scalar.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
# Load the saved label encoder for gender encoding
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
    
# Load the saved one-hot encoder for geography encoding
with open('one_hot_encoder_geography.pkl', 'rb') as f:
    one_hot_encoder_geography = pickle.load(f)
    
st.title('Churn Prediction')

# User input fields
geography = st.selectbox('Geography', one_hot_encoder_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare input data for prediction
input_data = pd.DataFrame(
    {
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]], 
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_credit_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary],      
    }
)

# One-hot encode geography field
geo_encoded = one_hot_encoder_geography.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geography.get_feature_names_out(['Geography']))

# Combine encoded geography with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data using the loaded scaler
input_data_scaled = scaler.transform(input_data)

# Make churn prediction using the model
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

# Display prediction result
if prediction_probability > 0.5:
    st.write('Churn')
else:
    st.write('Not Churn')
    
# Display churn probability
st.write(f'Probability of Churn: {prediction_probability}')
