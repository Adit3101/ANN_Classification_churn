import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import streamlit as st
model=tf.keras.models.load_model('model.keras')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender=pickle.load(f)
with open('onehot_encoder.pkl', 'rb') as f:
    onehot_encoder=pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler=pickle.load(f)

##streamlit app
st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', onehot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance', min_value=0.0, value=0.0)
credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=600)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data =pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})
geo_encoded=onehot_encoder.transform([[input_data['Geography'][0]]])
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder.get_feature_names_out(['Geography']))
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
input_data = input_data.drop('Geography', axis=1)
input_scaled=scaler.transform(input_data)
prediction=model.predict(input_scaled)
prediction_proba=prediction[0][0]
if prediction_proba > 0.5:
    st.write(f"The customer is likely to exit with a probability of {prediction_proba:.2f}")
else:
    st.write(f"The customer is not likely to exit with a probability of {(1-prediction_proba):.2f}")
