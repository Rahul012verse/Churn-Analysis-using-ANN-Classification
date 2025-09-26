import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

churn_model = tf.keras.models.load_model("churn_model.h5")

#Load all the pickle files:-
with open("le_gender.pkl","rb") as file:
    le_gender=pickle.load(file)

with open("scaler.pkl","rb") as file:
    scaler=pickle.load(file)

with open("scale.pkl","rb") as file:
    scale=pickle.load(file)

st.title("Churn_modelling")

# User input
geography = st.selectbox('Geography', scaler.categories_[0])
gender = st.selectbox('Gender', le_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [le_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-Hot Encoding(OHE) on Feature-"Geography":-
ohe_geo=scaler.transform([[geography]]).toarray()
ohe_geo_df=pd.DataFrame(ohe_geo,columns=scaler.get_feature_names_out(["Geography"]))

df_in=pd.concat([input_data.reset_index(drop=True), ohe_geo_df], axis=1)

# Scaling the features:-
scaled_df=scale.transform(df_in)

## Predict of churn
prediction=churn_model.predict(scaled_df)
prediction_proba = prediction[0][0]
st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')














