import pickle
import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

pipe = pickle.load(open('model_quikr_car_price_prediction.pkl', 'rb'))

data_cleaned = pd.read_csv('cleaned_data.csv')

st.title('Quikr Car Price Prediction')
st.text('It takes 6 inputs: \n 1. Name\n 2. Company\n 3. Year\n 4. Price\n 5. KMS driven\n 6. Fuel Type')
st.text('It uses Machine Learning Algorithm to predict the cost of car.')
st.header("Please enter the values below:")

a=st.selectbox('Name', (data_cleaned['name'].sort_values().unique().tolist()))
b=st.selectbox('Company', (data_cleaned['company'].sort_values().unique().tolist()))
c=st.number_input('Year', 1995)
d=st.number_input('KMS driven', 0.000)
e=st.radio('Fuel Type', ('Petrol', 'Diesel', 'LPG'))

input_data = np.array([[a,b,c,d,e]])
input_data = pd.DataFrame(input_data, columns=['name','company','year','kms_driven','fuel_type'])

if st.button("Estimate Price", key="predict"):
    if (b == np.array(data_cleaned[data_cleaned['name'] == a]['company'])[0]):
        output = pipe.predict(input_data)
        st.success("Price of car : Rs " + str(round(output[0][0], 2)))
    else:
        print(b, np.array(data_cleaned[data_cleaned['name'] == a]['company'])[0])
        st.warning("Wait!!!\n\nThe company of '{}' car should be '{}'\n\nPlease select a valid company.".format(a, np.array(data_cleaned[data_cleaned['name'] == a]['company'])[0]))


