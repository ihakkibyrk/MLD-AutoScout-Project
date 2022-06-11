from logging import PlaceHolder
import streamlit as st
import pandas as pd
import numpy as np


html_temp = """
<div style="background-color:tomato;padding:1.5px">
<h1 style="color:white;text-align:center;">AutoScout Price Prediction</h1>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)

import pickle
filename = 'model'
model = pickle.load(open(filename, 'rb'))

scaler = pickle.load(open('scaler', 'rb'))


hp_KW = st.sidebar.slider("hp_KW:",min_value=0, max_value=300)
age = st.sidebar.slider("age:",min_value=0, max_value=10)
km = st.sidebar.number_input("km:",min_value=0, max_value=400000)
make_model = st.sidebar.selectbox("make_model", ['Audi A1', 'Audi A3', 'Opel Astra', 'Opel Corsa', 'Opel Insignia', 'Renault Clio', 'Renault Duster', 'Renault Espace'])
Gearing_Type = st.sidebar.selectbox("Gearing_Type", ['Automatic', 'Manual', 'Semi-automatic'])

my_dict = {
    "hp_KW": hp_KW,
    "age": age,
    "km": km,
    'make_model': make_model,
    'Gearing_Type' : Gearing_Type 
}
df = pd.DataFrame.from_dict([my_dict])
st.table(df)
df_last = pd.get_dummies(df)
df_last = df_last.reindex(columns=['hp_kW', 'km', 'age', 'make_model_Audi A1', 'make_model_Audi A3',
       'make_model_Opel Astra', 'make_model_Opel Corsa',
       'make_model_Opel Insignia', 'make_model_Renault Clio',
       'make_model_Renault Duster', 'make_model_Renault Espace',
       'Gearing_Type_Automatic', 'Gearing_Type_Manual',
       'Gearing_Type_Semi-automatic'], fill_value=0)


#button


if st.button('Predict'):
    df_last = scaler.transform(df_last)
    pred = model.predict(df_last)
    st.write('Predicted result is:',round(pred[0],2))
    st.balloons()
    
