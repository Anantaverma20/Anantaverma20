#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("model.pkl","rb")
classifier=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_Compressive_strength(Fineness,IST_Min,FST_Min,LC_MM,AC_per,IR_per,MgO_per,SO3_per,LOI_per, NC):
    
   
   
    prediction=model.predict([[Fineness,IST_Min,FST_Min,LC_MM,AC_per,IR_per,MgO_per,SO3_per,LOI_per, NC]])
    print(prediction)
    return prediction



def main():
    st.title("Compressive Strength")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit COMPRESSIVE STRENGTH PREDICTION ML APP </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Fineness = st.text_input("Fineness","Type Here")
    IST_Min = st.text_input("IST_Min","Type Here")
    FST_Min = st.text_input("FST_Min","Type Here")
    LC_MM= st.text_input("LC MM","Type Here")
    AC_per = st.text_input("AC_per","Type Here")
    IR_per = st.text_input("IR_per","Type Here")
    MgO_per = st.text_input("MgO_per","Type Here")
    SO3_per = st.text_input("SO3_per","Type Here")
    LOI_per = st.text_input("LOI_per","Type Here")
    NC = st.text_input("NC","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_Compressive_strength(Fineness,IST_Min,FST_Min,LC_MM,AC_per,IR_per,MgO_per,SO3_per,LOI_per, NC)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Try Again")
        

if __name__=='__main__':
    main()


# In[ ]:




