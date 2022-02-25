# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 22:33:19 2022

@author: samshalshar
"""

import numpy as np
import pickle
import streamlit as st
# loading the saved model
loaded_model = pickle.load(open('C:/Users/samshalshar/OneDrive/Desktop/machinelearning/deploying model/trained_model .sav' , 'rb'))

def diabetes_prediction(input_data):
    input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
 
    if (prediction[0] == 0):
        return'The person is not diabetic'
    else:
        return'The person is diabetic'
        
def main():
    st.title('Diabetes Prediction Web App')
    
    
    
    Pregnancies=st.text_input('Number Of  Pregnancies')
    Glucose=st.text_input('Glucose Level')
    BloodPressure=st.text_input('Blood Pressure Level')
    SkinThickness=st.text_input('Skin Thickness Level')
    Insulin=st.text_input('Insulin Level')
    BMI=st.text_input('BMI Value')
    DiabetesPedigreeFunction=st.text_input('DiabetesPedigreeFunction Value')
    Age=st.text_input('Age')
    
    diagnosis=''
    # creating a button for diagnosis#
    if st.button('Diabetes Test Result'):
        diagnosis= diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    st.success(diagnosis)


if __name__ == '__main__':
    main()
    
        
    
    