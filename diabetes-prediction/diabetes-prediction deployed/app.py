import joblib
from flask import Flask, jsonify, request
import pickle
import pandas as pd
import numpy as np


diabetes_predictor = joblib.load('diabetes-prediction-rfc-model.joblib')
diabetes_predictor_api = Flask("Diabetes Predictor")

@diabetes_predictor_api.get('/')
def home():
	return "Welcome to Diabetes Predictor!"

@diabetes_predictor_api.post('/predict')
def predict():
    request_data = request.get_json()
    sample = {
        'Pregnancies' : request_data['Pregnancies'],
        'Glucose' : request_data['Glucose'],
        'BloodPressure' : request_data['BloodPressure'],
        'SkinThickness' : request_data['SkinThickness'],
        'Insulin' : request_data['Insulin'],
        'BMI' : request_data['BMI'],
        'DiabetesPedigreeFunction' : request_data['DiabetesPedigreeFunction'],
        'Age' : request_data['Age']
    }

    data = pd.DataFrame([sample])
    my_prediction = diabetes_predictor.predict(data)
    if my_prediction == 1:
        label = 'Oops! You have diabetes.'
    else:
        label = 'Great! You dont have diabetes.'
        
    return jsonify(label)

if __name__ == '__main__':
	diabetes_predictor_api.run(debug=True, host='0.0.0.0', port=8000)
      
