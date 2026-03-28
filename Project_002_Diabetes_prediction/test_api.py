import json
import requests

url = "http://127.0.0.1:8000/diabetes_predict"


input_data = {
    "Pregnancies": 1,
    "Glucose": 85,
    "BloodPressure": 66,
    "SkinThickness": 29,
    "Insulin": 0,
    "BMI": 26.6,
    "DiabetesPedigreeFunction": 0.351,
    "Age": 31
}

response = requests.post(url, json=input_data)

print(response.text)

