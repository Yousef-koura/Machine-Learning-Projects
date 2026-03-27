from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json

# create an instance of the FastAPI class
app = FastAPI()

# create a class for the input data
class model_input(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# load the model
with open("diabetes_model.pkl", "rb") as f:
    loaded = pickle.load(f)

model   = loaded["model"]
scaler  = loaded["scaler"]
feature_names = loaded["feature_names"]

@app.post("/diabetes_predict")

def diabetes_predict(input: model_input):
    # convert input to dictionary directly
    input_dict = input.model_dump()
    # convert the dictionary values to a list in the correct order
    input_list = [input_dict[feature] for feature in feature_names]
    # scale the input data
    input_scaled = scaler.transform([input_list])
    # make a prediction
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        return {"prediction": "Diabetic 🔴"}
    else:
        return {"prediction": "Not Diabetic 🟢"}
    
    
    




