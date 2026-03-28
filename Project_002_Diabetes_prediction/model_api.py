from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Create an instance of the FastAPI class
app = FastAPI()

# load the model
with open("diabetes_model.pkl", "rb") as f:
    loaded = pickle.load(f)

model         = loaded["model"]
scaler        = loaded["scaler"]
feature_names = loaded["feature_names"]

# input schema
class model_input(BaseModel):
    Pregnancies             : int
    Glucose                 : int
    BloodPressure           : int
    SkinThickness           : int
    Insulin                 : int
    BMI                     : float
    DiabetesPedigreeFunction: float
    Age                     : int 


@app.post("/diabetes_predict")
def predict_diabetes(input_parameters: model_input):
    # convert input to dictionary directly
    input_dict = input_parameters.model_dump()
    # create a list of features in the same order as the model expects
    input_list = [input_dict[feature] for feature in feature_names]
    # scale the input features
    input_scaled = scaler.transform([input_list])
    # make a prediction using the loaded model
    prediction = model.predict(input_scaled)
    # return the prediction as a JSON response
    if prediction[0] == 0:
        return "the person is not diabetic"
    else:
        return "the person is diabetic"
