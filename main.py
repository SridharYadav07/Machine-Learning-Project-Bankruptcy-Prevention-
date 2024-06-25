import fastapi
from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import numpy as np

app = FastAPI()
model = pickle.load(open('model_poly.pkl', 'rb'))

class userInput(BaseModel):
    industrial_risk : float
    management_risk : float
    financial_flexibility : float
    credibility : float
    cometitiveness : float
    operating_risk : float


@app.post('/predict')
async def predict(UserInput : userInput):
    prediction = model.predict(np.array([UserInput.industrial_risk,
                                         UserInput.management_risk,
                                         UserInput.financial_flexibility,
                                         UserInput.credibility,
                                         UserInput.cometitiveness,
                                         UserInput.operating_risk]).reshape(1, -1))
    return{'prediction': prediction.tolist()}
