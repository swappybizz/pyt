
from fastapi import FastAPI
from pydantic import BaseModel
import pickle 
import pandas as pd



app = FastAPI()

class Scoringitem(BaseModel):
    YearsAtCompany: float
    EmployeeSatisfaction :float
    Position: str
    Salaly: int



with open ("rfmodel.pkl","rb") as f:
    model = pickle.load(f) 


@app.post('/')
async def score_endpoint(item:Scoringitem):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = model.predict(df)
    return {"prediction":int(yhat)}