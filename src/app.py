from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from .model import load_model, add_features, MODEL_PATH
from .monitor import log_prediction


app = FastAPI(title="Decision Recruiter Model")

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    model = None


class CandidateFeatures(BaseModel):
    objective_len: int
    title_len: int
    remuneracao: float
    job_title_len: int
    job_level: int
    job_english: int
    job_area_len: int


@app.post("/predict")
def predict(features: CandidateFeatures) -> dict:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not available")
    df = pd.DataFrame([features.dict()])
    engineered = add_features(df)
    pred = model.predict(engineered)[0]
    prob = model.predict_proba(engineered)[0, 1]
    result = {"match": int(pred), "probability": float(prob)}
    log_prediction(features.dict(), result)
    return result
