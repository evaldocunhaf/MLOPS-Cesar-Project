import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel as PydanticModel

app = FastAPI(title="Gaming Mental Health Classifier")

PIPELINE_PATH = Path("models/model_pipeline.joblib")
pipeline = joblib.load(PIPELINE_PATH)

FEATURES = [
    "age", "gender", "daily_gaming_hours", "game_genre",
    "primary_game", "gaming_platform", "sleep_hours",
    "sleep_quality", "sleep_disruption_frequency",
    "face_to_face_social_hours_weekly",
]


class PredictRequest(PydanticModel):
    age: float
    gender: str
    daily_gaming_hours: float
    game_genre: str
    primary_game: str
    gaming_platform: str
    sleep_hours: float
    sleep_quality: str
    sleep_disruption_frequency: str
    face_to_face_social_hours_weekly: float


class PredictResponse(PydanticModel):
    prediction: str
    model: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    row = pd.DataFrame([request.model_dump()])[FEATURES]
    pred = pipeline.predict(row)[0]
    return PredictResponse(prediction=str(pred), model=PIPELINE_PATH.stem)
