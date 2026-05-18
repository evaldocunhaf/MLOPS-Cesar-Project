import logging
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel as PydanticModel
from pydantic import ConfigDict, Field

load_dotenv()

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="Gaming Mental Health Classifier",
    description="Predicts academic/work performance (High/Medium/Low) from gaming and sleep habits.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/evaldocunhaf/MLOPs-Cesar.mlflow",
)
MODEL_URI = os.getenv("MODEL_URI", "models:/gaming-mental-health/latest")
LOCAL_FALLBACK_PATH = Path("models/model_pipeline.joblib")

pipeline = None
model_source = "none"


def load_model():
    """Try to load from MLflow Registry first, fall back to local .joblib."""
    global pipeline, model_source
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"Loading model from MLflow Registry: {MODEL_URI} (tracking={MLFLOW_TRACKING_URI})")
        pipeline = mlflow.sklearn.load_model(MODEL_URI)
        model_source = f"mlflow:{MODEL_URI}"
        logger.info(f"Model loaded from MLflow Registry: {MODEL_URI}")
        return
    except Exception as e:
        logger.warning(f"Could not load from MLflow Registry ({type(e).__name__}: {e}). Falling back to local file.")

    try:
        if LOCAL_FALLBACK_PATH.exists():
            pipeline = joblib.load(LOCAL_FALLBACK_PATH)
            model_source = f"local:{LOCAL_FALLBACK_PATH}"
            logger.info(f"Model loaded from local fallback: {LOCAL_FALLBACK_PATH}")
        else:
            logger.error(f"Local fallback not found at {LOCAL_FALLBACK_PATH}")
    except Exception as e:
        logger.error(f"Failed to load local fallback: {e}")


load_model()

FEATURES = [
    "age", "gender", "daily_gaming_hours", "game_genre",
    "primary_game", "gaming_platform", "sleep_hours",
    "sleep_quality", "sleep_disruption_frequency",
    "face_to_face_social_hours_weekly",
]


class PredictRequest(PydanticModel):
    age: float = Field(ge=10, le=100, description="Player age in years")
    gender: str = Field(min_length=1, description="Male | Female | Other")
    daily_gaming_hours: float = Field(ge=0, le=24)
    game_genre: str = Field(min_length=1)
    primary_game: str = Field(min_length=1)
    gaming_platform: str = Field(min_length=1, description="PC | Console | Mobile | Multi-platform")
    sleep_hours: float = Field(ge=0, le=24)
    sleep_quality: str = Field(min_length=1, description="Very Poor | Poor | Fair | Good | Insomnia")
    sleep_disruption_frequency: str = Field(min_length=1, description="Never | Rarely | Sometimes | Often | Always")
    face_to_face_social_hours_weekly: float = Field(ge=0, le=168)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 22,
                "gender": "Male",
                "daily_gaming_hours": 4.5,
                "game_genre": "FPS",
                "primary_game": "Valorant",
                "gaming_platform": "PC",
                "sleep_hours": 6.0,
                "sleep_quality": "Fair",
                "sleep_disruption_frequency": "Sometimes",
                "face_to_face_social_hours_weekly": 8.0,
            }
        }
    )


class PredictResponse(PydanticModel):
    prediction: str
    model: str


class HealthResponse(PydanticModel):
    status: str
    model_loaded: bool
    model_source: str


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    logger.exception("Unhandled error during request")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {exc.__class__.__name__}"},
    )


@app.get("/health", response_model=HealthResponse)
def health():
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline not loaded",
        )
    return HealthResponse(status="ok", model_loaded=True, model_source=model_source)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline not loaded",
        )
    row = pd.DataFrame([request.model_dump()])[FEATURES]
    pred = pipeline.predict(row)[0]
    return PredictResponse(prediction=str(pred), model=model_source)
