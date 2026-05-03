from fastapi import APIRouter
from schemas.request import SessionRequest
from schemas.response import PredictionResponse
from services.predictor import PredictorService

router = APIRouter()
predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        predictor = PredictorService()
    return predictor


@router.post("/predict", response_model=PredictionResponse)
def predict(data: SessionRequest) -> PredictionResponse:
    predictor = get_predictor()
    probabilities, risk_label, confidence = predictor.predict(data.model_dump())

    return PredictionResponse(
        probabilities=probabilities, risk_label=risk_label, confidence=confidence
    )
