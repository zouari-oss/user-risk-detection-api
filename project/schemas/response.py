from pydantic import BaseModel


class PredictionResponse(BaseModel):
    probabilities: list[float]
    risk_label: int
    confidence: float
