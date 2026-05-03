from pathlib import Path
from typing import Self, Tuple

import joblib
import pandas as pd


class PredictorService:
    def __init__(
        self: Self,
        model_path: Path = Path("model/risk_model.pkl"),
    ) -> None:
        self.model_path = model_path

        data = joblib.load(self.model_path)
        self.model = data["model"]
        self.feature_names = data["features"]

    def predict(self: Self, input_dict: dict) -> Tuple[list[float], int, float]:
        X = pd.DataFrame([input_dict])[self.feature_names]
        probabilities = self.model.predict_proba(X)[0]  # [p0, p1, p2]
        risk_label = int(probabilities.argmax())
        confidence = float(probabilities[risk_label])

        return probabilities, risk_label, confidence
