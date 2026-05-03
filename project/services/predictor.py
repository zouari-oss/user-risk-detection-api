from pathlib import Path


class PredictorService:
    def __init__(
        self,
        model_path: Path = Path(__file__).resolve().parent.parent
        / "model"
        / "risk_model.pkl",
    ):
        self.model_path = model_path
        self.model = None
        self.feature_names = None

    def _load(self):
        if self.model is not None:
            return

        import joblib

        data = joblib.load(self.model_path)
        self.model = data["model"]
        self.feature_names = data["features"]

    def predict(self, input_dict: dict):
        self._load()

        import pandas as pd

        X = pd.DataFrame([input_dict])[self.feature_names]

        if self.model is None:
            raise ValueError("Object not constructed. Cannot access a 'None' object.")

        probabilities = self.model.predict_proba(X)[0]
        risk_label = int(probabilities.argmax())
        confidence = float(probabilities[risk_label])

        return probabilities, risk_label, confidence
