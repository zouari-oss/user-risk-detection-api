from pathlib import Path
import numpy as np
import onnxruntime as ort
import json


class PredictorService:
    def __init__(self):
        base = Path(__file__).resolve().parent.parent / "model"
        self.model_path = base / "risk_model.onnx"
        self.features_path = base / "features.json"
        self.session = None
        self.input_name = None
        self.feature_names = None

    def _load(self):
        if self.session is not None:
            return

        # load feature names
        with open(self.features_path) as f:
            self.feature_names = json.load(f)

        # load ONNX model
        self.session = ort.InferenceSession(str(self.model_path))

        # cache input name
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, input_dict: dict):
        self._load()

        # Ensure correct feature order
        try:
            X = np.array(
                [[input_dict[f] for f in self.feature_names]],
                dtype=np.float32,
            )
        except KeyError as e:
            raise ValueError(f"Missing feature: {e}")

        outputs = self.session.run(None, {self.input_name: X})

        # handle ONNX output format
        probabilities = outputs[1][0] if len(outputs) > 1 else outputs[0][0]

        risk_label = int(np.argmax(probabilities))
        confidence = float(probabilities[risk_label])

        return probabilities.tolist(), risk_label, confidence
