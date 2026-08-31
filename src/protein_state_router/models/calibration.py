"""Probability calibration fitted only on validation predictions."""

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        self.model = LogisticRegression().fit(logits.reshape(-1, 1), labels)
        return self

    def predict(self, logits: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(logits.reshape(-1, 1))[:, 1]


TemperatureScaler = PlattCalibrator
