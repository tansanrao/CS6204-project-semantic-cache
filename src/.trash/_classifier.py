import json
import math
import numpy as np

class Clasifier:

    alpha: float
    num_actions: int
    dimension: int
    regularization: float = 1.0
    min_propensity: float = 1e-3
    allow_feature_growth: bool = True

    def __init__(self):
        self.model = self.restore_model()

    def restore_model(self):
        pass

    def predict(self, features):
        """Select the TTL bucket index using LinUCB scoring."""
        vector = self._vectorize(features)
        scores = np.zeros(self._config.num_actions, dtype=np.float64)
        for action in range(self._config.num_actions):
            theta = np.linalg.solve(self._A[action], self._b[action])
            exploration = self._config.alpha * math.sqrt(
                float(vector.T @ np.linalg.solve(self._A[action], vector))
            )
            scores[action] = float(theta.T @ vector) + exploration
        chosen = int(np.argmax(scores))
        return chosen, float(scores[chosen])

    def update_model(self, featues, action, reward):
