import numpy as np
from multiDim.ApproximatorND import ApproximatorND
class ShepardInterpolator(ApproximatorND):
    
    def __init__(self, params, numPoints,power=2,name = "ShepardInterpolator_multi_dim"):
        self.name = name
        self.params = params
        self.power = power
        self.eps = 1e-12
        self.numPoints = numPoints
        self.inputDim = 0
        self.outputDim = 0

    def train(self, function):
        self.inputDim = function.inputDim
        self.outputDim = function.outputDim
        self.dataPoints = np.random.uniform(function.inDomainStart, function.inDomainEnd, size=(self.numPoints, self.inputDim))
        values = function.evaluate(self.dataPoints)
        self.values = np.asarray(values)  # sollte shape (numPoints, outputDim) haben

    def predict(self, input):
        query_points = np.asarray(input)  # shape (numQueryPoints, inputDim)
        numQueryPoints = query_points.shape[0]
        interpolated_values = np.empty((numQueryPoints, self.outputDim))

        for i, qp in enumerate(query_points):
            dists = np.linalg.norm(self.dataPoints - qp, axis=1)  # Distanz zu allen Trainingspunkten
            weights = 1.0 / (dists**self.power + self.eps)

            # Falls ein Punkt genau auf einem Trainingspunkt liegt, gib dessen Wert direkt zurück
            if np.any(dists < self.eps):
                interpolated_values[i, :] = self.values[np.argmin(dists)]
            else:
                # Gewichtete Summe für jede Output-Dimension separat
                weighted_sum = np.sum(weights[:, np.newaxis] * self.values, axis=0)
                sum_weights = np.sum(weights)
                interpolated_values[i, :] = weighted_sum / sum_weights

        return interpolated_values
