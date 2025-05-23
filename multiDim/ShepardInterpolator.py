import ApproximatorND
import FunctionND
import numpy as np
class ShepardInterpolator (ApproximatorND):
    
    def __init__(self, name,params,power, numPoints):
        self.name = name
        self.params = params
        self.inputDim = 0
        self.outputDim = 0
        self.power = power
        self.eps = 1e-12
        self.numPoints = numPoints

    def train(self, function):
        inputDim = function.inputDim
        outputDim = function.outputDim
        self.dataPoints = np.random.uniform(function.inDomainStart,function.inDomainEnd, size=(self.numPoints, inputDim))
        values = function.evaluate(self.dataPoints)
        
        self.values = np.asarray(values)
        
    def predict(self, input):

        query_points = np.asarray(input)
        interpolated_values = np.empty(len(query_points))

        for i, qp in enumerate(query_points):
            dists = np.linalg.norm(self.data_points - qp, axis=1)
            weights = 1.0 / (dists**self.power + self.eps)

            if np.any(dists < self.eps):
                interpolated_values[i] = self.values[np.argmin(dists)]
            else:
                weighted_sum = np.sum(weights * self.values)
                sum_weights = np.sum(weights)
                interpolated_values[i] = weighted_sum / sum_weights

        return interpolated_values