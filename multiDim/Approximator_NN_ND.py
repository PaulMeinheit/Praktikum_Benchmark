from .ApproximatorND import ApproximatorND
import numpy as np
from .FunctionND import FunctionND
import torch
import torch.nn as nn
import torch.optim as optim

class Approximator_NN_ND(ApproximatorND):
    def __init__(self, params,activationFunction=nn.ReLU(),lossCriterium=torch.nn.SmoothL1Loss()):
        self.function = 0
        self.epochs = params[0]
        self.activationFunction=activationFunction
        self.samplePoints = params[1]
        self.nodesPerLayer = params[2]
        self.nn_general = None
        self.optimizer = None
        self.learningRate = params[3] if len(params) > 3 else 0.01
        self.input_dim=None
        self.output_dim = None
        self.criterion = lossCriterium
        self.epochSum = 0
        
    def update_name(self):
        if self.input_dim is None or self.output_dim is None:
            self.name = f"NN_uninitialized"
            return
        layer_str = "-".join(map(str, [self.input_dim] + self.nodesPerLayer + [self.output_dim]))
        self.name = f"NN_{layer_str}_E{self.epochs}_N{self.samplePoints}"


    def generate_random_data(self, samplePoints:int):
        self.input_dim = self.function.inputDim
        x_start = self.function.inDomainStart
        x_end = self.function.inDomainEnd

        # Zufällige Punkte im Eingaberaum
        X = np.random.uniform(low=x_start, high=x_end, size=(self.samplePoints, self.input_dim))
        Y = self.function.evaluate(X)  # Erwartet: (samplePoints, outputDim)
        return X, Y

    def train(self, function):
        self.function = function
        self.input_dim = self.function.inputDim
        self.output_dim = self.function.outputDim

        X, Y = self.generate_random_data(self.samplePoints)

        input_tensor = torch.tensor(X, dtype=torch.float32)

        Y = np.array(Y)
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]  # shape (n,) -> (n, 1)
        target_tensor = torch.tensor(Y, dtype=torch.float32)

        self.nn_general = NN_General(self.input_dim, self.output_dim, self.nodesPerLayer,self.activationFunction)
        self.optimizer = optim.Adam(self.nn_general.parameters(), lr=self.learningRate)

        for epoch in range(self.epochs):
            self.nn_general.train()
            self.optimizer.zero_grad()
            output = self.nn_general(input_tensor)
            loss = self.criterion(output, target_tensor)
            loss.backward()
            self.optimizer.step()
        
        self.epochSum += self.epochs
        self.update_name()
    def predict(self, inputs):
        """
        inputs: np.ndarray mit Form (n, inputDim)
        returns: np.ndarray mit Form (n, outputDim)
        """
        if self.nn_general is None:
            raise RuntimeError("Das Modell wurde noch nicht trainiert!")
        self.nn_general.eval()
        input_tensor = torch.tensor(inputs, dtype=torch.float32)
        with torch.no_grad():
            pred = self.nn_general(input_tensor).cpu().numpy()
        return pred


class NN_General(nn.Module):
    def __init__(self, input_dim, output_dim, nodesPerLayer,activationFunction):
        super().__init__()
        layers = [nn.Linear(input_dim, nodesPerLayer[0]), activationFunction]
        for i in range(1, len(nodesPerLayer)):
            layers.append(nn.Linear(nodesPerLayer[i - 1], nodesPerLayer[i]))
            layers.append(activationFunction)
        layers.append(nn.Linear(nodesPerLayer[-1], output_dim))  # dynamische Output-Dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
