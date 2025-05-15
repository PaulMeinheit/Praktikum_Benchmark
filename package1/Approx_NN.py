from .Approximator import Approximator
import numpy as np
from .Function2D import Function2D
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class NN_Approximator(Approximator):
    def __init__(self, name,params):
        self.name = name
        self.function = 0
        self.epochs = params[0]
        self.samplePoints = params[1]
        self.nodesPerLayer = params[2]
        self.nn_general = NN_General(self.epochs,self.samplePoints,self.nodesPerLayer)

    def generate_data(self,function,samplePoints):
        x = np.linspace(function.xdomainstart, function.xdomainend, samplePoints)
        y = np.linspace(function.ydomainstart, function.ydomainend, samplePoints)
        X, Y = np.meshgrid(x, y)
        Z = function.evaluate(X,Y)
        return X, Y, Z

    def train(self,function):
        X, Y,Z = self.generate_data(function,self.samplePoints)

        inputs = np.stack([X.ravel(), Y.ravel()], axis=1)
        targets = Z.ravel().reshape(-1, 1)

        input_tensor = torch.tensor(inputs, dtype=torch.float32)
        target_tensor = torch.tensor(targets, dtype=torch.float32)

        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.nn_general.parameters(), lr=0.01)

        self.function = function
        for epoch in range(self.epochs):
            self.nn_general.train()
            optimizer.zero_grad()
            output = self.nn_general(input_tensor)
            loss = criterion(output, target_tensor)
            loss.backward()
            optimizer.step()

    def predict(self, x, y):
        self.nn_general.eval()
        inputs = np.stack([x.ravel(), y.ravel()], axis=1)
        input_tensor = torch.tensor(inputs, dtype=torch.float32)
        with torch.no_grad():
            pred = self.nn_general(input_tensor).cpu().numpy().reshape(300,300)
        return pred
     
class NN_General(nn.Module):
    def __init__(self,epochs,samplePoints,nodesPerLayer):
        super().__init__()
        layers = [nn.Linear(2, nodesPerLayer[0]), nn.ReLU()]
        for i in range(1, len(nodesPerLayer)):
            layers.append(nn.Linear(nodesPerLayer[i - 1], nodesPerLayer[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(nodesPerLayer[-1], 1))  # Letzter Layer: Ausgabe
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)