import numpy as np
from .FunctionND import FunctionND

class Function_Sin_2D(FunctionND):
    def __init__(self, name = "Sinus-2D", inputDim=2, outputDim=1,inDomainStart=[0,0], inDomainEnd=[2*np.pi,2*np.pi]):
        self.name = name
        self.inDomainStart = inDomainStart
        self.inDomainEnd = inDomainEnd
        self.inputDim = inputDim
        self.outputDim = outputDim

    

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """
        inputs: np.ndarray mit Form (n, 2), wobei jede Zeile ein Vektor [x, y] ist
        returns: np.ndarray mit Form (n,1), wobei jedes Ergebnis sin(x)*sin(y) ist
        """
        x = inputs[:, 0]
        y = inputs[:, 1]
        
        return np.sin(x) * np.sin(y)
