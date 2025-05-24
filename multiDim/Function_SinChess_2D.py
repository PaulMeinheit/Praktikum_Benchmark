import numpy as np
from .FunctionND import FunctionND

class Function_Sin_2D(FunctionND):
    def __init__(self, name,inDomainStart,inDomainEnd,inputDim=2,outputDim=1):
        self.name = name
        self.inDomainStart = inDomainStart
        self.inDomainEnd = inDomainEnd
        self.inputDim = inputDim
        self.outputDim = outputDim

    

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """
        inputs: np.ndarray mit Form (n, 4), wobei jede Zeile ein Vektor [x, y, z, u] ist
        returns: np.ndarray mit Form (n,1), wobei jedes Ergebnis sin(x)*sin(y)*sin(z)*sin(u) ist
        """
        x = inputs[:, 0]
        y = inputs[:, 1]
        
        return np.sin(x) * np.sin(y)
