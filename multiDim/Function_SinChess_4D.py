import numpy as np
from .FunctionND import FunctionND


class Sin2D_Function(FunctionND):
    def __init__(self, name,inDomainStart,inDomainEnd,outDomainStart,outDomainEnd,inputDim=4,outputDim=1):
        self.name = name
        self.inDomainStart = inDomainStart
        self.inDomainEnd = inDomainEnd
        self.outDomainStart = outDomainStart
        self.outDomainEnd = outDomainEnd
        self.inputDim = inputDim
        self.outputDim = outputDim

import numpy as np

def evaluate(inputs: np.ndarray) -> np.ndarray:
    """
    inputs: np.ndarray mit Form (n, 4), wobei jede Zeile ein Vektor [x, y, z, u] ist
    returns: np.ndarray mit Form (n,1), wobei jedes Ergebnis sin(x)*sin(y)*sin(z)*sin(u) ist
    """
    x = inputs[:, 0]
    y = inputs[:, 1]
    z = inputs[:, 2]
    u = inputs[:, 3]
    
    return np.sin(x) * np.sin(y) * np.sin(z) * np.sin(u)
