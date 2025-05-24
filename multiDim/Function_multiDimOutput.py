import numpy as np
from .FunctionND import FunctionND

class Function_MultiDimOutput(FunctionND):
    def __init__(self, name = "Multi-Dim-Function", inDomainStart=[-4,-4], inDomainEnd=[4,4], inputDim=2, outputDim=2):
        self.name = name
        self.inDomainStart = inDomainStart
        self.inDomainEnd = inDomainEnd
        self.inputDim = inputDim
        self.outputDim = outputDim

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """
        inputs: np.ndarray mit Form (n, inputDim)
        returns: np.ndarray mit Form (n, 2)
        
        Erste Ausgabe-Dimension: Produkt aller Eingabedimensionen
        Zweite Ausgabe-Dimension: Summe aller Eingabedimensionen
        """
        prod_all = np.prod(inputs, axis=1)
        sum_all = np.sum(inputs, axis=1)
        
        output = np.vstack([prod_all, sum_all]).T
        return output
