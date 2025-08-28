import numpy as np
from multiDim.FunctionND import FunctionND

#guter Approximator: approx_fourier = Approximator_Fourier_ND(params=[100000,200],ridge_lambda=1e-1)
class Function_Periodic_Behaviour(FunctionND):
    def __init__(self, name="Function_Periodic_Behaviour",inputDim=2,outputDim=1,inDomainStart=[-10,-10],inDomainEnd=[10,10]):
        self.name = name
        self.inDomainStart = inDomainStart
        self.inDomainEnd = inDomainEnd
        self.inputDim = inputDim
        self.outputDim = outputDim

    

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """
        inputs: np.ndarray mit Form (n, 4), wobei jede Zeile ein Vektor [x, y, z, u] ist
        returns: np.ndarray mit Form (n, 4), wobei jede Spalte eine periodische Funktion des Inputs ist
        """
        x = inputs[:, 0]
        y = inputs[:, 1]

        out1 =  + np.cos(22.1 * y) + np.sin(3.2223 * y) + np.sin(2.5 * y)
        out2 = np.sin(2.425 * np.pi * x) + np.cos(2.123 * np.pi * x) + np.cos(2.123 * np.pi * x)

        return self.format_output_shape(out1 + out2)