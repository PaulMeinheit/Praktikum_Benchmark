import numpy as np
from .FunctionND import FunctionND

#guter Approximator: approx_fourier = Approximator_Fourier_ND(params=[100000,200],ridge_lambda=1e-1)
class Function_Periodic_Behaviour(FunctionND):
    def __init__(self, name="Function_Periodic_Behaviour",inputDim=4,outputDim=4,inDomainStart=[-4,-4,-4,-4],inDomainEnd=[2*np.pi,2*np.pi,2*np.pi,2*np.pi]):
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
        z = inputs[:, 2]
        u = inputs[:, 3]

        out1 = np.sin(2 * np.pi * x) + np.cos(2 * np.pi * y)
        out2 = np.sin(3 * np.pi * y) + np.cos(2 * np.pi * z)
        out3 = np.sin(4 * np.pi * z + np.pi/4) + np.cos(3 * np.pi * u)
        out4 = np.sin(5 * np.pi * x) + np.sin(5 * np.pi * u)+y

        outputs = np.stack([out1, out2, out3, out4], axis=1) * 100  # optional: skalieren

        return self.format_output_shape(outputs)
