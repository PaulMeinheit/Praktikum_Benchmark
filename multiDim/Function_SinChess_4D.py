import numpy as np
from .FunctionND import FunctionND


#For this function, the following values are very good: approx_fourier = Approximator_Fourier_ND(params=[5000,30])

class Function_Sin_4D(FunctionND):
    def __init__(self, name="Sinus-4D",inputDim=4,outputDim=1,inDomainStart=[0,0,0,0],inDomainEnd=[2*np.pi,2*np.pi,2*np.pi,2*np.pi]):
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
        z = inputs[:, 2]
        u = inputs[:, 3]
        
        return self.format_output_shape((np.sin(4*x) + np.sin(4*y) + np.sin(4*z) + np.sin(4*u) )* 10000)
