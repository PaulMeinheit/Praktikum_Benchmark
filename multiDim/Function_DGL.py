from multiDim.FunctionND import FunctionND
import numpy as np

class Function_Lorentz_DGL(FunctionND):
    def __init__(self, name="Function_Lorentz_DGL", inputDim=3, outputDim=3, inDomainStart=[-20,30,0], inDomainEnd=[20,-30,60]):
        self.name = name
        self.inDomainStart = inDomainStart
        self.inDomainEnd = inDomainEnd
        self.inputDim = inputDim
        self.outputDim = outputDim

    def evaluate(self, input):
        # input: shape (..., 3)
        sigma = 10.0
        rho = 28.0
        beta = 8.0 / 3.0
        input = np.asarray(input)
        x = input[..., 0]
        y = input[..., 1]
        z = input[..., 2]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return self.format_output_shape(np.stack([dx, dy, dz], axis=-1))

