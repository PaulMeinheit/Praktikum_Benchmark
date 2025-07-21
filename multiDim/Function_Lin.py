from FunctionND import FunctionND
import numpy as np
class Function_Lin(FunctionND):
    def __init__(self, name="Function_Linear", inputDim=3,outputDim=3,inDomainStart=[-30,-30,-30],inDomainEnd=[30,30,30]):
        self.name = name
        self.inDomainStart = inDomainStart
        self.inDomainEnd = inDomainEnd
        self.inputDim = inputDim
        self.outputDim = outputDim

    def evaluate(self, t):
        x = input[..., 0]
        y = input[..., 1]
        z = input[..., 2]
        a= 8.0
        b= 16.0
        c= 64.0
        d= 120.0
        return a*x + b*y + c*z + d 