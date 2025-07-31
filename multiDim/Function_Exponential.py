from FunctionND import FunctionND
import numpy as np
class Function_Exponential(FunctionND):
    def __init__(self, name="Function_Exponential", inputDim=2,outputDim=1,inDomainStart=[-10,-10],inDomainEnd=[10,10]):
        self.name = name
        self.inDomainStart = inDomainStart
        self.inDomainEnd = inDomainEnd
        self.inputDim = inputDim
        self.outputDim = outputDim

    def evaluate(self, input):
        x = input[..., 0]
        y = input[..., 1]
        a= 1.0
        b= 1.0
        c= 0.5
        d= 12.0

        return 2**x + 3**y 

