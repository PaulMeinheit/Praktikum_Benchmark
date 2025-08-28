from multiDim.FunctionND import FunctionND
import numpy as np
class Function_Lin(FunctionND):
    def __init__(self, name="Function_Linear", inputDim=2,outputDim=1,inDomainStart=[-30,-30],inDomainEnd=[30,30]):
        self.name = name
        self.inDomainStart = inDomainStart
        self.inDomainEnd = inDomainEnd
        self.inputDim = inputDim
        self.outputDim = outputDim

    def evaluate(self, input):
        x = input[..., 0]
        y = input[..., 1]
        
        a= 8.0
        b= 16.0
        d= 64.0
        return self.format_output_shape(a*x + b*y + d)