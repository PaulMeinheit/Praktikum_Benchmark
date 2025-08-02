from multiDim.FunctionND import FunctionND
import numpy as np
class Function_Polynom(FunctionND):
    def __init__(self, name="Function_Polynom", inputDim=2,outputDim=1,inDomainStart=[-10,-10],inDomainEnd=[10,10]):
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
        
        d= 12.0

        return x * y + 2 * x + \
               a*np.power(x,2) + b*np.power(y,2) + d + \
               0.1*np.power(x,3) + 0.1*np.power(y,3) + \
               0.01*np.power(x,4) + 0.01*np.power(y,4) 

