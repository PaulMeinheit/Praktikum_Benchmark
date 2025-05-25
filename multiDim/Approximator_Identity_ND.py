from multiDim.FunctionND import FunctionND
from multiDim.ApproximatorND import ApproximatorND
class Approximator_Identity_ND(ApproximatorND):
    def __init__(self,params, name = "Identity-Approximator"):
        self.name = name
        self.params = params
        self.inputDim = None
        self.outputDim = None
        self.function = None
    def train(self, function:FunctionND):
        self.function = function
        self.inputDim = self.function.inputDim
        self.outputDim = self.function.outputDim
        return

    def predict(self, input):
        return self.function.evaluate(input)
