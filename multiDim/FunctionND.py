

class FunctionND:
    def __init__(self, name, inputDim,outputDim,inDomainStart,inDomainEnd,outDomainStart,outDomainEnd):
        self.name = name
        self.inDomainStart = inDomainStart
        self.inDomainEnd = inDomainEnd
        self.outDomainStart = outDomainStart
        self.outDomainEnd = outDomainEnd
        self.inputDim = inputDim
        self.outputDim = outputDim
        

    def evaluate(self, input):
        output = None
        return output
