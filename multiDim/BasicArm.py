import numpy as np
from .FunctionND import FunctionND
import ikpy.chain as Chain
from ikpy.link import URDFLink

class BasicArm(FunctionND):
    def __init__(self, name = "3dofarm", inputDim=3, outputDim=6,inDomainStart=[0,0], inDomainEnd=[2*np.pi,2*np.pi]):
        self.name = name
        self.inDomainStart = inDomainStart
        self.inDomainEnd = inDomainEnd
        self.inputDim = inputDim
        self.outputDim = outputDim

        self.my_chain = Chain(name='3-link-arm', links=[
            URDFLink(name="base", translation=[0, 0, 0], rotation=[0, 0, 0]),
            URDFLink(name="joint1", translation=[1, 0, 0], rotation=[0, 0, 0], bounds=(-3.14, 3.14)),
            URDFLink(name="joint2", translation=[1, 0, 0], rotation=[0, 0, 0], bounds=(-3.14, 3.14)),])

    def evaluate (self,input):
            output = np.array()
            for i in range(1,input.shape[1]):
                results = self.my_chain.forward_kinematics(input[i])
                output[i] = results
            return output
