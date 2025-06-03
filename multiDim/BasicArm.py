import numpy as np
from .FunctionND import FunctionND
from ikpy.chain import Chain
from ikpy.link import DHLink
from ikpy.link import OriginLink

class BasicArm(FunctionND):
    def __init__(self, name = "3dofarm", inputDim=3, outputDim=6,inDomainStart=[0,0], inDomainEnd=[2*np.pi,2*np.pi]):
        self.name = name
        self.inDomainStart = inDomainStart
        self.inDomainEnd = inDomainEnd
        self.inputDim = inputDim
        self.outputDim = outputDim

        self.my_chain = Chain(name='3-link-arm', active_links_mask = [False,False,False], links=[
            OriginLink(),
            DHLink(name, d=2, a=0, bounds=None, use_symbolic_matrix=False),
            DHLink(name, d=0, a=1, bounds=None, use_symbolic_matrix=False)])

    def evaluate (self,input):
            output = 0
            for i in range(1,input.shape[1]):
                results = self.my_chain.forward_kinematics(input[i])
                output[i] = results
            return self.format_output_shape(output)
