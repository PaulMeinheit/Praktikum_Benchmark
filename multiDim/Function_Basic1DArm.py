import numpy as np
from .FunctionND import FunctionND
from ikpy.chain import Chain
from ikpy.link import DHLink
from ikpy.link import OriginLink
from scipy.spatial.transform import Rotation as R

class Function_Basic1DArm(FunctionND):
    def __init__(self, name = "3dofarm", inputDim=1, outputDim=6,inDomainStart=[0], inDomainEnd=[2*np.pi]):
        self.name = name
        self.inDomainStart = inDomainStart
        self.inDomainEnd = inDomainEnd
        self.inputDim = inputDim
        self.outputDim = outputDim

        self.my_chain = Chain(name='1-link-arm', active_links_mask = [False,True], links=[
            OriginLink(),
            DHLink(name="joint1", d=0, a=1, alpha=0)])

    def evaluate (self,input):

        num_samples = input.shape[0]
        output = np.empty((num_samples, 6))

        for i in range(1,input.shape[1]):
            T = self.my_chain.forward_kinematics(input[i])
            position = T[:3, 3]
            rot_matrix = T[:3, :3]
            r = R.from_matrix(rot_matrix)
            rpy = r.as_euler('xyz', False)  # XYZ = roll-pitch-yaw
            pose_6d = np.concatenate((position, rpy))  # shape (6,)
            output[i] = pose_6d 

        return self.format_output_shape(output)
