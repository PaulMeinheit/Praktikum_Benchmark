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

    def evaluate(self, input):
        num_samples = input.shape[0]
        output = np.empty((num_samples, 6))

        for i in range(num_samples):
            joint_angles = input[i]
            full_joint_vector = [0.0] + joint_angles.tolist()  # prepend dummy for OriginLink
            T = self.my_chain.forward_kinematics(full_joint_vector)
            position = T[:3, 3]
            rot_matrix = T[:3, :3]
            r = R.from_matrix(rot_matrix)
            rpy = r.as_euler('xyz', degrees=False)
            pose_6d = np.concatenate((position, rpy))
            output[i] = pose_6d

        return self.format_output_shape(output)
