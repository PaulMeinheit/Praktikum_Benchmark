import numpy as np
from .FunctionND import FunctionND

class Function_Rotation3D(FunctionND):
    def __init__(self, name="Rotation3D", inputDim=3, outputDim=3,
                 inDomainStart=[-1, -1, -1], inDomainEnd=[1, 1, 1]):
        self.name = name
        self.inDomainStart = inDomainStart
        self.inDomainEnd = inDomainEnd
        self.inputDim = inputDim
        self.outputDim = outputDim
        
        
        # Normalisiere die Rotationsachse auf Einheitsvektor
        self.rotation_axis =np.array([0, 0, 1]) / np.linalg.norm(np.array([0, 0, 1]))

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """
        inputs: np.ndarray mit Form (n, 3), wobei jede Zeile ein Vektor [x, y, z] ist
        returns: np.ndarray mit Form (n, 3), Vektorfeld der Rotation um rotation_axis
        """
        # Kreuzprodukt rotation_axis x inputs (für jeden Punkt)
        return np.cross(np.tile(self.rotation_axis, (inputs.shape[0], 1)), inputs)
