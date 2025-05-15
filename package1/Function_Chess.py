import numpy as np
from .Function2D import Function2D
import math as math

class Function_Chess(Function2D):
    def __init__(self, name, xdomainstart,xdomainend,ydomainstart,ydomainend):
        self.name = name
        self.xdomainstart = xdomainstart
        self.xdomainend = xdomainend
        self.ydomainstart = ydomainstart
        self.ydomainend = ydomainend

    def evaluate(self, x,y):
        return (np.floor(x) + np.floor(y)) % 2