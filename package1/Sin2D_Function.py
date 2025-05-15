import numpy as np
from .Function2D import Function2D


class Sin2D_Function(Function2D):
    def __init__(self, name, xdomainstart,xdomainend,ydomainstart,ydomainend):
        self.name = name
        self.xdomainstart = xdomainstart
        self.xdomainend = xdomainend
        self.ydomainstart = ydomainstart
        self.ydomainend = ydomainend

    def evaluate(self, x,y):
        return np.multiply(np.sin(x), np.cos(y))