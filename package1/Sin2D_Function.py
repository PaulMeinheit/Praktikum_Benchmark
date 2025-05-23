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
        return np.multiply(np.sin(8*(np.pi/(self.xdomainend-self.xdomainstart)) * (x-np.pi/2)), np.sin(8*(np.pi/(self.ydomainend-self.ydomainstart)) * (y-np.pi/2)))