import numpy as np
from .Function2D import Function2D
import math as math

class Function_Ackley(Function2D):
    def __init__(self, name, xdomainstart,xdomainend,ydomainstart,ydomainend):
        self.name = name
        self.xdomainstart = xdomainstart
        self.xdomainend = xdomainend
        self.ydomainstart = ydomainstart
        self.ydomainend = ydomainend

    def evaluate(self, x,y):
        return (-20)*np.exp((-0.2)*np.sqrt(0.5*(x**2+y**2)))-np.exp(0.5 * (np.cos(2*np.pi *x)+np.cos(2*np.pi*y))) + np.e +20