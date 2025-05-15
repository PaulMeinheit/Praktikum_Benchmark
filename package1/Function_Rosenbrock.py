import numpy as np
from .Function2D import Function2D
import math as math

class Function_Rosenbrock(Function2D):
    def __init__(self, name, xdomainstart,xdomainend,ydomainstart,ydomainend):
        self.name = name
        self.xdomainstart = xdomainstart
        self.xdomainend = xdomainend
        self.ydomainstart = ydomainstart
        self.ydomainend = ydomainend

    def evaluate(self, X,Y):
        return (1-X)**2 + 100*((Y-(X**2))**2)