from .Approximator import Approximator
class Identity_Approximator(Approximator):
    def __init__(self, name,params):
        self.name = name
        self.function = 0
        self.params = 0


    def train(self,function):
        self.function = function

    def predict(self, x, y):
        return self.function.evaluate(x,y) 
