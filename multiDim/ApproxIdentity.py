import ApproximatorND
class Identity_Approximator(ApproximatorND):
    def __init__(self, name,):
        self.name = name
        self.function = 0
        


    def train(self,function):
        self.function = function

    def predict(self, input):
        return self.function.evaluate(input) 
