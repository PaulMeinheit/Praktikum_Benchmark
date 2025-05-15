class Approximator:
    def __init__(self, name,params):
        self.name = name
        self.params = params

    def train(self, function):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
