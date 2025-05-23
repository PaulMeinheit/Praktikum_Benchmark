class ApproximatorND:
    def __init__(self, name,params):
        self.name = name
        self.params = params

    def train(self, function):
        raise NotImplementedError

    def predict(self, input):
        raise NotImplementedError
