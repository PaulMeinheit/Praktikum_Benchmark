import numpy as np

class FunctionND:
    def __init__(self, name, inputDim,outputDim,inDomainStart,inDomainEnd):
        self.name = name
        self.inDomainStart = inDomainStart
        self.inDomainEnd = inDomainEnd
        self.inputDim = inputDim
        self.outputDim = outputDim
        


    def format_output_shape(self, result: np.ndarray) -> np.ndarray:
        """
        Formatiert das Ergebnis der evaluate-Methode so, dass es immer
        die Form (n, outputDim) hat.
        Falls result (n,) ist und outputDim=1, wird auf (n,1) erweitert.
        Falls result schon (n, outputDim) ist, wird es unverändert zurückgegeben.
        """
        if result.ndim == 1 and self.outputDim == 1:
            return result[:, np.newaxis]
        elif result.ndim == 2 and result.shape[1] == self.outputDim:
            return result
        else:
            raise ValueError(f"Unexpected shape {result.shape} for outputDim={self.outputDim}")

    
    def evaluate(self, input):
        output = None
        return output
