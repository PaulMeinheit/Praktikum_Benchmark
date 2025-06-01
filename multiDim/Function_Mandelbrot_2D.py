import numpy as np
from .FunctionND import FunctionND

class Function_Mandelbrot(FunctionND):
    def __init__(self, name="Mandelbrot", inputDim=2, outputDim=1,
                 inDomainStart=[-2.0, -1.5], inDomainEnd=[1.0, 1.5], max_iter=1000):
        self.name = name
        self.inDomainStart = inDomainStart
        self.inDomainEnd = inDomainEnd
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.max_iter = max_iter

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """
        inputs: np.ndarray mit Form (n, 2), wobei jede Zeile ein Vektor [x, y] ist
        returns: np.ndarray mit Form (n,1), wobei jedes Ergebnis 1 ist, wenn (x + iy) in der Mandelbrot-Menge liegt, sonst 0
        """
        x = inputs[:, 0]
        y = inputs[:, 1]
        c = x + 1j * y
        z = np.zeros_like(c, dtype=np.complex128)
        mask = np.ones_like(x, dtype=np.uint8)

        for i in range(self.max_iter):
            z = z*z + c
            diverged = np.abs(z) > 2
            mask[diverged] = 0
            z[diverged] = 2  # optional: vermeidet unnötige Rechnungen bei bereits divergierten Werten

        return self.format_output_shape(mask)
