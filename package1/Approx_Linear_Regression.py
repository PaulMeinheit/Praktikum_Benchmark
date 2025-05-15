from .Approximator import Approximator
import numpy as np
class Approx_Linear_Regression(Approximator):
    def __init__(self, name,params):
        self.name = name
        self.function = 0
        self.n_samples=params[0]



    def train(self, function):
        # Sample Trainingspunkte zufällig aus dem Definitionsbereich der Funktion
        xs = np.random.uniform(function.xdomainstart, function.xdomainend, self.n_samples)
        ys = np.random.uniform(function.ydomainstart, function.ydomainend, self.n_samples)
        
        # Funktionswerte an den Sample-Punkten
        zs = function.evaluate(xs, ys)

        # Designmatrix für lineare Regression: [1, x, y] für Bias + 2 Variablen
        X = np.column_stack((np.ones(self.n_samples), xs, ys))
        
        # Lineare Regression: coeffs = (X^T X)^-1 X^T z
        XtX = X.T @ X
        Xtz = X.T @ zs
        
        self.coeffs = np.linalg.solve(XtX, Xtz)  # Effizienter als np.linalg.inv

    def predict(self, X, Y):
        # X, Y sind 2D Gitterpunkte (meshgrid)
        # Wir wollen Z = coeffs[0] + coeffs[1]*X + coeffs[2]*Y
        if self.coeffs is None:
            raise Exception("Model not trained yet!")

        Z = self.coeffs[0] + self.coeffs[1]*X + self.coeffs[2]*Y
        return Z