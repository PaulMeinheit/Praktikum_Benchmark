import numpy as np
from .ApproximatorND import ApproximatorND

class Approximator_Fourier_ND(ApproximatorND):
    def __init__(self, params=[5000,20]):
        # params = [samplePoints, max_frequency]
        
        self.samplePoints = params[0]
        self.max_frequency = params[1]  # Anzahl Frequenzen pro Dimension
        self.coeffs = None
        self.name = None
        self.input_dim = None
        self.output_dim = None
        self.function = None

    def update_name(self):
        if self.input_dim is None or self.output_dim is None:
            self.name = f"Fourier_uninitialized"
            return
        self.name = f"Fourier_Regressor_N{self.samplePoints}_Freq_per_Dim{self.max_frequency}"


    def _fourier_features(self, X):
        # X shape (n_samples, input_dim)
        n_samples, input_dim = X.shape
        features = []
        for freq in range(self.max_frequency + 1):
            for d in range(input_dim):
                features.append(np.sin(2 * np.pi * freq * X[:, d]))
                features.append(np.cos(2 * np.pi * freq * X[:, d]))
        return np.column_stack(features)  # shape (n_samples, n_features)

    def generate_random_data(self, samplePoints):
        x_start = self.function.inDomainStart
        x_end = self.function.inDomainEnd
        X = np.random.uniform(low=x_start, high=x_end, size=(samplePoints, self.input_dim))
        Y = self.function.evaluate(X)
        return X, Y

    def train(self, function):
        self.function = function
        self.input_dim = function.inputDim
        self.output_dim = function.outputDim
        self.update_name()
        
        X, Y = self.generate_random_data(self.samplePoints)
        Phi = self._fourier_features(X)  # Feature Matrix (n_samples, n_features)
        
        # Einfache Least Squares für jeden Output
        # Falls Y mehrdimensional: shape (n_samples, output_dim)
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        
        # Lineares Lösen Phi * coeffs = Y für alle Outputs
        self.coeffs = np.linalg.lstsq(Phi, Y, rcond=None)[0]  # shape (n_features, output_dim)

    def predict(self, inputs):
        Phi = self._fourier_features(inputs)  # shape (n_samples, n_features)
        Y_pred = Phi @ self.coeffs  # (n_samples, output_dim)
        return Y_pred
