import numpy as np
from .ApproximatorND import ApproximatorND
from sklearn.linear_model import Ridge
class Approximator_Fourier_ND(ApproximatorND):
    def __init__(self, params=[5000,20],ridge_lambda=1e-2):
        # params = [samplePoints, max_frequency]
        
        self.samplePoints = params[0]
        self.max_frequency = params[1]  # Anzahl Frequenzen pro Dimension
        self.coeffs = None
        self.name = None
        self.lambda_reg=ridge_lambda
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

        X_norm = self._normalize_inputs(X)

        for freq in range(self.max_frequency + 1):
            for d in range(input_dim):
                arg = 2 * np.pi * freq * X_norm[:, d]  # jetzt sind freq Perioden in Domain
                features.append(np.sin(arg))
                features.append(np.cos(arg))
        return np.column_stack(features)

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
        


        # Ridge Regression fitten (alpha ist der Regularisierungsparameter)
        ridge = Ridge(alpha=self.lambda_reg, fit_intercept=False)  
        # fit_intercept False, weil Phi schon die "Konstante" durch freq=0 enthält
        ridge.fit(Phi, Y)
        self.coeffs = ridge.coef_.T  # sklearn gibt (output_dim, n_features) zurück, wir brauchen (n_features, output_dim)
   
    def _normalize_inputs(self, X):
        """
        Normiert die Eingaben X (n_samples, input_dim) auf [0,1] basierend auf der Funktionsdomain.
        """
        start = np.array(self.function.inDomainStart)
        end = np.array(self.function.inDomainEnd)
        return (X - start) / (end - start)

    def predict(self, inputs):
        Phi = self._fourier_features(inputs)  # shape (n_samples, n_features)
        Y_pred = Phi @ self.coeffs  # (n_samples, output_dim)
        if Y_pred.ndim == 1:
            Y_pred = Y_pred[:, np.newaxis]  # macht aus (n,) -> (n,1)

        return Y_pred
