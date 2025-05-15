import matplotlib.pyplot as plt
import numpy as np

class Experiment:
    def __init__(self, approximators, function):
        self.approximators = approximators
        self.function = function

    def run(self):
        x_fine = np.linspace(self.function.xdomainstart, self.function.xdomainend, 100)
        y_fine = np.linspace(self.function.ydomainstart, self.function.ydomainend, 100)
        X, Y = np.meshgrid(x_fine, y_fine)

        Z_true = self.function.evaluate(X, Y)

        n_approx = len(self.approximators)
        
        # Erstelle eine Figur mit 1 Reihe, n_approx Spalten
        fig, axes = plt.subplots(1, n_approx+1, figsize=(5*(n_approx+1), 4))
        
        # Erster Plot: Originalfunktion
        ax = axes[0]
        im = ax.imshow(Z_true, extent=[self.function.xdomainstart, self.function.xdomainend,
                                      self.function.ydomainstart, self.function.ydomainend],
                       origin='lower', cmap='plasma', aspect='auto')
        ax.set_title("Original Function")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Plotte jede Approximation in eigenen Subplots
        for i, approximator in enumerate(self.approximators):
            approximator.train(self.function)
            Z_pred = approximator.predict(X, Y)

            mse = np.mean((Z_true - Z_pred) ** 2)
            
            ax = axes[i+1]
            im = ax.imshow(Z_pred, extent=[self.function.xdomainstart, self.function.xdomainend,
                                          self.function.ydomainstart, self.function.ydomainend],
                           origin='lower', cmap='viridis', aspect='auto')
            ax.set_title(f'Approximator {i+1} (MSE: {mse:.4f})')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

        return [approximator.predict(X, Y) for approximator in self.approximators]
