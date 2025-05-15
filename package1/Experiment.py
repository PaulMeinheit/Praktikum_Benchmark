import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import time

def train_and_predict(approximator_data):
    import copy
    approximator, function_class, function_params, X, Y, Z_true = approximator_data

    # Deepcopy (Sicherheitsmaßnahme)
    approximator = copy.deepcopy(approximator)
    function = function_class(*function_params)

    start_train = time.time()
    approximator.train(function)
    train_time = time.time() - start_train

    start_eval = time.time()
    Z_pred = approximator.predict(X, Y)
    eval_time = time.time() - start_eval

    mse = np.mean((Z_true - Z_pred) ** 2)
    return approximator.name, Z_pred, mse, train_time, eval_time
 

from concurrent.futures import ProcessPoolExecutor
import copy

class Experiment:
    def __init__(self, parallel, approximators, function):
        self.approximators = approximators
        self.function = function
        self.parallel = parallel

    def run(self):
        x_fine = np.linspace(self.function.xdomainstart, self.function.xdomainend, 100)
        y_fine = np.linspace(self.function.ydomainstart, self.function.ydomainend, 100)
        X, Y = np.meshgrid(x_fine, y_fine)
        Z_true = self.function.evaluate(X, Y)

        n_approx = len(self.approximators)

        max_cols = 3
        total_plots = n_approx + 1
        n_cols = min(max_cols, total_plots)
        n_rows = (total_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).reshape(-1)

        ax = axes[0]
        im = ax.imshow(Z_true, extent=[self.function.xdomainstart, self.function.xdomainend,
                                       self.function.ydomainstart, self.function.ydomainend],
                       origin='lower', cmap='viridis', aspect='auto')
        ax.set_title("Original Function")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if not self.parallel:
            results = []
            for approximator in self.approximators:
                approximator.train(self.function)
                Z_pred = approximator.predict(X, Y)
                mse = np.mean((Z_true - Z_pred) ** 2)
                results.append((approximator.name, Z_pred, mse, None, None))
        else:
            function_class = self.function.__class__
            function_params = self.function.__dict__.values()

            data = [
                (copy.deepcopy(approximator), function_class, list(function_params),
                 X, Y, Z_true)
                for approximator in self.approximators
            ]
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(train_and_predict, data))

        for i, (name, Z_pred, mse, train_time, eval_time) in enumerate(results):
            ax = axes[i + 1]
            im = ax.imshow(Z_pred, extent=[self.function.xdomainstart, self.function.xdomainend,
                                           self.function.ydomainstart, self.function.ydomainend],
                           origin='lower', cmap='viridis', aspect='auto')
            title = f'{name}\n (MSE: {mse:.4f})'
            if train_time is not None:
                title += f'\nTrain: {train_time:.2f}s, Eval: {eval_time:.2f}s'
            ax.set_title(title)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for axis in axes[total_plots:]:
            axis.axis('off')

        plt.tight_layout()
        plt.savefig("results_large.svg", format="svg")
        if np.size(self.approximators)<4:
            plt.show()
        return [r[1] for r in results]
