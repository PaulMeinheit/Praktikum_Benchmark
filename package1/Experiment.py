import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from matplotlib.colors import LogNorm  ### LOG-SCALE MODE

import time

def train_and_predict(approximator_data):
    import copy
    approximator, function_class, function_params, X, Y, Z_true, logarithmic = approximator_data

    # Deepcopy (Sicherheitsmaßnahme)
    approximator = copy.deepcopy(approximator)
    function = function_class(*function_params)

    start_train = time.time()
    approximator.train(function)
    train_time = time.time() - start_train

    start_eval = time.time()
    Z_pred = approximator.predict(X, Y)
    eval_time = time.time() - start_eval
    if logarithmic:
        Z_pred = np.clip(Z_pred, 1e-8, None)  ### LOG-SCALE MOD: Verhindert log(0) oder negative Werte

    mse = np.mean((Z_true - Z_pred) ** 2)
    return approximator.name, Z_pred, mse, train_time, eval_time
 

from concurrent.futures import ProcessPoolExecutor
import copy

class Experiment:
    def __init__(self, parallel,logarithmic, approximators, function):
        self.approximators = approximators
        self.function = function
        self.parallel = parallel
        self.logarithmic = logarithmic

    def run(self):
        x_fine = np.linspace(self.function.xdomainstart, self.function.xdomainend, 300)
        y_fine = np.linspace(self.function.ydomainstart, self.function.ydomainend, 300)
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
        norm = LogNorm(vmin=Z_true.min()+1e-8, vmax=Z_true.max()) if self.logarithmic else None  ### LOG-SCALE MODE
        im = ax.imshow(Z_true, extent=[self.function.xdomainstart, self.function.xdomainend,
                                       self.function.ydomainstart, self.function.ydomainend],
                       origin='lower', cmap='viridis', aspect='auto', norm=norm)  ### LOG-SCALE MODE

        ax.set_title("Original Function")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if not self.parallel:
            function_class = self.function.__class__
            function_params = list(self.function.__dict__.values())
            results = []
            for approximator in self.approximators:
                data = (copy.deepcopy(approximator), function_class, function_params,
                        X, Y, Z_true, self.logarithmic)  ### LOG-SCALE MODE
                results.append(train_and_predict(data))  
        else:
            function_class = self.function.__class__
            function_params = self.function.__dict__.values()

            data = [
                (copy.deepcopy(approximator), function_class, list(function_params),
                 X, Y, Z_true,self.logarithmic)
                for approximator in self.approximators
            ]
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(train_and_predict, data))

        for i, (name, Z_pred, mse, train_time, eval_time) in enumerate(results):
            ax = axes[i + 1]
            norm = LogNorm(vmin=10e-4, vmax=10e3) if self.logarithmic else None  ### LOG-SCALE MOD
            im = ax.imshow(Z_pred, extent=[self.function.xdomainstart, self.function.xdomainend,
                                           self.function.ydomainstart, self.function.ydomainend],
                           origin='lower', cmap='viridis', aspect='auto', norm=norm)  ### LOG-SCALE MOD

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
