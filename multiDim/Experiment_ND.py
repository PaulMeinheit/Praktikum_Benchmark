import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LogNorm
from itertools import combinations
from sklearn.linear_model import LinearRegression
from concurrent.futures import ProcessPoolExecutor
import copy
import time

def train_and_predict(approximator_data):
    approximator, function_class, function_args, X, Y_true, loss_fn = approximator_data
    approximator = copy.deepcopy(approximator)
    function = function_class(*function_args)
    approximator.train(function)
    Y_pred = approximator.predict(X)
    loss = loss_fn(torch.tensor(Y_pred), torch.tensor(Y_true)).item()
    return approximator.name, Y_pred, loss, approximator

class Experiment_ND:
    def __init__(self, name, approximators, function, loss_fn=torch.nn.MSELoss(),
                 parallel=True, logarithmic=False, vmin=1e-3, vmax=1.0):
        self.name = name
        self.approximators = approximators
        self.function = function
        self.loss_fn = loss_fn
        self.parallel = parallel
        self.logarithmic = logarithmic
        self.vmin = vmin
        self.vmax = vmax
        self.results = []
        self.X = None
        self.Y_true = None

    def train(self):
        input_dim = self.function.inputDim
        output_dim = self.function.outputDim
        self.X = np.random.uniform(self.function.inDomainStart, self.function.inDomainEnd,
                                   size=(1000, input_dim))
        self.Y_true = self.function.evaluate(self.X)

        if self.parallel:
            data = [
                (copy.deepcopy(apx), self.function.__class__, list(self.function.__dict__.values()),
                 self.X, self.Y_true, self.loss_fn)
                for apx in self.approximators
            ]
            with ProcessPoolExecutor() as executor:
                results_raw = list(executor.map(train_and_predict, data))
        else:
            results_raw = []
            for apx in self.approximators:
                start = time.time()
                apx.train(self.function)
                Y_pred = apx.predict(self.X)
                loss = self.loss_fn(torch.tensor(Y_pred), torch.tensor(self.Y_true)).item()
                results_raw.append((apx.name, Y_pred, loss, apx))
                print(f"✅ {apx.name} trained in {time.time() - start:.2f}s")

        self.results = [{
            'name': name,
            'Y_pred': Y_pred,
            'loss': loss,
            'model': model
        } for name, Y_pred, loss, model in results_raw]

    def _dimension_ranking(self, Y, X):
        correlations = []
        for i in range(X.shape[1]):
            x_col = X[:, i].reshape(-1, 1)
            reg = LinearRegression().fit(x_col, Y)
            score = reg.score(x_col, Y)
            correlations.append(score)
        return np.argsort(correlations)[::-1]

    def plot_best_and_worst_2d_projections(self):
        for res in self.results:
            Y_pred = res['Y_pred']
            name = res['name']
            model = res['model']

            dim_ranking = self._dimension_ranking(Y_pred[:, 0], self.X)
            best_dims = dim_ranking[:2]
            worst_dims = dim_ranking[-2:]

            def _make_2d_plot(dim_pair, title_suffix):
                x = self.X[:, dim_pair[0]]
                y = self.X[:, dim_pair[1]]
                z = Y_pred[:, 0]

                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                norm = LogNorm(vmin=self.vmin, vmax=self.vmax) if self.logarithmic else None
                sc = ax.scatter(x, y, z, c=z, cmap='viridis', norm=norm)
                ax.set_xlabel(f'Input dim {dim_pair[0]}')
                ax.set_ylabel(f'Input dim {dim_pair[1]}')
                ax.set_zlabel('Output')
                ax.set_title(f"{name}: {title_suffix}")
                fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5)
                plt.tight_layout()
                plt.savefig(f"{self.name}_{name}_{title_suffix}.png")
                plt.close()

            _make_2d_plot(best_dims, "Best Dimensions")
            _make_2d_plot(worst_dims, "Worst Dimensions")
            print(f"📊 {name}: geplottet - beste {best_dims}, schlechteste {worst_dims}")

    def print_loss_summary(self):
        print("\n📉 Loss Summary:")
        for res in self.results:
            print(f"{res['name']}: Loss = {res['loss']:.6f}")
