import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LinearRegression
from concurrent.futures import ProcessPoolExecutor
import copy
import time
import os
from itertools import combinations
from sklearn.decomposition import PCA

def save_plot(fig, filename, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    if not filename.endswith(".svg"):
        filename = filename.rsplit('.', 1)[0] + ".svg"
    fig.savefig(os.path.join(save_dir, filename), bbox_inches="tight", format='svg')
    plt.close(fig)

def to_tensor_2d(array):
    arr = np.array(array)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    return torch.tensor(arr, dtype=torch.float32)

def train_and_predict(approximator_data):
    approximator, function_class, function_args, X, Y_true, loss_fn = approximator_data
    approximator = copy.deepcopy(approximator)
    function = function_class(*function_args)
    approximator.train(function)
    Y_pred = approximator.predict(X)
    loss = loss_fn(to_tensor_2d(Y_pred), to_tensor_2d(Y_true)).item()
    return approximator.name, Y_pred, loss, approximator

class Experiment_ND:
    def __init__(self, name, approximators, function, loss_fn=torch.nn.MSELoss(),
                 parallel=True, vmin=1e-3, vmax=1.0):
        self.name = name
        self.approximators = approximators
        self.function = function
        self.loss_fn = loss_fn
        self.parallel = parallel
        self.vmin = vmin
        self.vmax = vmax
        self.results = []
        self.X = None
        self.Y_true = None

    def train(self):
        input_dim = self.function.inputDim
        output_dim = self.function.outputDim
        self.X = np.random.uniform(self.function.inDomainStart, self.function.inDomainEnd, size=(1000, input_dim))
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
                Y_pred = np.atleast_2d(Y_pred)
                if Y_pred.shape[0] == 1 and Y_pred.shape[1] != output_dim:
                    Y_pred = Y_pred.T
                loss = self.loss_fn(to_tensor_2d(Y_pred), to_tensor_2d(self.Y_true)).item()
                results_raw.append((apx.name, Y_pred, loss, apx))
                print(f"✅ {apx.name} trained in {time.time() - start:.2f}s")

        self.results = [{
            'name': name,
            'Y_pred': Y_pred,
            'loss': loss,
            'model': model
        } for name, Y_pred, loss, model in results_raw]

    def plot_error_histograms(self, bins=50, save_dir="plots"):
        for res in self.results:
            Y_true = np.atleast_2d(self.Y_true)
            Y_pred = np.atleast_2d(res['Y_pred'])
            name = res['name']
            error = Y_true - Y_pred
            output_dim = error.shape[1] if error.ndim > 1 else 1

            fig, axs = plt.subplots(output_dim, 1, figsize=(10, 3 * output_dim))
            if output_dim == 1:
                axs = [axs]

            for i in range(output_dim):
                axs[i].hist(error[:, i], bins=bins, color='lightcoral', edgecolor='black')
                axs[i].set_title(f"Fehlerverteilung – Output-Dim {i}")
                axs[i].set_xlabel("Fehlerwert")
                axs[i].set_ylabel("Häufigkeit")
                axs[i].grid(True)

            plt.tight_layout()
            save_plot(fig, f"{name}_error_histogram.svg", save_dir)

    def print_loss_summary(self):
        print("\n📉 Loss Summary:")
        for res in self.results:
            print(f"{res['name']}: Loss = {res['loss']:.6f}")



    def plot_1d_slices(self, resolution=200):
        if self.X is None:
            raise ValueError("You must call train() before plotting.")

        input_dim = self.function.inputDim
        x_ranges = [
            np.linspace(self.X[:, i].min(), self.X[:, i].max(), resolution)
            for i in range(input_dim)
        ]

        for res in self.results:
            name = res['name']
            model = res['model']
            fig, axs = plt.subplots(input_dim, 1, figsize=(8, 3 * input_dim))
            if input_dim == 1:
                axs = [axs]

            for i in range(input_dim):
                X_slice = np.tile(np.mean(self.X, axis=0), (resolution, 1))
                X_slice[:, i] = x_ranges[i]
                Y_pred = model.predict(X_slice)
                Y_true = self.function.evaluate(X_slice)

                axs[i].plot(x_ranges[i], Y_true, label="True", color="black", linestyle="--")
                axs[i].plot(x_ranges[i], Y_pred, label="Predicted", color="blue")
                axs[i].set_title(f"1D-Schnitt – Dimension {i}")
                axs[i].set_xlabel(f"x_{i}")
                axs[i].set_ylabel("Output")
                axs[i].legend()
                axs[i].grid(True)

            plt.tight_layout()
            save_plot(fig, f"{name}_1d_slices.svg")

    def plot_pca_querschnitt_all_outputs(self, n_points=200, n_cols=4, save_dir="plots"):
        X = self.X
        Y_true = self.Y_true
        output_dim = Y_true.shape[1] if Y_true.ndim > 1 else 1

        # PCA auf Inputdaten
        pca = PCA(n_components=1)
        pca_axis = pca.fit_transform(X).ravel()

        # Raster von Punkten auf der PCA-Achse
        x_vals = np.linspace(pca_axis.min(), pca_axis.max(), n_points)
        # Punkte in Original-Raum zurückprojizieren (1D -> N-D)
        X_grid = pca.inverse_transform(x_vals[:, np.newaxis])

        # Funktion am Rasterpunkt auswerten (Originalfunktion)
        Y_func = self.function.evaluate(X_grid)
        if output_dim == 1:
            Y_func = Y_func.reshape(-1)
        else:
            Y_func = np.atleast_2d(Y_func)

        # Plots erzeugen: Subplot Grid
        n_rows = (output_dim + n_cols - 1) // n_cols
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        fig.suptitle(f"{self.name} – PCA-Querschnitt, alle Output-Dimensionen", fontsize=16)

        # Alle Approximatoren vorbereiten: Vorhersagen am Raster
        preds = {}
        for res in self.results:
            pred_vals = res['model'].predict(X_grid)
            pred_vals = np.atleast_2d(pred_vals)
            if output_dim == 1:
                pred_vals = pred_vals.reshape(-1)
            preds[res['name']] = pred_vals

        # Plotten je Output-Dimension
        for i in range(output_dim):
            row = i // n_cols
            col = i % n_cols
            ax = axs[row, col]

            # Originalfunktion plotten
            if output_dim == 1:
                ax.plot(x_vals, Y_func, label="Original Funktion", linewidth=2, color='black')
            else:
                ax.plot(x_vals, Y_func[:, i], label="Original Funktion", linewidth=2, color='black')

            # Approximatoren plotten
            for name, pred_vals in preds.items():
                if output_dim == 1:
                    ax.plot(x_vals, pred_vals, label=name, linestyle='--')
                else:
                    ax.plot(x_vals, pred_vals[:, i], label=name, linestyle='--')

            ax.set_title(f"Output-Dimension {i}")
            ax.set_xlabel("PCA Komponente 1")
            ax.set_ylabel("Funktionswert")
            ax.grid(True)
            ax.legend(fontsize='small')

        # Leere Subplots ausblenden (falls output_dim < n_rows * n_cols)
        for j in range(output_dim, n_rows * n_cols):
            fig.delaxes(axs[j // n_cols, j % n_cols])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_plot(fig, f"{self.name}_pca_querschnitt_all_outputs.svg", save_dir)























#nicht mehr relevant 

    def plot_best_and_worst_2d_projections(self):
        input_dims = [0, 1]
        x = self.X[:, input_dims[0]]
        y = self.X[:, input_dims[1]]

        grid_x, grid_y = np.meshgrid(
            np.linspace(x.min(), x.max(), 100),
            np.linspace(y.min(), y.max(), 100)
        )
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

        for res in self.results:
            Y_pred = np.atleast_2d(res['Y_pred'])
            Y_true = np.atleast_2d(self.Y_true)
            name = res['name']
            loss = res['loss']
            model = res['model']

            if Y_pred.shape[0] == 1 and Y_pred.shape[1] != self.function.outputDim:
                Y_pred = Y_pred.T
            if Y_true.shape[0] == 1 and Y_true.shape[1] != self.function.outputDim:
                Y_true = Y_true.T

            output_dim = Y_pred.shape[1]
            mse_per_output = np.mean((Y_pred - Y_true) ** 2, axis=0)

            n_plot_dims = min(4, output_dim)
            best_dims = np.argsort(mse_per_output)[:n_plot_dims]
            worst_dims = np.argsort(mse_per_output)[-n_plot_dims:]

            fig, axs = plt.subplots(2, 2 * n_plot_dims, figsize=(6 * n_plot_dims, 10))
            fig.suptitle(f"{name} – Beste/Schlechteste 2D-Projektionen", fontsize=16)

            def plot(ax_pred, ax_true, out_dim, label):
                input_full = np.zeros((grid_points.shape[0], self.function.inputDim))
                for i_dim in range(self.function.inputDim):
                    if i_dim in input_dims:
                        idx = input_dims.index(i_dim)
                        input_full[:, i_dim] = grid_points[:, idx]
                    else:
                        input_full[:, i_dim] = np.mean(self.X[:, i_dim])

                pred = np.atleast_2d(model.predict(input_full))
                if pred.shape[0] == 1 and pred.shape[1] != output_dim:
                    pred = pred.T
                Z_pred = pred[:, out_dim].reshape(grid_x.shape)

                true = np.atleast_2d(self.function.evaluate(input_full))
                if true.shape[0] == 1 and true.shape[1] != output_dim:
                    true = true.T
                Z_true = true[:, out_dim].reshape(grid_x.shape)

                im1 = ax_pred.imshow(Z_pred, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower')
                ax_pred.set_title(f"Pred dim {out_dim} ({label})\nMSE={mse_per_output[out_dim]:.4f}")
                fig.colorbar(im1, ax=ax_pred)

                im2 = ax_true.imshow(Z_true, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower')
                ax_true.set_title(f"True dim {out_dim} ({label})")
                fig.colorbar(im2, ax=ax_true)

            for i, dim in enumerate(best_dims):
                plot(axs[0, 2 * i], axs[0, 2 * i + 1], dim, "Best")
            for i, dim in enumerate(worst_dims):
                plot(axs[1, 2 * i], axs[1, 2 * i + 1], dim, "Worst")

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_plot(fig, f"{name}_best_worst_heatmaps.svg")
