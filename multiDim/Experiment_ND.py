import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LinearRegression
from concurrent.futures import ProcessPoolExecutor
from multiDim.Approximator_NN_ND import Approximator_NN_ND
import copy
import time
import os
from itertools import combinations
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D 

def _train_nn_with_epochs(args):
    """
    Hilfsfunktion für paralleles Training:
    args = (epochs, sample_points, nodes_per_layer, activation_function, loss_fn_class, function_class, function_args,
            n_test_points, sampling_method)
    """
    (
        epochs, sample_points, nodes_per_layer, activation_function, 
        loss_fn_class, function_class, function_args,
        n_test_points, sampling_method
    ) = args

    # Instanz der Funktion
    function = function_class(*function_args)
    loss_fn = loss_fn_class()

    nn_approx = Approximator_NN_ND(
        name=f"NN_{epochs}_epochs",
        params=[epochs, sample_points, nodes_per_layer],
        activationFunction=activation_function,
        lossCriterium=loss_fn
    )

    nn_approx.train(function)

    input_dim = function.inputDim
    domain_start = np.array(function.inDomainStart)
    domain_end = np.array(function.inDomainEnd)

    # === Punkt-Generator ===
    if sampling_method == "random":
        X_test = np.random.uniform(domain_start, domain_end, size=(n_test_points, input_dim))

    elif sampling_method == "grid":
        # Berechne Grid-Auflösung pro Dimension
        points_per_dim = int(np.ceil(n_test_points ** (1 / input_dim)))
        linspaces = [
            np.linspace(domain_start[i], domain_end[i], points_per_dim)
            for i in range(input_dim)
        ]
        mesh = np.meshgrid(*linspaces, indexing='ij')  # shape: (dim1, dim2, ..., dimD)
        X_test = np.stack(mesh, axis=-1).reshape(-1, input_dim)
        if X_test.shape[0] > n_test_points:
            X_test = X_test[:n_test_points]  # trimmen falls zu viele Punkte
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")

    # === Vorhersage und Fehlerberechnung ===
    Y_pred = nn_approx.predict(X_test)
    Y_true = function.evaluate(X_test)

    if Y_pred.shape != Y_true.shape:
        try:
            Y_pred = Y_pred.reshape(Y_true.shape)
        except:
            raise ValueError(f"Shape mismatch: Y_pred shape {Y_pred.shape}, Y_true shape {Y_true.shape}")

    mse = np.mean((Y_true - Y_pred) ** 2)
    max_norm = np.max(np.abs(Y_true - Y_pred))

    return (epochs, mse, max_norm, nn_approx)

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

def train_and_predict(args):
    # Wrapper Funktion für paralleles Training
    apx, func_class, func_params, X, Y_true, loss_fn = args
    # Reinitialisiere die Funktion, falls nötig
    function = func_class(*func_params)
    
    # Trainiere den Approximator auf der Funktion
    apx.train(function)
    
    # Vorhersage
    Y_pred = apx.predict(X)
    Y_pred = np.atleast_2d(Y_pred)
    output_dim = function.outputDim
    if Y_pred.shape[0] == 1 and Y_pred.shape[1] != output_dim:
        Y_pred = Y_pred.T
    
    # Verlust berechnen
    loss = loss_fn(to_tensor_2d(Y_pred), to_tensor_2d(Y_true)).item()
    
    # Rückgabe mit dem trainierten Modell (Apprximator)
    return (apx.name, Y_pred, loss, apx)

class Experiment_ND:
    def __init__(self, name, approximators, function, loss_fn=torch.nn.MSELoss(),
                 parallel=True, vmin=1e-3, vmax=1.0,logscale=False):
        self.name = name
        self.approximators = approximators
        self.function = function
        self.loss_fn = loss_fn
        self.parallel = parallel
        self.logscale=logscale
        self.vmin = vmin
        self.vmax = vmax
        self.results = []
        self.X = None
        self.Y_true = None

    def train(self):
        input_dim = self.function.inputDim
        output_dim = self.function.outputDim
        low = np.array(self.function.inDomainStart)
        high = np.array(self.function.inDomainEnd)
        
        self.X = np.random.uniform(low, high, size=(1000, input_dim))
        self.Y_true = self.function.evaluate(self.X)

        if self.parallel:
            # Daten für paralleles Training vorbereiten
            func_params = (self.function.name, input_dim, output_dim, self.function.inDomainStart, self.function.inDomainEnd)
            data = [
                (copy.deepcopy(apx), self.function.__class__, func_params, self.X, self.Y_true, self.loss_fn)
                for apx in self.approximators
            ]
            
            with ProcessPoolExecutor() as executor:
                results_raw = list(executor.map(train_and_predict, data))
            
            # Update Approximatoren in-place mit trainierten Modellen
            for i, (_, _, _, trained_model) in enumerate(results_raw):
                self.approximators[i] = trained_model
                
        else:
            results_raw = []
            for i, apx in enumerate(self.approximators):
                start = time.time()
                apx.train(self.function)
                Y_pred = apx.predict(self.X)
                Y_pred = np.atleast_2d(Y_pred)
                if Y_pred.shape[0] == 1 and Y_pred.shape[1] != output_dim:
                    Y_pred = Y_pred.T
                loss = self.loss_fn(to_tensor_2d(Y_pred), to_tensor_2d(self.Y_true)).item()
                results_raw.append((apx.name, Y_pred, loss, apx))
                print(f"✅ {apx.name} trained in {time.time() - start:.2f}s")
        
        # Ergebnisliste speichern
        self.results = [{
            'name': name,
            'Y_pred': Y_pred,
            'loss': loss,
            'model': model
        } for name, Y_pred, loss, model in results_raw]

    def _apply_logscale(self, values):
        if self.logscale:
            values = np.clip(values, self.vmin, self.vmax)
        return np.log10(values) if self.logscale else values

    def _label(self, base_label):
        return f"log10({base_label})" if self.logscale else base_label

    def plot_error_histograms(self, bins=50, save_dir="plots"):
        for res in self.results:
            Y_true = np.atleast_2d(self.Y_true)
            Y_pred = np.atleast_2d(res['Y_pred'])
            name = res['name']
            error = Y_true - Y_pred
            error = self._apply_logscale(error)
            output_dim = error.shape[1] if error.ndim > 1 else 1

            fig, axs = plt.subplots(output_dim, 1, figsize=(10, 3 * output_dim))
            if output_dim == 1:
                axs = [axs]

            for i in range(output_dim):
                axs[i].hist(error[:, i], bins=bins, color='lightcoral', edgecolor='black')
                axs[i].set_title(f"Fehlerverteilung – Output-Dim {i}")
                axs[i].set_xlabel(self._label("Fehlerwert"))  
                axs[i].set_ylabel("Häufigkeit")
                axs[i].grid(True)

            plt.tight_layout()
            save_plot(fig, f"{name}_error_histogram.svg", save_dir)

    def print_loss_summary(self):
        print("\n📉 Loss Summary:")
        for res in self.results:
            print(f"{res['name']}: Loss = {res['loss']:.6f}")

    def plot_1d_slices(self, resolution=2000):
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
                Y_pred = self._apply_logscale(Y_pred)
                Y_true = self._apply_logscale(Y_true)

                axs[i].plot(x_ranges[i], Y_true, label="True", color="black", linestyle="--")
                axs[i].plot(x_ranges[i], Y_pred, label="Predicted", color="blue")
                axs[i].set_title(f"1D-Schnitt – Dimension {i}")
                axs[i].set_xlabel(f"x_{i}")
                axs[i].set_ylabel(self._label("Output"))
                axs[i].legend()
                axs[i].grid(True)

            plt.tight_layout()
            save_plot(fig, f"{name}_1d_slices.svg")

    def plot_pca_querschnitt_all_outputs(self, n_points=2000, n_cols=4, save_dir="plots"):
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
        #fig.suptitle(f"{self.name} – PCA-Querschnitt, alle Output-Dimensionen", fontsize=16)

        # Alle Approximatoren vorbereiten: Vorhersagen am Raster
        preds = {}
        for res in self.results:
            pred_vals = res['model'].predict(X_grid)
            pred_vals = np.atleast_2d(pred_vals)
            if output_dim == 1:
                pred_vals = pred_vals.reshape(-1)
            pred_vals = self._apply_logscale(pred_vals)     
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
            ax.set_ylabel(self._label("Funktionswert"))
            ax.grid(True)
            ax.legend(fontsize='small')

        # Leere Subplots ausblenden (falls output_dim < n_rows * n_cols)
        for j in range(output_dim, n_rows * n_cols):
            fig.delaxes(axs[j // n_cols, j % n_cols])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_plot(fig, f"{self.name}_pca_querschnitt_all_outputs.svg", save_dir)

    def plot_norms_vs_epochs(self, epoch_list, sample_points, nodes_per_layer, activation_function=None, loss_fn_class=torch.nn.MSELoss, save_dir="plots"):
        if activation_function is None:
            activation_function = torch.nn.ReLU()

        function_class = self.function.__class__
        function_args = [self.function.name, self.function.inputDim, self.function.outputDim,
                        self.function.inDomainStart, self.function.inDomainEnd]

        # Daten für parallele Verarbeitung vorbereiten
        args_list = [
            (epochs, sample_points, nodes_per_layer, activation_function, loss_fn_class, function_class, function_args,100000,"random")
            for epochs in epoch_list
        ]

        with ProcessPoolExecutor() as executor:
            results = list(executor.map(_train_nn_with_epochs, args_list))

        # Sortieren nach Epochenzahl (falls durcheinander)
        results.sort(key=lambda x: x[0])

        epochs_sorted = [r[0] for r in results]
        mse_losses = [r[1] for r in results]
        mse_losses=self._apply_logscale(mse_losses)
        maxnorm_losses = [r[2] for r in results]
        maxnorm_losses=self._apply_logscale(maxnorm_losses)
        # Plot erzeugen
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(epochs_sorted, mse_losses, label="MSE Loss", marker='o')
        ax.plot(epochs_sorted, maxnorm_losses, label="Max Norm Loss", marker='s')
        ax.set_xlabel("Epochenzahl")
        ax.set_ylabel(self._label("Fehler"))
        ax.set_title(f"Fehler vs. Epochenzahl für NN Approximator\n({self.name})")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

        save_plot(fig, f"{self.name}_error_vs_epochs_parallel.svg", save_dir)

    def plot_vector_fields_3D_all(self, names=None, n_per_axis=7, scale=0.2):
        """
        Plottet das 3D-Vektorfeld für alle Approximatoren, wenn inputDim=3 und outputDim=3.
        
        Parameters:
        - names: Optional, Liste von Namen für die Plots
        - n_per_axis: Auflösung des Rasters
        - scale: Pfeillänge im Plot
        """
        
        if not self.approximators:
            print("⚠️ Keine Approximatoren übergeben.")
            return

        # Vorabprüfung: Alle Approximatoren müssen inputDim=3 und outputDim=3 haben
        for approximator in self.approximators:
            if self.function.inputDim != 3 or self.function.outputDim != 3:
                print(f"⚠️ Approximator '{approximator.name}' hat nicht die Dimension 3→3. Überspringe Plot.")
                return  # alternativ: continue für partielles Plotten

        # Raster erzeugen
        x = np.linspace(-1, 1, n_per_axis)
        y = np.linspace(-1, 1, n_per_axis)
        z = np.linspace(-1, 1, n_per_axis)
        X, Y, Z = np.meshgrid(x, y, z)
        grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

        num_models = len(self.approximators)
        fig = plt.figure(figsize=(6 * num_models, 6))

        for i, approximator in enumerate(self.approximators):
            vectors = approximator.predict(grid_points)  # (N, 3)
            U = vectors[:, 0].reshape(X.shape)
            V = vectors[:, 1].reshape(Y.shape)
            W = vectors[:, 2].reshape(Z.shape)

            ax = fig.add_subplot(1, num_models, i + 1, projection='3d')
            ax.quiver(X, Y, Z, U, V, W, length=scale, normalize=True, color='blue')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            title = names[i] if names and i < len(names) else getattr(approximator, "name", f"Model {i+1}")
            ax.set_title(title)

        fig.suptitle("3D-Vektorfelder der Approximatoren", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
        # Datei speichern
        save_plot(fig, "vector_fields_3D_all.svg")
