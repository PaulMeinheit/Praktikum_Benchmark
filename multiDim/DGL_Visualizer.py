import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from concurrent.futures import ProcessPoolExecutor
import copy
import time
import os
import imageio
import imageio_ffmpeg
from multiDim.Approximator_Identity_ND import Approximator_Identity_ND

#Wrapper für plotting bei parallelem Ausführen
def plotTrajectories(args):
    approximator,n_steps,delta,start_points = args
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    colors = ['#FF6F00', '#00E676', '#2979FF', '#D500F9']
    start_time = time.time()
    for idx, sp in enumerate(start_points):
        traj = [sp.copy()]
        p = sp.copy()
        for i in range(n_steps):
            direction = approximator.predict(p.reshape(1, -1))[0]
            p = p + delta * direction
            traj.append(p.copy())
            #if i%2000==1:
                #print(direction)
            #    print(f"{approximator.name}-simulation at starting_point: {idx} & step: {i}")
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=colors[idx % len(colors)], linewidth=1.2, alpha=0.9)
    print(f"✅ {approximator.name} plotting done in {time.time() - start_time:.4f}s")
    ax.set_xlabel("x", color='white')
    ax.set_ylabel("y", color='white')
    ax.set_zlabel("z", color='white')
    ax.tick_params(colors='white')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    title = approximator.name
    ax.set_title(title, color='white')

    plt.tight_layout()
    
    save_dir= "DGL_Visualizer"
    os.makedirs(save_dir, exist_ok=True)
    base, _ = os.path.splitext(f"trajectory_{title}")
    base += "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    
    full_path = os.path.join(save_dir, f"{base}.{"svg"}")
    plt.close(fig)
    fig.savefig(full_path, bbox_inches="tight", format="svg")
    


# Wrapper Funktion für paralleles Training
def train_apprx(args):
    apx, func_class, func_params = args
    # Reinitialisiere die Funktion, falls nötig
    function = func_class(*func_params)
    start = time.time()

    # Trainiere den Approximator auf der Funktion
    apx.train(function)
    print(f"✅ {apx.name} trained in {time.time() - start:.2f}s")
    
    # Rückgabe mit dem trainierten Modell (Apprximator)
    return (apx)

class DGL_Visualizer:
    def __init__(self, name, approximators, function, loss_fn=torch.nn.MSELoss(),
                 parallel=False,vmin=1e-17,vmax=1e50,logscale=False,max_workers=2):
        self.name = name
        self.max_workers = max_workers
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
    def save_plot(self,fig, filename, save_dir=None, ext="svg", timestamp=True):
        if save_dir==None:
            save_dir=f"{self.name}"
        os.makedirs(save_dir, exist_ok=True)
        base, _ = os.path.splitext(filename)
        if timestamp:
            base += "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
        full_path = os.path.join(save_dir, f"{base}.{ext}")

        fig.savefig(full_path, bbox_inches="tight", format=ext)
        plt.close(fig)

    def train(self):
        output_dim = self.function.outputDim
        low = np.array(self.function.inDomainStart)
        high = np.array(self.function.inDomainEnd)
        # Daten für paralleles Training vorbereiten
        func_params = (self.function.name, self.function.inputDim,self.function.outputDim, self.function.inDomainStart, self.function.inDomainEnd)
        data = [
            (copy.deepcopy(apx), self.function.__class__, func_params)
            for apx in self.approximators
        ]
        
        if self.parallel:
            with ProcessPoolExecutor() as executor:
                results_raw = list(executor.map(train_apprx, data))
        else:
            results_raw = [train_apprx(arg) for arg in data]
        
        # Update Approximatoren in-place mit trainierten Modellen
        for (i, trained_model) in enumerate(results_raw):
            self.approximators[i] = trained_model

    def plot_trajectories_3D_all(self, n_steps=10000, delta=0.001):
        """
        Plottet und speichert für jeden Approximator einzeln die Trajektorien (3D → 3D) als dunkles SVG.
        """
        print("Es sind gerade "+str(len(self.approximators))+" Approximatoren am Plotten!")
        if not self.approximators:
            print("⚠️ Keine Approximatoren übergeben.")
            return

        if self.function.inputDim != 3 or self.function.outputDim != 3:
            print("⚠️ Funktion hat nicht die Dimension 3→3. Überspringe Plot.")
            return
        
        # Startpunkte für die Trajektorien
        start_points = [
            np.array([10.0, 20.0, 51.05]),
            np.array([-5.0, 20.5, 41.05]),
            np.array([15.0, 20.5, 31.05]),
            np.array([-10.0, 10.0, 21.05]),
            np.array([10.0, 10.5, 12.05]),
            np.array([0.0, -20.5, 10.05]),
            np.array([5.0, -20.0, 50.05])
        ]
        data = [
            (copy.deepcopy(apx),n_steps,delta,start_points)
            for apx in self.approximators
        ]
        
        if self.parallel:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                results_raw = list(executor.map(plotTrajectories, data))
        else:
            results_raw = [plotTrajectories(arg) for arg in data]
        
        return
        
    def plot_trajectories_video(self, n_steps=500, delta=0.01, fps=30, save_dir=None, combine=False):
        """
        Erstellt MP4-Videos der Trajektorien-Entwicklung.
        - combine=False: Für jeden Approximator ein Video (immer mit Identity-Vergleich)
        - combine=True: Ein gemeinsames Video für alle Approximatoren + Identity
        Speichert den letzten Frame als SVG.
        """
        if save_dir is None:
            save_dir = f"{self.name}_videos"
        os.makedirs(save_dir, exist_ok=True)

        # Identity Approximator erzeugen (nicht aus self.approximators!)
        identity_apx = Approximator_Identity_ND([])
        identity_apx.train(self.function)

        start_points = [
            #np.array([10.0, 20.0, 51.05]),
            np.array([10.0, -20.5, 20.05]),
            #np.array([5.0, -20.0, 50.05])
        ]
        base_colors = ["#2D00A8", '#00E676', "#000000", '#D500F9', '#FF1744', '#00B8D4', '#FFD600']
        model_colors = ['#2D00A8', '#00E676', '#000000', '#D500F9', '#FF1744', '#00B8D4', '#FFD600']

        my_line_width = 1.0

        my_alpha = 0.95
        my_dpi = 150
        my_figsize = (9.6, 5.4)
        if combine:
            # --- Gemeinsames Video für alle Modelle + Identity ---
            all_models = [identity_apx] + self.approximators
            model_names = ["Identity"] + [getattr(apx, "name", f"Model_{i+1}") for i, apx in enumerate(self.approximators)]
            trajs = [ [ [sp.copy()] for sp in start_points ] for _ in all_models ]  # [modell][startpunkt][zeit]
            frames = []

            for step in range(n_steps):
                for m, model in enumerate(all_models):
                    for idx, sp in enumerate(start_points):
                        p = trajs[m][idx][-1]
                        direction = model.predict(p.reshape(1, -1))[0]
                        p_new = p + delta * direction
                        trajs[m][idx].append(p_new)

                # --- Plot Frame ---
                fig = plt.figure(figsize=my_figsize, dpi=my_dpi)
                ax = fig.add_subplot(111, projection='3d')
                fig.patch.set_facecolor('black')
                ax.set_facecolor('black')
                for m, model in enumerate(all_models):
                    for idx in range(len(start_points)):
                        traj = np.array(trajs[m][idx])
                        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                                color=model_colors[m % len(model_colors)], linewidth=my_line_width, alpha=my_alpha)
                ax.set_xlabel("x", color='white')
                ax.set_ylabel("y", color='white')
                ax.set_zlabel("z", color='white')
                ax.tick_params(colors='white')
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.set_title("Alle Modelle (inkl. Identity)", color='white')

                # Legende unter dem Plot
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], color=model_colors[m % len(model_colors)], lw=2, label=model_names[m])
                    for m in range(len(all_models))
                ]
                ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.18),
                          ncol=2, frameon=False, fontsize='medium', labelcolor='white')

                plt.tight_layout()
                fig.canvas.draw()
                frame = np.array(fig.canvas.renderer.buffer_rgba())
                frame = frame[..., :3]  # RGB
                frames.append(frame)
                plt.close(fig)

            # --- Video speichern ---
            save_name = f"trajectory_all_models"
            save_name+= "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
            video_path = os.path.join(save_dir, save_name + ".mp4")
            imageio.mimsave(video_path, frames, fps=fps, macro_block_size=None)
            print(f"🎬 Video gespeichert: {video_path}")

            # --- Letzten Frame als SVG speichern ---
            fig = plt.figure(figsize=my_figsize, dpi=my_dpi)
            ax = fig.add_subplot(111, projection='3d')
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            for m, model in enumerate(all_models):
                for idx in range(len(start_points)):
                    traj = np.array(trajs[m][idx])
                    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                            color=model_colors[m % len(model_colors)], linewidth=my_line_width, alpha=my_alpha)
            ax.set_xlabel("x", color='white')
            ax.set_ylabel("y", color='white')
            ax.set_zlabel("z", color='white')
            ax.tick_params(colors='white')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_title("Alle Modelle (inkl. Identity)", color='white')
            legend_elements = [
                Line2D([0], [0], color=model_colors[m % len(model_colors)], lw=2, label=model_names[m])
                for m in range(len(all_models))
            ]
            ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.18),
                      ncol=2, frameon=False, fontsize='medium', labelcolor='white')
            plt.tight_layout()
            self.save_plot(fig, f"trajectory_all_models_lastframe", save_dir=save_dir, ext="svg", timestamp=True)

        else:
            # --- Einzelvideos: Jeder Approximator + Identity ---
            for approximator in self.approximators:
                frames = []
                trajs_id = [ [sp.copy()] for sp in start_points ]
                trajs_apx = [ [sp.copy()] for sp in start_points ]

                for step in range(n_steps):
                    for idx, sp in enumerate(start_points):
                        # Identity
                        p_id = trajs_id[idx][-1]
                        dir_id = identity_apx.predict(p_id.reshape(1, -1))[0]
                        p_id_new = p_id + delta * dir_id
                        trajs_id[idx].append(p_id_new)
                        # Approximator
                        p_apx = trajs_apx[idx][-1]
                        dir_apx = approximator.predict(p_apx.reshape(1, -1))[0]
                        p_apx_new = p_apx + delta * dir_apx
                        trajs_apx[idx].append(p_apx_new)

                    # --- Plot Frame ---
                    fig = plt.figure(figsize=my_figsize, dpi=my_dpi)
                    ax = fig.add_subplot(111, projection='3d')
                    fig.patch.set_facecolor('black')
                    ax.set_facecolor('black')
                    for idx in range(len(start_points)):
                        traj_id = np.array(trajs_id[idx])
                        traj_apx = np.array(trajs_apx[idx])
                        ax.plot(traj_id[:, 0], traj_id[:, 1], traj_id[:, 2], color='orange', linewidth=my_line_width, alpha=my_alpha, label="Identity" if idx==0 else "")
                        ax.plot(traj_apx[:, 0], traj_apx[:, 1], traj_apx[:, 2], color=base_colors[idx % len(base_colors)], linewidth=my_line_width, alpha=my_alpha, label=approximator.name if idx==0 else "")
                    ax.set_xlabel("x", color='white')
                    ax.set_ylabel("y", color='white')
                    ax.set_zlabel("z", color='white')
                    ax.tick_params(colors='white')
                    ax.grid(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_zticks([])
                    ax.set_title(f"{approximator.name} vs. Identity", color='white')
                    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize='medium', labelcolor='white')
                    plt.tight_layout()
                    fig.canvas.draw()
                    frame = np.array(fig.canvas.renderer.buffer_rgba())
                    frame = frame[..., :3]  # RGB
                    frames.append(frame)
                    plt.close(fig)

                # --- Video speichern ---
                save_name = f"trajectory_{approximator.name}"
                save_name += "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
                video_path = os.path.join(save_dir, f"{save_name}.mp4")
                imageio.mimsave(video_path, frames, fps=fps, macro_block_size=None)
                print(f"🎬 Video gespeichert: {video_path}")

                # --- Letzten Frame als SVG speichern ---
                fig = plt.figure(figsize=my_figsize, dpi=my_dpi)
                ax = fig.add_subplot(111, projection='3d')
                fig.patch.set_facecolor('black')
                ax.set_facecolor('black')
                for idx in range(len(start_points)):
                    traj_id = np.array(trajs_id[idx])
                    traj_apx = np.array(trajs_apx[idx])
                    ax.plot(traj_id[:, 0], traj_id[:, 1], traj_id[:, 2], color='orange', linewidth=my_line_width, alpha=my_alpha, label="Identity" if idx==0 else "")
                    ax.plot(traj_apx[:, 0], traj_apx[:, 1], traj_apx[:, 2], color='blue', linewidth=my_line_width, alpha=my_alpha, label=approximator.name if idx==0 else "")
                ax.set_xlabel("x", color='white')
                ax.set_ylabel("y", color='white')
                ax.set_zlabel("z", color='white')
                ax.tick_params(colors='white')
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.set_title(f"{approximator.name} vs. Identity", color='white')
                ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize='medium', labelcolor='white')
                plt.tight_layout()
                self.save_plot(fig, f"trajectory_{approximator.name}_lastframe", save_dir=save_dir, ext="svg", timestamp=True)
