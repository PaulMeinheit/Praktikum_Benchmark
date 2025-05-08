import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

xShape = 50
yShape = 50

# 1. Daten vorbereiten
x_vals = np.linspace(-2 * np.pi, 2 * np.pi, xShape)
y_vals = np.linspace(-2 * np.pi, 2 * np.pi, yShape)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.sin(2*X) * np.cos(2*Y)

inputs = np.stack([X.ravel(), Y.ravel()], axis=1)
targets = Z.ravel().reshape(-1, 1)

x_train = torch.tensor(inputs, dtype=torch.float32)
y_train = torch.tensor(targets, dtype=torch.float32)

# 2. Netzwerk definieren
class NN_Flach(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )

    def forward(self, x):
        return self.net(x)


# 2. Netzwerk definieren
class NN_Hoch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4,4),
            nn.ReLU(),
            nn.Linear(4,4),
            nn.ReLU(),
            nn.Linear(4,4),
            nn.ReLU(),
            nn.Linear(4,4),
            nn.ReLU(),
            nn.Linear(4,4),
            nn.ReLU(),
            nn.Linear(4,4),
            nn.ReLU(),
            nn.Linear(4,1)
        )

    def forward(self, x):
        return self.net(x)



model_NN_flach = NN_Flach()
criterion = nn.MSELoss()
optimizer = optim.Adam(model_NN_flach.parameters(), lr=0.01)

# 3. Plot vorbereiten
plt.ion()  # interaktiver Modus an
fig, ax = plt.subplots()
img = ax.imshow(np.zeros_like(Z), extent=(-2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi), origin='lower', cmap='viridis')
ax.set_title("NN Approximation während Training")
plt.colorbar(img)

qualityReached = 0.000005

# 4. Training mit Live-Update
for epoch in range(30001):
    model_NN_flach.train()
    optimizer.zero_grad()
    output = model_NN_flach(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    if loss< qualityReached:
        model_NN_flach.eval()
        with torch.no_grad():
            pred = model_NN_flach(x_train).cpu().numpy().reshape(xShape, yShape)
            img.set_data(pred)
            ax.set_title(f"Epoch {epoch}, Loss = {loss.item():.6f}")
            plt.pause(0.01)
        break
    # alle 10 Epochen: visualisieren
    if epoch % 100 == 0:
        model_NN_flach.eval()
        with torch.no_grad():
            pred = model_NN_flach(x_train).cpu().numpy().reshape(xShape, yShape)
            img.set_data(pred)
            ax.set_title(f"Epoch {epoch}, Loss = {loss.item():.6f}")
            plt.pause(0.01)

plt.ioff()
plt.show()


