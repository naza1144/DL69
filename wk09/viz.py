import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# 1. สร้าง XOR Data
N = 200
X = torch.rand(N, 2) * 4 - 2  # [-2, 2]
y = (X[:, 0] * X[:, 1] > 0).float().unsqueeze(1)

# 2. นิยาม MLP: Linear(2, 4) -> ReLU -> Linear(4, 1) -> Sigmoid
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

loss_fn = nn.BCELoss()
opt = optim.Adam(model.parameters(), lr=0.05)

# ข้อมูลสำหรับ scatter
Xp, yp = X.numpy(), y.numpy().ravel()

# Grid สำหรับ Contour Plot
gx = np.linspace(-2.2, 2.2, 100)
gy = np.linspace(-2.2, 2.2, 100)
gx_grid, gy_grid = np.meshgrid(gx, gy)
grid_tensor = torch.tensor(np.c_[gx_grid.ravel(), gy_grid.ravel()], dtype=torch.float32)

plt.ion()  # interactive mode
fig, ax = plt.subplots(figsize=(6, 5))

for epoch in range(201):
    y_hat = model(X)
    loss = loss_fn(y_hat, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 20 == 0:
        ax.clear()
        with torch.no_grad():
            preds = model(grid_tensor).reshape(gx_grid.shape).numpy()

        # วาด Decision Boundary Contour
        ax.contourf(gx_grid, gy_grid, preds, levels=[0, 0.5, 1],
                    colors=['#fca5a5', '#86efac'], alpha=0.4)
        ax.contour(gx_grid, gy_grid, preds, levels=[0.5],
                   colors=['#0f172a'], linewidths=2)

        # วาดจุด XOR
        ax.scatter(Xp[yp == 1, 0], Xp[yp == 1, 1], c="tab:green", s=20, label="Class 1")
        ax.scatter(Xp[yp == 0, 0], Xp[yp == 0, 1], c="tab:red", s=20, label="Class 0")

        acc = ((y_hat > 0.5).float() == y).float().mean().item()
        ax.set_title(f"Epoch {epoch} | Loss {loss.item():.4f} | Acc {acc*100:.1f}%")
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        plt.pause(0.01)

plt.ioff()
plt.show()