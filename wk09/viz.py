import matplotlib.pyplot as plt
import torch

torch.manual_seed(42)
N = 200
X = torch.randn(N, 2)
y = ((X[:, 0] * 1.5 + X[:, 1] - 0.5)
     > 0).float().unsqueeze(1)

model = torch.nn.Sequential(
    torch.nn.Linear(2, 1),
    torch.nn.Sigmoid(),
)                        # ยังไม่เทรนเลย!

Xp, yp = X.numpy(), y.numpy().ravel()
plt.scatter(Xp[yp == 1, 0], Xp[yp == 1, 1],
            c="tab:green", s=15)
plt.scatter(Xp[yp == 0, 0], Xp[yp == 0, 1],
            c="tab:red", s=15)
plt.show()