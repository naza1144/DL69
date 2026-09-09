# viz.py — ดูว่าโมเดล "เดา" ได้แย่แค่ไหน
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

loss_fn = torch.nn.BCELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)

Xp, yp = X.numpy(), y.numpy().ravel()
#plt.scatter(Xp[yp == 1, 0], Xp[yp == 1, 1], c="tab:green", s=15)
#plt.scatter(Xp[yp == 0, 0], Xp[yp == 0, 1], c="tab:red", s=15)
#plt.show()

def plot_boundary(model):
    w = model[0].weight.data.ravel()
    b = model[0].bias.data.item()
    xs = torch.linspace(-3, 3, 2)
    ys = -(w[0] * xs + b) / w[1]
    plt.plot(xs.numpy(), ys.numpy(), "k-", lw=2)
    plt.show()

for epoch in range(200):
    y_hat = model(X)
    loss = loss_fn(y_hat, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 10 == 0:      # ทุก 10 epoch
        plt.clf()            # เคลียร์กราฟเดิม
        plt.scatter(Xp[yp == 1, 0], Xp[yp == 1, 1], c="tab:green", s=15)
        plt.scatter(Xp[yp == 0, 0], Xp[yp == 0, 1], c="tab:red", s=15)
        plot_boundary(model)
        plt.title(f"epoch {epoch} | loss {loss.item():.3f}")
        plt.pause(0.001)     # ให้หน้าต่าง "เคลื่อน" สด ๆ
