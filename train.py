import torch

torch.manual_seed(42)

# ข้อมูลจำลอง: 200 จุดใน 2 มิติ แบ่งด้วยเส้นตรง 1.5x + y > 0.5
N = 200
X = torch.randn(N, 2)
y = ((X[:, 0] * 1.5 + X[:, 1] - 0.5) > 0).float().unsqueeze(1)

model = torch.nn.Sequential(
    torch.nn.Linear(2, 1),
    torch.nn.Sigmoid(),
)
loss_fn = torch.nn.BCELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)

EPOCH = 20
for epoch in range(EPOCH):
    y_hat = model(X)
    loss = loss_fn(y_hat, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print('Epoch: ', epoch, model[0].weight, model[0].bias)
