import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# 1. ข้อมูล XOR 200 จุด ใน 2 มิติ ช่วง [-2, 2]
def make_xor_data(n=200):
    X = torch.rand(n, 2) * 4 - 2
    y = (X[:, 0] * X[:, 1] > 0).float().unsqueeze(1)
    return X, y

X, y = make_xor_data(200)

# 2. นิยาม MLP model: Linear(2, 4) -> ReLU -> Linear(4, 1) -> Sigmoid
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

loss_fn = nn.BCELoss()
opt = optim.Adam(model.parameters(), lr=0.05)

# 3. เทรนโมเดล
EPOCHS = 200
for epoch in range(EPOCHS):
    y_hat = model(X)
    loss = loss_fn(y_hat, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if (epoch + 1) % 20 == 0:
        acc = ((y_hat > 0.5).float() == y).float().mean()
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Accuracy: {acc.item()*100:.1f}%")