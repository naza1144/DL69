import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# 1. ข้อมูล XOR 200 จุด ใน 2 มิติ ช่วง [-2, 2]
def make_xor_data(n=200):
    torch.manual_seed(42)
    X = torch.rand(n, 2) * 4 - 2
    y = (X[:, 0] * X[:, 1] > 0).float().unsqueeze(1)
    return X, y

# 2. นิยาม MLP model: Linear(2, 4) -> ReLU -> Linear(4, 1) -> Sigmoid
def build_model():
    return nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
        nn.Sigmoid()
    )

# 3. ฟังก์ชันเทรนโมเดล
def train(epochs=200, lr=0.05):
    X, y = make_xor_data(200)
    model = build_model()
    loss_fn = nn.BCELoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            acc = ((y_hat > 0.5).float() == y).float().mean()
            print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Accuracy: {acc.item()*100:.1f}%")

    return model

if __name__ == '__main__':
    train(200)
