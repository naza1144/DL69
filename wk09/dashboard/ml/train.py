import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def make_data(seed=42):
    torch.manual_seed(seed)
    X = torch.randn(200, 2)
    y = ((X[:, 0] * 1.5 + X[:, 1] - 0.5) > 0).float().unsqueeze(1)
    return X, y

def run_single_train(lr, num_epochs=200):
    """
    ฝึกสอน Perceptron ด้วยมือ (Manual Update: p -= lr * p.grad)
    """
    X, y = make_data(42)
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(2, 1),
        nn.Sigmoid()
    )
    loss_fn = nn.BCELoss()

    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        y_hat = model(X)
        loss = loss_fn(y_hat, y)

        # Backward pass
        loss.backward()

        # Update weights (ด้วยมือ)
        with torch.no_grad():
            for p in model.parameters():
                p -= lr * p.grad
        model.zero_grad()

        acc = ((y_hat > 0.5).float() == y).float().mean().item()
        losses.append(loss.item())
        accuracies.append(acc)

    return losses, accuracies, losses[-1], accuracies[-1]

def train(on_progress=None, num_epochs=200, main_lr=0.1):
    """
    1. ฝึกสอนโมเดลหลักสำหรับ SSE Realtime Display (Manual Weight Update)
    2. ทดลอง 3 Learning Rates (0.001, 0.1, 1.0)
    3. บันทึก loss_curve.png ไปยัง static/dashboard/loss_curve.png
    """
    X, y = make_data(42)
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(2, 1),
        nn.Sigmoid()
    )
    loss_fn = nn.BCELoss()

    # ทดลอง 3 Learning Rates สำหรับตารางเปรียบเทียบ
    lrs = [0.001, 0.1, 1.0]
    lr_results = {}
    for l_rate in lrs:
        l_hist, a_hist, f_loss, f_acc = run_single_train(l_rate, num_epochs)
        lr_results[str(l_rate)] = {
            'lr': l_rate,
            'final_loss': round(f_loss, 4),
            'final_acc': round(f_acc * 100, 2),
            'losses': l_hist
        }

    # บันทึก loss_curve.png
    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'dashboard')
    os.makedirs(save_dir, exist_ok=True)
    img_path = os.path.join(save_dir, 'loss_curve.png')

    plt.figure(figsize=(8, 4.5))
    for l_rate in lrs:
        plt.plot(lr_results[str(l_rate)]['losses'], label=f'LR = {l_rate}')
    plt.title('Loss Curve Comparison for Learning Rates (0.001, 0.1, 1.0)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(img_path, dpi=150)
    plt.close()

    # Training Loop สำหรับ Realtime SSE Display
    for epoch in range(num_epochs):
        y_hat = model(X)
        loss = loss_fn(y_hat, y)

        loss.backward()

        # Update weights ด้วยมือ (ไม่ใช้ optimizer)
        with torch.no_grad():
            for p in model.parameters():
                p -= main_lr * p.grad
        model.zero_grad()

        acc = ((y_hat > 0.5).float() == y).float().mean().item()
        w = [round(x, 4) for x in model[0].weight.data.tolist()[0]]
        b = round(model[0].bias.data.item(), 4)

        if on_progress:
            on_progress(epoch + 1, round(loss.item(), 4), round(acc, 4), w, b, lr_results)

    return model, lr_results

if __name__ == '__main__':
    train()
