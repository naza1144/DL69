from django.shortcuts import render
from django.http import StreamingHttpResponse
import torch
import torch.nn as nn
import json
from time import sleep
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def home(request):
    return render(request, 'dashboard/home.html')


def home_mlp(request):
    return render(request, 'dashboard/home_mlp.html')


def train_mlp(request):
    # ==============================================================================
    # คลาสโมเดล BreastCancerMLP ตามสเปกการบ้าน:
    # Input 30 -> Hidden(16) -> ReLU -> Hidden(8) -> ReLU -> Output(1) -> Sigmoid
    # ==============================================================================
    class BreastCancerMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(30, 16)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(16, 8)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(8, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            x = self.sigmoid(self.fc3(x))
            return x

    def generate():
        # 1. โหลดข้อมูล Breast Cancer (30 features)
        data = load_breast_cancer()
        X, y = data.data, data.target

        # แบ่ง Train/Test และ Standardize 30D Features
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # ทำ PCA 2D สำหรับวาด Scatter 2D บน Canvas
        pca = PCA(n_components=2)
        X_train_2d = pca.fit_transform(X_train_scaled) / 3.5

        # เตรียม Data Scatter Points
        pts = [
            {"x": float(X_train_2d[i, 0]), "y": float(X_train_2d[i, 1]), "l": int(y_train[i])}
            for i in range(len(y_train))
        ]

        # เตรียม 2D Grid บน PCA Space สำหรับคำนวณ Decision Boundary
        res = 35
        gx = torch.linspace(-3.5, 3.5, res)
        gy = torch.linspace(-3.5, 3.5, res)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        grid_2d = torch.stack([grid_x.ravel(), grid_y.ravel()], dim=1).numpy() * 3.5
        grid_30d = pca.inverse_transform(grid_2d)
        grid_30d_t = torch.tensor(grid_30d, dtype=torch.float32)

        # แปลงข้อมูลเป็น Tensors
        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_test_t  = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_t  = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        # 2. สร้างโมเดล, BCELoss, และ Adam Optimizer
        torch.manual_seed(42)
        model = BreastCancerMLP()
        loss_fn = nn.BCELoss()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)

        total_epochs = 200
        loss_history = []

        # 3. Real-time Training Loop
        for ep in range(1, total_epochs + 1):
            model.train()
            y_pred = model(X_train_t)
            loss = loss_fn(y_pred, y_train_t)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_history.append(round(loss.item(), 4))

            # ประเมินผล Test Accuracy, Decision Boundary Grid, และ 1D Probability Predictions
            model.eval()
            with torch.no_grad():
                test_preds = model(X_test_t)
                acc = ((test_preds > 0.5).float() == y_test_t).float().mean().item()
                grid_preds = model(grid_30d_t).reshape(res, res).tolist()
                train_probs = y_pred.squeeze().tolist()

            is_done = (ep == total_epochs)
            inference_results = []

            # 4. เมื่อเทรนเสร็จ ให้บันทึก Weights และทดสอบโหลดมาทดลองใช้ทันที
            if is_done:
                torch.save(model.state_dict(), 'breast_cancer_mlp.pth')

                eval_model = BreastCancerMLP()
                eval_model.load_state_dict(torch.load('breast_cancer_mlp.pth'))
                eval_model.eval()

                with torch.no_grad():
                    sample_preds = eval_model(X_test_t[:5]).squeeze().tolist()
                    sample_targets = y_test_t[:5].squeeze().tolist()

                    for i in range(5):
                        t_lbl = "Benign (เนื้อดี)" if sample_targets[i] == 1 else "Malignant (เนื้อร้าย)"
                        p_lbl = "Benign (เนื้อดี)" if sample_preds[i] > 0.5 else "Malignant (เนื้อร้าย)"
                        inference_results.append({
                            "sample": i + 1,
                            "true_label": t_lbl,
                            "pred_prob": round(sample_preds[i], 4),
                            "pred_label": p_lbl,
                            "correct": (sample_preds[i] > 0.5) == (sample_targets[i] == 1)
                        })

            # Yield SSE Stream (ส่ง train_probs สำหรับวาดกราฟที่ 3 แกน X-Y Separation)
            yield "data: " + json.dumps(dict(
                epoch=ep,
                loss=round(loss.item(), 4),
                acc=round(acc, 4),
                pts=pts,
                grid=grid_preds,
                res=res,
                probs=train_probs,
                loss_history=loss_history,
                total_epochs=total_epochs,
                saved=is_done,
                saved_file='breast_cancer_mlp.pth' if is_done else '',
                inference=inference_results
            )) + "\n\n"

            sleep(0.04)

    return StreamingHttpResponse(
        generate(),
        content_type="text/event-stream")


def train(request):
    def generate():
        torch.manual_seed(42)
        X = torch.randn(200, 2)
        y = ((X[:, 0] * 1.5 + X[:, 1] - 0.5) > 0).float().unsqueeze(1)
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 1), torch.nn.Sigmoid())
        loss_fn = torch.nn.BCELoss()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        for ep in range(200):
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            opt.zero_grad(); loss.backward(); opt.step()
            sleep(0.05)
            yield "data: " + json.dumps(dict(
                epoch=ep,
                loss=round(loss.item(), 4),
                acc=round(((y_hat > 0.5).float() == y).float().mean().item(), 4),
                w=[round(x, 4) for x in model[0].weight.data.tolist()[0]],
                b=round(model[0].bias.data.item(), 4)
            )) + "\n\n"
    return StreamingHttpResponse(
        generate(),
        content_type='text/event-stream')
