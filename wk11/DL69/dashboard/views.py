from django.shortcuts import render
from django.http import StreamingHttpResponse
import torch
import json
from time import sleep
from .dl.models import MyMLP

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create your views here.
def home(request):
    return render(request, 'dashboard/home.html')

def home_mlp(request):
    return render(request, 'dashboard/home_mlp.html')

def train_mlp(request):
    """
    การบ้านสัปดาห์ที่ 10: MLP สำหรับ Breast Cancer dataset
    Input 30 features -> Hidden(16) -> ReLU -> Hidden(8) -> ReLU -> Output(1) -> Sigmoid
    ใช้ Adam + BCELoss, วาด loss curve, บันทึก weights ด้วย state_dict()
    """
    def _train():
        # 1. โหลดข้อมูล Breast Cancer (30 features)
        data = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
        )

        # Standardize ให้ทุก feature อยู่สเกลเดียวกัน
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        # 2. สร้าง model + loss + optimizer ตามสไลด์
        torch.manual_seed(42)
        model = MyMLP()
        loss_fn = torch.nn.BCELoss()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)

        arch = str(model)                                       # print(model)
        n_params = sum(p.numel() for p in model.parameters())

        total_epochs = 200
        loss_history = []
        acc_history = []

        # 3. Training loop: forward -> loss -> zero_grad -> backward -> step
        for ep in range(1, total_epochs + 1):
            model.train()
            y_hat = model(X_train_t)
            loss = loss_fn(y_hat, y_train_t)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_history.append(round(loss.item(), 4))

            # ประเมินผลด้วย torch.no_grad()
            model.eval()
            with torch.no_grad():
                acc = ((model(X_test_t) > 0.5).float() == y_test_t).float().mean().item()
            acc_history.append(round(acc, 4))

            is_done = (ep == total_epochs)
            inference = []

            # 4. เทรนเสร็จ -> บันทึก state_dict() แล้วโหลดกลับมาทดลองทำนาย
            if is_done:
                torch.save(model.state_dict(), 'breast_cancer_mlp.pth')

                eval_model = MyMLP()
                eval_model.load_state_dict(torch.load('breast_cancer_mlp.pth'))
                eval_model.eval()

                with torch.no_grad():
                    preds = eval_model(X_test_t[:5]).squeeze().tolist()
                    targets = y_test_t[:5].squeeze().tolist()

                for i in range(5):
                    t_lbl = "Benign (เนื้อดี)" if targets[i] == 1 else "Malignant (เนื้อร้าย)"
                    p_lbl = "Benign (เนื้อดี)" if preds[i] > 0.5 else "Malignant (เนื้อร้าย)"
                    inference.append({
                        "sample": i + 1,
                        "true_label": t_lbl,
                        "pred_label": p_lbl,
                        "pred_prob": round(preds[i], 4),
                        "correct": (preds[i] > 0.5) == (targets[i] == 1)
                    })

            sleep(0.04)

            yield "data: " + json.dumps(dict(
                epoch=ep,
                loss=round(loss.item(), 4),
                acc=round(acc, 4),
                loss_history=loss_history,
                acc_history=acc_history,
                total_epochs=total_epochs,
                arch=arch,
                n_params=n_params,
                n_train=len(y_train),
                n_test=len(y_test),
                saved=is_done,
                saved_file='breast_cancer_mlp.pth' if is_done else '',
                inference=inference
            )) + "\n\n"

    return StreamingHttpResponse(
        _train(),
        content_type='text/event-stream')

def train(request):
    def generate():
        torch.manual_seed(42)
        X = torch.randn(200, 2)
        y = ((1.5*X[:,0]+1.0*X[:,1]-0.5)>0).float().unsqueeze(1)
        model = torch.nn.Sequential(
                torch.nn.Linear(2,1), 
                torch.nn.Sigmoid()
            )
        loss_fn = torch.nn.BCELoss()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        for ep in range(200):
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            opt.zero_grad() 
            loss.backward() 
            opt.step()
            sleep(0.3)
            yield "data: " + json.dumps(dict(
                epoch=ep,
                loss=round(loss.item(),4),
                acc=round(((y_hat>0.5).float()==y).float().mean().item(),4),
                w=[round(x,4) for x in model[0].weight.data.tolist()[0]],
                b=round(model[0].bias.data.item(),4)
            )) + "\n\n"
    return StreamingHttpResponse(
        generate(),
        content_type='text/event-stream')
