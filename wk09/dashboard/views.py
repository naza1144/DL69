from django.shortcuts import render
import json
import torch
from django.http import StreamingHttpResponse
from time import sleep

def home(request):
    return render(request, 'dashboard/home.html')

def train(request):
    def generate():
        # กำหนด Seed และสร้างข้อมูลด้วย PyTorch จริง
        torch.manual_seed(42)
        X = torch.randn(200, 2)
        y = ((X[:, 0] * 1.5 + X[:, 1] - 0.5) > 0).float().unsqueeze(1)
        
        # ส่งข้อมูลจุดที่ PyTorch สร้างไปยัง Frontend
        points = [
            {'x': round(X[i, 0].item(), 3), 'y': round(X[i, 1].item(), 3), 'l': int(y[i, 0].item())}
            for i in range(200)
        ]

        # นิยามโมเดล Linear + Sigmoid ด้วย PyTorch
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 1),
            torch.nn.Sigmoid()
        )
        loss_fn = torch.nn.BCELoss()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)

        # ส่ง event แรกพร้อมจุดข้อมูล
        yield "data: " + json.dumps({
            'type': 'init',
            'points': points
        }) + "\n\n"

        # วนลูป Train Model จริงๆ ด้วย PyTorch 200 Epochs
        for ep in range(200):
            # 1. Forward Pass
            y_hat = model(X)
            # 2. Compute Loss
            loss = loss_fn(y_hat, y)
            # 3. Optimization & Backward Pass
            opt.zero_grad()
            loss.backward()
            opt.step()

            # คำนวณ Accuracy จริงจากผลทำนายของโมเดล
            acc = ((y_hat > 0.5).float() == y).float().mean().item()
            w = [round(x, 4) for x in model[0].weight.data.tolist()[0]]
            b = round(model[0].bias.data.item(), 4)

            sleep(0.06)  # หน่วงเวลาเล็กน้อยเพื่อให้ browser รับและแสดงผลสดได้ทัน

            # ส่งข้อมูลของ Epoch นี้กลับไปยัง Client ผ่าน SSE
            yield "data: " + json.dumps({
                'type': 'epoch',
                'epoch': ep + 1,
                'total_epochs': 200,
                'loss': round(loss.item(), 4),
                'acc': round(acc, 4),
                'w': w,
                'b': b
            }) + "\n\n"

    return StreamingHttpResponse(
        generate(),
        content_type='text/event-stream'
    )
