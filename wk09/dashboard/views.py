import json
import time
from django.shortcuts import render
from django.http import StreamingHttpResponse
from .ml.train import train as ml_train, make_data

def home(request):
    """
    หน้า Landing Page แสดง รหัสนักศึกษา และ ชื่อ-นามสกุล
    """
    return render(request, 'dashboard/index.html')

def train(request):
    """
    SSE Streaming Endpoint สำหรับส่งข้อมูลการฝึกสอนแบบ Real-time
    """
    def generate():
        X, y = make_data(42)
        points = [
            {'x': round(X[i, 0].item(), 3), 'y': round(X[i, 1].item(), 3), 'l': int(y[i, 0].item())}
            for i in range(200)
        ]

        # Event แรกส่งข้อมูลจุดเริ่มต้น
        yield "data: " + json.dumps({
            'type': 'init',
            'points': points
        }) + "\n\n"

        # callback สำหรับแต่ละ epoch
        def on_epoch(epoch, loss_val, acc_val, w_val, b_val, lr_res):
            data = json.dumps({
                'type': 'epoch',
                'epoch': epoch,
                'total_epochs': 200,
                'loss': loss_val,
                'accuracy': acc_val,
                'acc': acc_val,
                'w': w_val,
                'b': b_val,
                'lr_results': lr_res
            })
            time.sleep(0.04)  # delay ให้รับผลสดทัน
            return data

        # รันการฝึกสอน Perceptron ด้วยมือ
        epoch_data_list = []
        def callback_fn(epoch, loss_val, acc_val, w_val, b_val, lr_res):
            payload = json.dumps({
                'type': 'epoch',
                'epoch': epoch,
                'total_epochs': 200,
                'loss': loss_val,
                'accuracy': acc_val,
                'acc': acc_val,
                'w': w_val,
                'b': b_val,
                'lr_results': lr_res
            })
            epoch_data_list.append(payload)

        # เรียก ml/train.py
        model, lr_results = ml_train(on_progress=callback_fn, num_epochs=200, main_lr=0.1)

        for payload in epoch_data_list:
            time.sleep(0.03)
            yield f"data: {payload}\n\n"

    response = StreamingHttpResponse(generate(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response
