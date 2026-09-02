from django.shortcuts import render
from django.http import StreamingHttpResponse
import torch
import json
from time import sleep
from .dl.models import MyMLP

# Create your views here.
def home(request):
    return render(request, 'dashboard/home.html')

def train_mlp(request):
    def generate():
        torch.manual_seed(42)
        n = 200
        X = torch.randn(n, 2)
        y = ((X[:, 0] * 1.5 + X[:, 1] - 0.5) > 0).float().unsqueeze(1)
        model = MyMLP()
        loss_fn = torch.nn.BCELoss()
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        for ep in range(200):
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sleep(0.05)
            acc = ((y_hat > 0.5).float() == y).float().mean().item()
            yield "data: " + json.dumps(dict(
                epoch=ep,
                loss=round(loss.item(), 4),
                acc=round(acc, 4)
            )) + "\n\n"
    return StreamingHttpResponse(
        generate(),
        content_type="text/event-stream")

def home_mlp(request):
    return render(request, 'dashboard/home_mlp.html')

def train(request):
    def generate():
        torch.manual_seed(42)
        X = torch.randn(200, 2)
        y = ((X[:,0]*1.5+X[:,1]-0.5)>0
                ).float().unsqueeze(1)
        model = torch.nn.Sequential(
            torch.nn.Linear(2,1), torch.nn.Sigmoid())
        loss_fn = torch.nn.BCELoss()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        for ep in range(200):
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            opt.zero_grad(); loss.backward(); opt.step()
            sleep(1)
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
