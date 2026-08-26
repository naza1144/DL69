import threading, json, torch
from django.http import JsonResponse

_state = dict(epoch=0, loss=0, acc=0,
              w=[0.0, 0.0], b=0.0, running=False)

def _train():
    torch.manual_seed(42)
    X = torch.randn(200, 2)
    y = ((X[:,0]*1.5+X[:,1]-0.5)>0
         ).float().unsqueeze(1)
    model = torch.nn.Sequential(
        torch.nn.Linear(2,1), torch.nn.Sigmoid())
    loss_fn = torch.nn.BCELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    _state['running'] = True
    for ep in range(200):
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        opt.zero_grad(); loss.backward(); opt.step()
        _state.update(
            epoch=ep,
            loss=round(loss.item(),4),
            acc=round(((y_hat>0.5).float()==y
                ).float().mean().item(),4),
            w=[round(x,4) for x in
                model[0].weight.data.tolist()[0]],
            b=round(model[0].bias.data.item(),4))
    _state['running'] = False

def train_start(request):
    threading.Thread(target=_train, daemon=True).start()
    return JsonResponse({'ok': True})

def train_state(request):
    return JsonResponse(_state)