from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
from django.utils import timezone
import torch
import json
import uuid
from pathlib import Path
from time import sleep


def home(request):
    return render(request, 'dashboard/home.html')


def training(request):
    return render(request, 'dashboard/training.html')


def predict_mlp(request):
    return render(request, 'dashboard/predict.html')


@csrf_exempt
def train(request):
    # Parse initial data from POST (frontend generated points + geometry)
    X = None
    y = None
    # geometry for MLPModel recording — defaults match XOR / canvas center
    label_rule = ">"
    center_x = 0.0
    center_y = 0.0
    rotation = 0.0
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode() or '{}')
            # geometry: accept multiple key aliases from frontend
            _op = data.get('op') or data.get('label_rule') or data.get('labelRule') or ">"
            if _op in (">", "<"):
                label_rule = _op
            # center / rotation may arrive as strings
            for k, v in [
                ("center_x", data.get('center_x', data.get('cx', data.get('centerX')))),
                ("center_y", data.get('center_y', data.get('cy', data.get('centerY')))),
                ("rotation", data.get('rotation', data.get('angle', data.get('rotate')))),
            ]:
                if v is not None:
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        continue
                    if k == "center_x":
                        center_x = fv
                    elif k == "center_y":
                        center_y = fv
                    elif k == "rotation":
                        rotation = fv
            pts = data.get('pts') or data.get('points') or data.get('X')
            if pts and isinstance(pts, list) and len(pts) > 0:
                if isinstance(pts[0], dict):
                    X_list = [[float(p['x']), float(p['y'])] for p in pts]
                    y_list = [[float(p.get('l', p.get('label', 0)))] for p in pts]
                elif isinstance(pts[0], (list, tuple)):
                    if len(pts[0]) >= 3:
                        X_list = [[float(p[0]), float(p[1])] for p in pts]
                        y_list = [[float(p[2])] for p in pts]
                    else:
                        X_list = [[float(p[0]), float(p[1])] for p in pts]
                        y_list = [[1.0 if float(p[0]) * 1.5 + float(p[1]) - 0.5 > 0 else 0.0] for p in pts]
                else:
                    X_list = None
                    y_list = None
                if X_list is not None:
                    X = torch.tensor(X_list, dtype=torch.float32)
                    y = torch.tensor(y_list, dtype=torch.float32)
        except Exception:
            X = None
            y = None

    def event_stream():
        nonlocal X, y, label_rule, center_x, center_y, rotation
        # wk10 spec: XOR as default (per slide); fallback synthetic perceptron data if needed elsewhere
        if X is None or y is None:
            X_local = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
            y_local = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
        else:
            X_local = X
            y_local = y

        # wk10 MLP: Linear(2,8) -> ReLU -> Linear(8,1) -> Sigmoid  (slide Part 2/3)
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
            torch.nn.Sigmoid(),
        )
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

        last_loss = None
        last_acc = None
        for epoch in range(201):
            y_hat = model(X_local)
            loss = criterion(y_hat, y_local)
            last_loss = float(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # keep SSE snappy for demo; wk09 used sleep(1), wk10 slide implies fast loop
            # sleep(0.05) balances live feel vs 201-epoch speed
            sleep(0.05)

            with torch.no_grad():
                pred_raw = model(X_local).squeeze().tolist()
                # normalize to list even for single sample
                if isinstance(pred_raw, float):
                    pred_raw = [pred_raw]
                pred = [round(float(v), 4) for v in pred_raw]
                # for perceptron-style datasets, also compute acc
                y_hat_eval = model(X_local)
                last_acc = round(((y_hat_eval > 0.5).float() == y_local).float().mean().item(), 4)
                # expose MLP weights for frontend heatmap
                w1 = [[round(float(v), 4) for v in row] for row in model[0].weight.data.tolist()]
                b1 = [round(float(v), 4) for v in model[0].bias.data.tolist()]
                w2 = [[round(float(v), 4) for v in row] for row in model[2].weight.data.tolist()]
                b2 = round(float(model[2].bias.data.item()), 4)

            yield f"data: {json.dumps(dict(epoch=epoch, loss=round(last_loss, 4), acc=last_acc, pred=pred, w1=w1, b1=b1, w2=w2, b2=b2))}\n\n"

        # --- persist: save actual trained torch model + MLPModel DB record ---
        try:
            # ensure import works even if called during streaming
            from dashboard.models import MLPModel

            save_dir = Path(settings.BASE_DIR) / "saved_models"
            save_dir.mkdir(parents=True, exist_ok=True)
            fname = f"mlp_{timezone.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.pt"
            fpath = save_dir / fname

            # save state_dict + metadata so file is self-describing
            payload = {
                "state_dict": model.state_dict(),
                "meta": {
                    "label_rule": label_rule,
                    "center_x": center_x,
                    "center_y": center_y,
                    "rotation": rotation,
                    "epochs": 201,
                    "loss": last_loss,
                    "acc": last_acc,
                    "arch": "Linear(2,8)->ReLU->Linear(8,1)->Sigmoid",
                    "optimizer": "Adam(lr=0.05)",
                    "criterion": "BCELoss",
                },
            }
            torch.save(payload, fpath)

            # store relative path for portability (fallback to absolute if outside BASE_DIR)
            try:
                rel = str(fpath.relative_to(settings.BASE_DIR))
            except ValueError:
                rel = str(fpath)

            # normalize label_rule to ">" / "<" choice
            if label_rule not in (">", "<"):
                label_rule = ">"

            MLPModel.objects.create(
                label_rule=label_rule,
                center_x=center_x,
                center_y=center_y,
                rotation=rotation,
                file_path=rel,
            )
        except Exception as e:
            # don't break SSE on save failure — surface as final event if possible
            try:
                yield f"data: {json.dumps(dict(epoch=200, loss=round(last_loss or 0, 4), acc=last_acc or 0, pred=[], error=f'save_failed: {e}'))}\n\n"
            except Exception:
                pass

    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')


# ---- helpers for predict/mlp ----
def _mlp_file_resolve(file_path: str) -> Path:
    p = Path(file_path)
    if p.is_absolute():
        return p
    # stored relative to BASE_DIR, but also handle legacy absolute
    candidate = Path(settings.BASE_DIR) / p
    if candidate.exists():
        return candidate
    return p


def _extract_weights(state_dict):
    # state_dict keys: 0.weight [8,2], 0.bias [8], 2.weight [1,8], 2.bias [1]
    w1 = state_dict.get("0.weight")
    b1 = state_dict.get("0.bias")
    w2 = state_dict.get("2.weight")
    b2 = state_dict.get("2.bias")
    if w1 is None or b1 is None or w2 is None or b2 is None:
        raise KeyError("state_dict missing expected keys 0.weight/0.bias/2.weight/2.bias")
    # to lists
    w1_l = w1.detach().cpu().tolist() if hasattr(w1, "tolist") else list(w1)
    b1_l = b1.detach().cpu().tolist() if hasattr(b1, "tolist") else list(b1)
    w2_l = w2.detach().cpu().tolist() if hasattr(w2, "tolist") else list(w2)
    b2_v = float(b2.detach().cpu().item()) if hasattr(b2, "detach") else float(b2[0] if isinstance(b2, (list, tuple)) else b2)
    # round for JSON
    w1_r = [[round(float(v), 4) for v in row] for row in w1_l]
    b1_r = [round(float(v), 4) for v in b1_l]
    w2_r = [[round(float(v), 4) for v in row] for row in w2_l]
    return w1_r, b1_r, w2_r, b2_v


def _load_mlp_payload(file_path: str):
    p = _mlp_file_resolve(file_path)
    if not p.exists():
        raise FileNotFoundError(f"model file not found: {p}")
    payload = torch.load(p, map_location="cpu")
    # payload may be {"state_dict": ..., "meta": ...} or raw state_dict
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        meta = payload.get("meta", {})
    elif isinstance(payload, dict) and "0.weight" in payload:
        state_dict = payload
        meta = {}
    else:
        # may be whole Sequential — extract state
        try:
            state_dict = payload.state_dict()
            meta = {}
        except Exception:
            raise ValueError("unsupported .pt format")
    w1, b1, w2, b2 = _extract_weights(state_dict)
    return {"state_dict": state_dict, "meta": meta, "w1": w1, "b1": b1, "w2": w2, "b2": b2, "path": str(p)}


@csrf_exempt
@require_http_methods(["GET"])
def mlp_list(request):
    from .models import MLPModel

    qs = MLPModel.objects.order_by("-create_on")
    data = []
    for m in qs:
        item = {
            "id": m.id,
            "label_rule": m.label_rule,
            "center_x": m.center_x,
            "center_y": m.center_y,
            "rotation": m.rotation,
            "file_path": m.file_path,
            "create_on": m.create_on.isoformat() if m.create_on else None,
            "update_on": m.update_on.isoformat() if m.update_on else None,
        }
        # enrich with meta if file exists (loss/acc/arch)
        try:
            loaded = _load_mlp_payload(m.file_path) if m.file_path else None
            if loaded:
                meta = loaded.get("meta", {})
                item["meta"] = meta
                # also expose loss/acc for convenience
                if "loss" in meta:
                    item["loss"] = meta["loss"]
                if "acc" in meta:
                    item["acc"] = meta["acc"]
        except Exception:
            pass
        data.append(item)
    return JsonResponse(data, safe=False)


@csrf_exempt
@require_http_methods(["GET"])
def mlp_detail(request, pk: int):
    from .models import MLPModel

    try:
        m = MLPModel.objects.get(pk=pk)
    except MLPModel.DoesNotExist:
        return JsonResponse({"error": "not found"}, status=404)
    try:
        loaded = _load_mlp_payload(m.file_path)
    except Exception as e:
        return JsonResponse({"error": f"load failed: {e}", "file_path": m.file_path}, status=500)
    return JsonResponse(
        {
            "id": m.id,
            "label_rule": m.label_rule,
            "center_x": m.center_x,
            "center_y": m.center_y,
            "rotation": m.rotation,
            "file_path": m.file_path,
            "create_on": m.create_on.isoformat() if m.create_on else None,
            "update_on": m.update_on.isoformat() if m.update_on else None,
            "meta": loaded.get("meta", {}),
            "w1": loaded["w1"],
            "b1": loaded["b1"],
            "w2": loaded["w2"],
            "b2": loaded["b2"],
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def mlp_predict(request):
    from .models import MLPModel

    try:
        body = json.loads(request.body.decode() or "{}")
    except Exception:
        return JsonResponse({"error": "invalid json"}, status=400)
    pk = body.get("model_id") or body.get("id") or body.get("pk")
    if pk is None:
        return JsonResponse({"error": "model_id required"}, status=400)
    try:
        pk = int(pk)
    except Exception:
        return JsonResponse({"error": "model_id must be int"}, status=400)
    try:
        m = MLPModel.objects.get(pk=pk)
    except MLPModel.DoesNotExist:
        return JsonResponse({"error": "model not found"}, status=404)

    pts = body.get("pts") or body.get("points") or body.get("X")
    if not pts or not isinstance(pts, list):
        return JsonResponse({"error": "pts required as list of {x,y} or [x,y,l]"}, status=400)

    # parse X
    try:
        X_list = []
        y_list = None
        has_labels = False
        if isinstance(pts[0], dict):
            X_list = [[float(p["x"]), float(p["y"])] for p in pts]
            if any("l" in p or "label" in p for p in pts):
                y_list = [[float(p.get("l", p.get("label", 0)))] for p in pts]
                has_labels = True
        elif isinstance(pts[0], (list, tuple)):
            # [x,y] or [x,y,l]
            X_list = [[float(p[0]), float(p[1])] for p in pts]
            if len(pts[0]) >= 3:
                y_list = [[float(p[2])] for p in pts]
                has_labels = True
        else:
            return JsonResponse({"error": "unsupported pts format"}, status=400)
        X_t = torch.tensor(X_list, dtype=torch.float32)
    except Exception as e:
        return JsonResponse({"error": f"pts parse failed: {e}"}, status=400)

    try:
        loaded = _load_mlp_payload(m.file_path)
        state_dict = loaded["state_dict"]
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
            torch.nn.Sigmoid(),
        )
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            pred_t = model(X_t).squeeze()
            if pred_t.dim() == 0:
                pred = [round(float(pred_t.item()), 4)]
                pred_label = [1 if pred[0] > 0.5 else 0]
            else:
                pred = [round(float(v), 4) for v in pred_t.tolist()]
                pred_label = [1 if v > 0.5 else 0 for v in pred_t.tolist()]
            acc = None
            if has_labels and y_list is not None:
                y_t = torch.tensor(y_list, dtype=torch.float32).squeeze()
                # align dims
                if y_t.dim() == 0:
                    y_t = y_t.unsqueeze(0)
                if pred_t.dim() == 0:
                    pred_t = pred_t.unsqueeze(0)
                acc = round(((pred_t > 0.5).float() == y_t).float().mean().item(), 4)
    except Exception as e:
        return JsonResponse({"error": f"inference failed: {e}"}, status=500)

    resp = {
        "model_id": m.id,
        "label_rule": m.label_rule,
        "center_x": m.center_x,
        "center_y": m.center_y,
        "rotation": m.rotation,
        "file_path": m.file_path,
        "meta": loaded.get("meta", {}),
        "pred": pred,
        "pred_label": pred_label,
        "w1": loaded["w1"],
        "b1": loaded["b1"],
        "w2": loaded["w2"],
        "b2": loaded["b2"],
    }
    if acc is not None:
        resp["acc"] = acc
    return JsonResponse(resp)
