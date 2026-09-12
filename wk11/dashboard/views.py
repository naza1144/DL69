import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.conf import settings

# Global cached data dictionary to avoid re-reading the 43MB CSV on each web request
_DATA_CACHE = {}


class TimeSeriesDataset(Dataset):
    def __init__(self, data, lookback):
        self.data = torch.FloatTensor(data)
        self.lookback = lookback

    def __len__(self):
        return len(self.data) - self.lookback

    def __getitem__(self, i):
        return (
            self.data[i : i + self.lookback],
            self.data[i + self.lookback]
        )


class TemperatureLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class TemperatureRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class TemperatureGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


MODEL_BUILDERS = {
    'lstm': lambda: TemperatureLSTM(input_size=1, hidden_size=32, num_layers=1),
    'rnn': lambda: TemperatureRNN(input_size=1, hidden_size=32, num_layers=1),
    'gru': lambda: TemperatureGRU(input_size=1, hidden_size=32, num_layers=1),
}


def get_cached_dataset(lookback=60):
    global _DATA_CACHE
    if 'scaler' not in _DATA_CACHE:
        base_dir = Path(settings.BASE_DIR)
        csv_path = base_dir / 'archive' / 'jena_climate_2009_2016.csv'
        if not csv_path.exists():
            csv_path = base_dir / 'jena_climate_2009_2016.csv'

        if not csv_path.exists():
            raise FileNotFoundError(f"Jena Climate CSV not found at {csv_path}")

        df = pd.read_csv(csv_path)
        temp = df['T (degC)'].values.astype(np.float32)

        scaler = MinMaxScaler()
        temp_scaled = scaler.fit_transform(temp.reshape(-1, 1)).flatten()

        train_size = 200000
        train_data = temp_scaled[:train_size]
        test_data = temp_scaled[train_size - lookback:]

        _DATA_CACHE['scaler'] = scaler
        _DATA_CACHE['raw_temp'] = temp
        _DATA_CACHE['temp_scaled'] = temp_scaled
        _DATA_CACHE['train_size'] = train_size
        _DATA_CACHE['train_data'] = train_data
        _DATA_CACHE['test_data'] = test_data
        _DATA_CACHE['lookback'] = lookback

    return _DATA_CACHE


def index(request):
    base_dir = Path(settings.BASE_DIR)
    model_path = base_dir / 'lstm_temp_model.pth'
    plot_path = base_dir / 'prediction_vs_actual.png'

    has_saved_model = model_path.exists()
    model_size_kb = round(model_path.stat().st_size / 1024, 1) if has_saved_model else 0
    model_modified = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(model_path.stat().st_mtime)) if has_saved_model else 'N/A'

    context = {
        'has_saved_model': has_saved_model,
        'model_size_kb': model_size_kb,
        'model_modified': model_modified,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'has_plot': plot_path.exists(),
    }
    return render(request, 'dashboard/index.html', context)


def model_info(request):
    base_dir = Path(settings.BASE_DIR)
    model_path = base_dir / 'lstm_temp_model.pth'
    plot_path = base_dir / 'prediction_vs_actual.png'

    has_saved_model = model_path.exists()
    return JsonResponse({
        'has_saved_model': has_saved_model,
        'model_size_kb': round(model_path.stat().st_size / 1024, 1) if has_saved_model else 0,
        'model_modified': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(model_path.stat().st_mtime)) if has_saved_model else None,
        'has_plot': plot_path.exists(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    })


def train_stream(request):
    model_type = request.GET.get('model', 'lstm').lower()
    if model_type not in MODEL_BUILDERS:
        model_type = 'lstm'

    try:
        epochs = int(request.GET.get('epochs', 20))
        epochs = max(1, min(epochs, 50))  # limit between 1 and 50
    except ValueError:
        epochs = 20

    try:
        lr = float(request.GET.get('lr', 0.001))
    except ValueError:
        lr = 0.001

    try:
        batch_size = int(request.GET.get('batch_size', 256))
        batch_size = max(64, min(batch_size, 1024))
    except ValueError:
        batch_size = 256

    lookback = 60

    def event_stream():
        yield f"data: {json.dumps({'type': 'log', 'message': f'Starting training: {model_type.upper()} | Epochs: {epochs} | LR: {lr} | Lookback: {lookback}'})}\n\n"
        
        # 1. Prepare data
        t0 = time.time()
        try:
            cache = get_cached_dataset(lookback)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        scaler = cache['scaler']
        train_data = cache['train_data']
        test_data = cache['test_data']

        train_ds = TimeSeriesDataset(train_data, lookback)
        test_ds = TimeSeriesDataset(test_data, lookback)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MODEL_BUILDERS[model_type]().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Pre-select 200 points from the start of test set for real-time visualization preview
        preview_samples = 150
        preview_X = []
        preview_y = []
        for i in range(preview_samples):
            x_seq, y_val = test_ds[i]
            preview_X.append(x_seq.unsqueeze(-1))
            preview_y.append(y_val)
        preview_X_tensor = torch.stack(preview_X).to(device) # (150, 60, 1)
        actual_preview_degC = scaler.inverse_transform(np.array(preview_y).reshape(-1, 1)).flatten().tolist()

        yield f"data: {json.dumps({
            'type': 'init',
            'model': model_type.upper(),
            'total_epochs': epochs,
            'train_samples': len(train_ds),
            'test_samples': len(test_ds),
            'device': str(device),
            'preview_actual': actual_preview_degC
        })}\n\n"

        # 2. Training Loop
        total_training_start = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            model.train()
            train_loss = 0.0
            
            for X_b, y_b in train_loader:
                X_b = X_b.unsqueeze(-1).to(device)
                y_b = y_b.unsqueeze(-1).to(device)

                optimizer.zero_grad()
                pred = model(X_b)
                loss = criterion(pred, y_b)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(y_b)

            train_loss /= len(train_ds)

            # Evaluate test loss
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                # For quick epoch-level test loss check, test on a subset or full test set
                # Full test evaluation:
                for X_b, y_b in test_loader:
                    X_b = X_b.unsqueeze(-1).to(device)
                    y_b = y_b.unsqueeze(-1).to(device)
                    pred = model(X_b)
                    loss = criterion(pred, y_b)
                    test_loss += loss.item() * len(y_b)

            test_loss /= len(test_ds)

            # Predict preview sample for live visual curve updates
            with torch.no_grad():
                pred_preview = model(preview_X_tensor).cpu().numpy()
                pred_preview_degC = scaler.inverse_transform(pred_preview).flatten().tolist()

            epoch_time = round(time.time() - epoch_start, 2)
            # Estimate preview MAE
            preview_mae = float(np.mean(np.abs(np.array(actual_preview_degC) - np.array(pred_preview_degC))))

            yield f"data: {json.dumps({
                'type': 'epoch',
                'epoch': epoch,
                'total_epochs': epochs,
                'train_loss': float(round(train_loss, 6)),
                'test_loss': float(round(test_loss, 6)),
                'preview_mae': float(round(preview_mae, 4)),
                'epoch_time': epoch_time,
                'pred_preview': pred_preview_degC,
                'elapsed': round(time.time() - total_training_start, 1)
            })}\n\n"

        # 3. Final Evaluation & Save Model
        yield f"data: {json.dumps({'type': 'log', 'message': 'Calculating final test metrics and saving model...'})}\n\n"
        
        all_preds = []
        all_actuals = []
        model.eval()
        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b = X_b.unsqueeze(-1).to(device)
                pred = model(X_b)
                all_preds.append(pred.cpu().numpy())
                all_actuals.append(y_b.numpy())

        pred_scaled = np.vstack(all_preds)
        actual_scaled = np.concatenate(all_actuals).reshape(-1, 1)

        pred_degC = scaler.inverse_transform(pred_scaled).flatten()
        actual_degC = scaler.inverse_transform(actual_scaled).flatten()

        final_mae = float(round(mean_absolute_error(actual_degC, pred_degC), 4))

        # Save model state_dict
        base_dir = Path(settings.BASE_DIR)
        model_save_path = base_dir / 'lstm_temp_model.pth'
        torch.save(model.state_dict(), model_save_path)

        # Send 500 representative points for the full interactive plot
        plot_steps = 500
        yield f"data: {json.dumps({
            'type': 'done',
            'model': model_type.upper(),
            'final_mae': final_mae,
            'model_saved': str(model_save_path.name),
            'model_size_kb': round(model_save_path.stat().st_size / 1024, 1),
            'total_time': round(time.time() - total_training_start, 1),
            'full_actual': actual_degC[:plot_steps].tolist(),
            'full_pred': pred_degC[:plot_steps].tolist()
        })}\n\n"

    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response
