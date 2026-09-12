import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# ==========================================
# 1. Dataset & DataLoader (Slide 12)
# ==========================================
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


# ==========================================
# 2. LSTM Model Definition (Slide 13 & Homework)
# Requirements: LSTM(1, 32, 1) + Linear
# ==========================================
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
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.lstm(x)
        # Take the output from the last time step
        pred = self.fc(out[:, -1, :])
        return pred


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train LSTM Model for Jena Climate Temperature Prediction")
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train (default: 20)')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for Adam (default: 0.001)')
    parser.add_argument('--lookback', type=int, default=60, help='Lookback window size (default: 60)')
    parser.add_argument('--save-model', type=str, default='lstm_temp_model.pth', help='Path to save state_dict')
    parser.add_argument('--plot-name', type=str, default='prediction_vs_actual.png', help='Path to save plot')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Jena Climate dataset
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / 'archive' / 'jena_climate_2009_2016.csv'
    if not csv_path.exists():
        csv_path = base_dir / 'jena_climate_2009_2016.csv'

    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    temp = df['T (degC)'].values.astype(np.float32)
    print(f"Total observations: {len(temp):,}")

    # 2. Normalization (MinMaxScaler)
    scaler = MinMaxScaler()
    temp_scaled = scaler.fit_transform(temp.reshape(-1, 1)).flatten()

    # 3. Train/Test split (Chronological!)
    LOOKBACK = args.lookback  # 60 steps = 10 hours (10 min per step)
    train_size = 200000
    train_data = temp_scaled[:train_size]
    test_data = temp_scaled[train_size - LOOKBACK:]

    train_ds = TimeSeriesDataset(train_data, LOOKBACK)
    test_ds = TimeSeriesDataset(test_data, LOOKBACK)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Train samples: {len(train_ds):,}, Test samples: {len(test_ds):,}")

    # 4. Model, Loss, Optimizer
    # Requirements: LSTM(1, 32, 1) + Linear, Adam lr=0.001
    model = TemperatureLSTM(input_size=1, hidden_size=32, num_layers=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 5. Training Loop: 20 epochs
    EPOCHS = args.epochs
    print(f"\nStarting training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b = X_b.unsqueeze(-1).to(device)  # shape: (B, 60, 1)
            y_b = y_b.unsqueeze(-1).to(device)  # shape: (B, 1)

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
            for X_b, y_b in test_loader:
                X_b = X_b.unsqueeze(-1).to(device)
                y_b = y_b.unsqueeze(-1).to(device)
                pred = model(X_b)
                loss = criterion(pred, y_b)
                test_loss += loss.item() * len(y_b)

        test_loss /= len(test_ds)

        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] - Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

    # 6. Evaluation & Inverse Transform
    print("\nEvaluating model on test set...")
    model.eval()
    all_preds = []
    all_actuals = []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.unsqueeze(-1).to(device)
            pred = model(X_b)
            all_preds.append(pred.cpu().numpy())
            all_actuals.append(y_b.numpy())

    pred_scaled = np.vstack(all_preds)
    actual_scaled = np.concatenate(all_actuals).reshape(-1, 1)

    # Inverse transform to original degree Celsius (°C)
    pred_degC = scaler.inverse_transform(pred_scaled).flatten()
    actual_degC = scaler.inverse_transform(actual_scaled).flatten()

    mae = mean_absolute_error(actual_degC, pred_degC)
    print("=" * 45)
    print(f"Mean Absolute Error (MAE): {mae:.4f} °C")
    print("=" * 45)

    # 7. Plot Prediction vs Actual
    plot_path = base_dir / args.plot_name
    plt.figure(figsize=(14, 5))
    display_steps = 1440  # 10 days of data (1440 * 10 min)
    plt.plot(actual_degC[:display_steps], label='Actual Temperature', color='#2563eb', alpha=0.8, linewidth=1.2)
    plt.plot(pred_degC[:display_steps], label='LSTM Prediction', color='#dc2626', linestyle='--', alpha=0.85, linewidth=1.2)
    plt.title(f'Jena Climate: LSTM Temperature Prediction vs Actual (First {display_steps} steps / 10 days)\nTest MAE = {mae:.2f} °C')
    plt.xlabel('Time Step (10-minute intervals)')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {plot_path}")

    # 8. Save model with state_dict()
    model_save_path = base_dir / args.save_model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved with state_dict() to: {model_save_path}")


if __name__ == '__main__':
    main()
