"""
PyTorch Deep Learning Models for PARVAAH-X Future Prediction
- LSTM: Month-by-month cost escalation forecasting
- Transformer: Project completion date prediction
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import json
import os
import random

# ─────────────────────────────────────────────
# 1. LSTM MODEL — Cost Escalation Forecasting
# ─────────────────────────────────────────────
class CostForecastLSTM(nn.Module):
    """
    LSTM that takes a sequence of monthly project snapshots and
    predicts the cost for the next N months.
    Input shape : (batch, seq_len, input_features)
    Output shape: (batch, forecast_horizon)
    """
    def __init__(self, input_size=6, hidden_size=128, num_layers=2,
                 forecast_horizon=12, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, forecast_horizon)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.fc(lstm_out[:, -1, :])   # use last timestep
        return out


# ─────────────────────────────────────────────
# 2. TRANSFORMER MODEL — Delay Classification
# ─────────────────────────────────────────────
class DelayTransformer(nn.Module):
    """
    Transformer encoder that predicts delay months from project trajectory.
    Input shape : (batch, seq_len, input_features)
    Output shape: (batch, 1) — predicted delay in months
    """
    def __init__(self, input_size=6, d_model=64, nhead=4,
                 num_encoder_layers=3, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        enc = self.transformer(x)
        out = self.fc(enc[:, -1, :])  # last position
        return out.squeeze(-1)


# ─────────────────────────────────────────────
# 3. DATA GENERATION — Synthetic Time-Series
# ─────────────────────────────────────────────
def build_time_series_from_snapshot(df: pd.DataFrame, seq_len: int = 12) -> tuple:
    """
    Expand each project snapshot into a time-series of `seq_len` monthly steps.
    Features per step: [normalized_expenditure, progress, climate, geo, supply, time_ratio]
    Target cost: future cumulative cost (normalized)
    Target delay: delay months
    """
    sequences, cost_targets, delay_targets = [], [], []

    for _, row in df.iterrows():
        orig_cost = max(row['Original_Approved_Cost_Cr'], 1)
        rev_cost  = max(row['Revised_Cost_Cr'], orig_cost)
        planned   = max(row['Planned_Duration_Days'], 1)
        current   = max(row['Current_Duration_Days'], planned)
        delay_months = max(0, (current - planned) / 30)

        # Build 12-step historical trajectory (interpolated)
        seq = []
        for t in range(seq_len):
            frac = (t + 1) / seq_len
            exp_t   = (row['Cumulative_Expenditure_Cr'] / orig_cost) * frac + random.gauss(0, 0.02)
            prog_t  = (row['Physical_Progress_Pct'] / 100) * frac + random.gauss(0, 0.01)
            clim    = row['Climate_Issue_Severity'] + random.gauss(0, 0.05)
            geo     = row['Geopolitical_War_Impact'] + random.gauss(0, 0.05)
            supply  = row['Supply_Chain_Disruption'] + random.gauss(0, 0.05)
            t_ratio = frac
            step = [
                np.clip(exp_t, 0, 3),
                np.clip(prog_t, 0, 1),
                np.clip(clim, 0, 1),
                np.clip(geo, 0, 1),
                np.clip(supply, 0, 1),
                t_ratio
            ]
            seq.append(step)

        # Target: next 12 months cost trajectory (normalized)
        future_costs = []
        for f in range(1, 13):
            noise = random.gauss(0, 0.03)
            projected = min((row['Cumulative_Expenditure_Cr'] / orig_cost) +
                            (f / 12) * (rev_cost - row['Cumulative_Expenditure_Cr']) / orig_cost + noise, 2.5)
            future_costs.append(max(0, projected))

        sequences.append(seq)
        cost_targets.append(future_costs)
        delay_targets.append(delay_months / 60)  # normalize to 0-1 (max 60 months delay)

    X = torch.FloatTensor(sequences)              # (N, seq_len, 6)
    y_cost = torch.FloatTensor(cost_targets)      # (N, 12)
    y_delay = torch.FloatTensor(delay_targets)    # (N,)
    return X, y_cost, y_delay


# ─────────────────────────────────────────────
# 4. TRAINING
# ─────────────────────────────────────────────
def train_all(data_path='data/real_augmented_parvaah_x_data.csv',
              out_dir='models/', epochs=60, lr=1e-3):

    os.makedirs(out_dir, exist_ok=True)
    print("Loading data...")
    df = pd.read_csv(data_path)
    for col in ['Start_Date', 'Planned_Completion_Date', 'Current_Completion_Date']:
        df[col] = pd.to_datetime(df[col])
    df['Planned_Duration_Days'] = (df['Planned_Completion_Date'] - df['Start_Date']).dt.days
    df['Current_Duration_Days'] = (df['Current_Completion_Date'] - df['Start_Date']).dt.days

    print("Building time-series sequences...")
    torch.manual_seed(42)
    random.seed(42)
    X, y_cost, y_delay = build_time_series_from_snapshot(df, seq_len=12)

    n = len(X)
    split = int(0.85 * n)
    X_train, X_val = X[:split], X[split:]
    yc_train, yc_val = y_cost[:split], y_cost[split:]
    yd_train, yd_val = y_delay[:split], y_delay[split:]

    # ── Train LSTM (cost forecast) ──
    print("\n--- Training LSTM Cost Forecaster ---")
    lstm_model = CostForecastLSTM(input_size=6, hidden_size=128, num_layers=2,
                                  forecast_horizon=12, dropout=0.2)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_val = float('inf')
    for epoch in range(epochs):
        lstm_model.train()
        optimizer.zero_grad()
        pred = lstm_model(X_train)
        loss = criterion(pred, yc_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            lstm_model.eval()
            with torch.no_grad():
                val_loss = criterion(lstm_model(X_val), yc_val).item()
            print(f"  Epoch {epoch+1:3d}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(lstm_model.state_dict(), os.path.join(out_dir, 'lstm_cost_model.pt'))

    print(f"  LSTM saved. Best Val Loss: {best_val:.4f}")

    # ── Train Transformer (delay prediction) ──
    print("\n--- Training Transformer Delay Predictor ---")
    trans_model = DelayTransformer(input_size=6, d_model=64, nhead=4,
                                   num_encoder_layers=3, dim_feedforward=128)
    optimizer2 = torch.optim.Adam(trans_model.parameters(), lr=lr)
    criterion2 = nn.MSELoss()
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=20, gamma=0.5)

    best_val2 = float('inf')
    for epoch in range(epochs):
        trans_model.train()
        optimizer2.zero_grad()
        pred2 = trans_model(X_train)
        loss2 = criterion2(pred2, yd_train)
        loss2.backward()
        torch.nn.utils.clip_grad_norm_(trans_model.parameters(), 1.0)
        optimizer2.step()
        scheduler2.step()

        if (epoch + 1) % 10 == 0:
            trans_model.eval()
            with torch.no_grad():
                val_loss2 = criterion2(trans_model(X_val), yd_val).item()
            print(f"  Epoch {epoch+1:3d}/{epochs} | Train Loss: {loss2.item():.4f} | Val Loss: {val_loss2:.4f}")
            if val_loss2 < best_val2:
                best_val2 = val_loss2
                torch.save(trans_model.state_dict(), os.path.join(out_dir, 'transformer_delay_model.pt'))

    print(f"  Transformer saved. Best Val Loss: {best_val2:.4f}")
    print("\nAll deep learning models trained and saved successfully!")


if __name__ == "__main__":
    train_all()
