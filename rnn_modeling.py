import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from config import Config

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Data Preparation
# -----------------------------
def data_preprocessing_for_rnn(path: str) -> pd.DataFrame:
    print("Starting data preparation with pandas...")

    # Load data
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Parquet file not found at {path}. Please provide the dataset or adjust the path."
        )

    start_month = Config.Crop.GROWING_SEASON_START_MONTH
    end_month = Config.Crop.GROWING_SEASON_END_MONTH

    # Filter by growing season months
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[
        (df["Date"].dt.month >= start_month) & (df["Date"].dt.month <= end_month)
    ].copy()

    # Week/year features
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Year"] = df["Date"].dt.year.astype(int)
    df["Week"] = df["Week"] - df.groupby("Year")["Week"].transform("min") + 1

    numeric_cols = Config.Model. features + [Config.Model.target_col]

    # Daily to weekly aggregation
    df = df.groupby(["Date", "State"], as_index=False).mean(numeric_only=True)
    df = df.groupby(["Year", "State", "Week"], as_index=False)[numeric_cols].mean()

    # Ensure 31 weeks per season (trim/align)
    df["total_weeks_in_year"] = df.groupby("Year")["Week"].transform("nunique")
    df["max_week_per_year"] = df.groupby("Year")["Week"].transform("max")
    df = df[
        (df["total_weeks_in_year"] == 31)
        | (df["Week"] > (df["max_week_per_year"] - 31))
    ].copy()
    # Fix off-by-one years starting at Week=2
    min_week_by_year = df.groupby("Year")["Week"].transform("min")
    df.loc[min_week_by_year == 2, "Week"] = df.loc[min_week_by_year == 2, "Week"] - 1
    df.drop(columns=["total_weeks_in_year", "max_week_per_year"], inplace=True)

    # Re-number weeks 1..31 within (Year, State)
    df["Week"] = df.groupby(["Year", "State"])["Week"].transform(
        lambda x: pd.Series(range(1, len(x) + 1), index=x.index)
    )

    # Build sequences
    feature_cols = Config.Model.features
    target_col = Config.Model.target_col

    df = df.sort_values(by=["Year", "State", "Week"])
    grp = df.groupby(["Year", "State"])

    features_sequences = grp[feature_cols].apply(
        lambda x: [list(row) for row in x.values]
    )
    targets = grp[target_col].apply(lambda x: x.iloc[-1])

    rnn_df = pd.DataFrame(
        {
            "Year": features_sequences.index.get_level_values("Year").values,
            "State": features_sequences.index.get_level_values("State").values,
            "features": features_sequences.values,
            "target": targets.values,
        }
    )

    # Sanity prints
    if rnn_df.empty:
        raise ValueError(
            "rnn_df is empty after data preparation. Cannot proceed with model building."
        )
    max_sequence_length = len(rnn_df["features"].iloc[0])
    num_features = len(rnn_df["features"].iloc[0][0])
    print("\nData preparation complete. RNN DataFrame head:")
    print(rnn_df.head())
    print(f"\nTimesteps per sequence: {max_sequence_length}")
    print(f"Features per timestep: {num_features}")

    return rnn_df


# -----------------------------
# Model
# -----------------------------
class LSTMRegressor(nn.Module):
    def __init__(
        self, input_size, hidden_size=256, output_size=1, num_layers=2, dropout_rate=0.3
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, T, F]
        out, _ = self.lstm(x)  # [B, T, H]
        out = out[:, -1, :]  # last timestep [B, H]
        out = self.norm(out)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc_out(out)  # [B, 1]
        return out


# -----------------------------
# Data preparation for loaders
# -----------------------------
def prepare_dataloaders(rnn_df, batch_size=32, test_year_threshold=2022):
    if rnn_df.empty:
        raise ValueError("Input DataFrame is empty.")

    # 1. Sort the data once for all subsequent operations.
    rnn_df.sort_values(by=["Year", "State"], inplace=True)

    X_np = np.array(
        [np.array(seq, dtype=np.float32) for seq in rnn_df["features"].values]
    )
    y_np = np.array(rnn_df["target"].values, dtype=np.float32)
    years_np = rnn_df["Year"].values

    # Temporal split BEFORE scaling to avoid leakage
    train_val_mask = years_np < test_year_threshold
    test_mask = years_np >= test_year_threshold

    X_train_val = X_np[train_val_mask]
    y_train_val = y_np[train_val_mask]
    X_test = X_np[test_mask]
    y_test = y_np[test_mask]

    # Fit scalers on TRAIN ONLY
    n_features = X_np.shape[2]
    X_scaler = StandardScaler()
    X_train_val_flat = X_train_val.reshape(-1, n_features)
    X_scaler.fit(X_train_val_flat)

    X_train_val_scaled = X_scaler.transform(X_train_val_flat).reshape(X_train_val.shape)
    X_test_scaled = X_scaler.transform(X_test.reshape(-1, n_features)).reshape(
        X_test.shape
    )

    y_scaler = StandardScaler()
    y_train_val_scaled = y_scaler.fit_transform(y_train_val.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    # Train/Val split from pre-2022
    train_idx, val_idx = train_test_split(
        np.arange(len(X_train_val_scaled)),
        test_size=0.2,
        random_state=SEED,
        shuffle=True,
    )

    def to_tensor_dataset(X, y):
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        return TensorDataset(X_t, y_t)

    train_ds = Subset(
        to_tensor_dataset(X_train_val_scaled, y_train_val_scaled), train_idx.tolist()
    )
    val_ds = Subset(
        to_tensor_dataset(X_train_val_scaled, y_train_val_scaled), val_idx.tolist()
    )
    test_ds = to_tensor_dataset(X_test_scaled, y_test_scaled)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    meta = {
        "X_scaler": X_scaler,
        "y_scaler": y_scaler,
        "train_counts": len(train_ds),
        "val_counts": len(val_ds),
        "test_counts": len(test_ds),
        "n_features": n_features,
        "timesteps": X_np.shape[1],
    }

    print(
        "\nData successfully prepared with temporal split (train/val before 2022, test 2022+)."
    )
    print(
        f"Train: {meta['train_counts']}, Val: {meta['val_counts']}, Test: {meta['test_counts']}"
    )
    return train_loader, val_loader, test_loader, meta


# -----------------------------
# Optional sanity checks
# -----------------------------
def check_feature_target_correlation(rnn_df):
    # Average each sequence over time to approximate per-sequence feature summary
    feature_cols = Config.Model.features
    tmp = pd.DataFrame(
        np.vstack(rnn_df["features"].apply(lambda x: np.mean(x, axis=0))).astype(float),
        columns=feature_cols,
    )
    tmp["target"] = rnn_df["target"].values
    corr = tmp.corr(numeric_only=True)["target"].sort_values(ascending=False)
    print("\nFeature ↔ Target correlations (sequence means):")
    print(corr)


def plot_train_test_distribution_shift(
    X_train_val, X_test, title="Feature Distribution (Train vs Test)"
):
    # Simple flattened histogram comparison
    plt.figure(figsize=(8, 5))
    plt.hist(X_train_val.reshape(-1), bins=60, alpha=0.5, label="train (pre-2022)")
    plt.hist(X_test.reshape(-1), bins=60, alpha=0.5, label="test (2022+)")
    plt.legend()
    plt.title(title)
    plt.xlabel("Feature Values (raw)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# -----------------------------
# Training Loop with Early Stopping & Scheduler
# -----------------------------
def train_model(
    model,
    train_loader,
    val_loader,
    epochs=150,
    lr=1e-3,
    weight_decay=1e-4,
    patience=12,
    max_grad_norm=1.0,
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    print("\nTraining the model...")
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        running = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            running += loss.item() * Xb.size(0)
        train_loss = running / len(train_loader.dataset)

        # Validate
        model.eval()
        v_running = 0.0
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                outv = model(Xv)
                vloss = criterion(outv, yv)
                v_running += vloss.item() * Xv.size(0)
        val_loss = v_running / len(val_loader.dataset)
        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch}/{epochs}] | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Early stopping
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}. Best Val: {best_val:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, data_loader, y_scaler):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for Xb, yb in data_loader:
            Xb = Xb.to(device)
            out = model(Xb).cpu().numpy().flatten()
            preds.append(out)
            targets.append(yb.cpu().numpy().flatten())
    preds = np.concatenate(preds)
    targets_scaled = np.concatenate(targets)

    # Inverse transform target scaling
    preds_actual = y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    targets_actual = y_scaler.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()

    mse = mean_squared_error(targets_actual, preds_actual)
    mae = mean_absolute_error(targets_actual, preds_actual)
    r2 = r2_score(targets_actual, preds_actual)
    return preds_actual, targets_actual, mse, mae, r2


def plot_actual_vs_pred(y_true, y_pred, title="Actual vs Predicted"):
    plt.figure(figsize=(9, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Tiny Overfit Debug
# -----------------------------
def tiny_overfit_debug(model, train_loader, steps=200, lr=5e-3):
    """Try to overfit on a tiny subset to ensure the model can learn."""
    print("\n[Debug] Tiny overfit test on a few batches...")
    model_debug = type(model)(
        input_size=model.lstm.input_size,
        hidden_size=model.hidden_size,
        output_size=1,
        num_layers=model.num_layers,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_debug.parameters(), lr=lr)
    # Grab 1-2 small batches
    small_X, small_y = next(iter(train_loader))
    small_ds = TensorDataset(small_X, small_y)
    small_loader = DataLoader(small_ds, batch_size=8, shuffle=True)
    for step in range(steps):
        for Xb, yb in small_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model_debug(Xb)
            loss = criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model_debug.parameters(), 1.0)
            optimizer.step()
        if (step + 1) % 50 == 0:
            print(f" Step {step+1}/{steps} | Loss: {loss.item():.4f}")
    print("[Debug] Tiny overfit test completed.")


# -----------------------------
# Main
# -----------------------------
def LSTM_model(
        data_path: str):

    rnn_df = data_preprocessing_for_rnn(data_path)

    # Optional sanity check: correlations
    check_feature_target_correlation(rnn_df)

    # Prepare loaders (with proper scaling)
    train_loader, val_loader, test_loader, meta = prepare_dataloaders(
        rnn_df, batch_size=32, test_year_threshold=2022
    )

    # (Optional) Tiny overfit debug — set to True if you want to sanity check learning
    DO_TINY_OVERFIT_DEBUG = False
    if DO_TINY_OVERFIT_DEBUG:
        tmp_model = LSTMRegressor(input_size=meta["n_features"]).to(device)
        tiny_overfit_debug(tmp_model, train_loader)

    # Instantiate model
    model = LSTMRegressor(
        input_size=meta["n_features"], hidden_size=100, num_layers=3, dropout_rate=0.2
    ).to(device)
    print("\nModel Summary:")
    print(model)

    # Train
    model = train_model(
        model,
        train_loader,
        val_loader,
        epochs=200,
        lr=1e-3,
        weight_decay=1e-4,
        patience=10,
        max_grad_norm=1.0,
    )

    # Evaluate on train/val/test with inverse transform
    print("\nEvaluating on TRAIN set...")
    y_pred_tr, y_true_tr, mse_tr, mae_tr, r2_tr = evaluate(
        model, train_loader, meta["y_scaler"]
    )
    print(f"Train MSE: {mse_tr:.2f} | MAE: {mae_tr:.2f} | R²: {r2_tr:.3f}")

    print("\nEvaluating on VAL set...")
    y_pred_va, y_true_va, mse_va, mae_va, r2_va = evaluate(
        model, val_loader, meta["y_scaler"]
    )
    print(f"Val   MSE: {mse_va:.2f} | MAE: {mae_va:.2f} | R²: {r2_va:.3f}")

    print("\nEvaluating on TEST set...")
    y_pred_te, y_true_te, mse_te, mae_te, r2_te = evaluate(
        model, test_loader, meta["y_scaler"]
    )
    print(f"Test  MSE: {mse_te:.2f} | MAE: {mae_te:.2f} | R²: {r2_te:.3f}")

    # Plots
    plot_actual_vs_pred(y_true_te, y_pred_te, title="Actual vs Predicted (TEST)")

    # (Optional) If you want to quickly compare distributions pre- and post-2022 (raw, before scaling):
    # You can rebuild raw arrays here if desired — omitted to keep runtime clean.

    # Print arrays (if needed)
    print("\nSample Actual (TEST):", np.round(y_true_te[:20], 2))
    print("Sample Pred   (TEST):", np.round(y_pred_te[:20], 2))
