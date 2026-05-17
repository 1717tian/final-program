import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


CURRENT_FILE = Path(__file__).resolve()
MODEL_DIR = CURRENT_FILE.parent
BACKEND_DIR = MODEL_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "pm25_sites"

BASE_FEATURES = [
    "pm25",
    "pm10",
    "o3",
    "no2",
    "so2",
    "co",
]

TIME_FEATURES = [
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
]

LAG_FEATURES = [
    "pm25_lag1",
    "pm25_lag3",
    "pm25_lag7",
    "pm10_lag1",
    "co_lag1",
]

ROLL_FEATURES = [
    "pm25_roll3",
    "pm25_roll7",
    "pm10_roll3",
    "co_roll3",
]

DIFF_FEATURES = [
    "pm25_diff1",
    "pm10_diff1",
]

FEATURED_FEATURES = (
    BASE_FEATURES
    + TIME_FEATURES
    + LAG_FEATURES
    + ROLL_FEATURES
    + DIFF_FEATURES
)

TARGET = "pm25"


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def safe_name(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"[^\w\u4e00-\u9fa5.-]+", "_", text)
    text = text.strip("._")
    return text or "pm25_lstm"


class PM25LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 7,
        dropout: float = 0.2,
    ):
        super(PM25LSTM, self).__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def read_csv_auto_encoding(file_path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb2312"]

    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"CSV读取失败：{file_path}，错误：{last_error}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [str(col).strip().lower() for col in df.columns]

    rename_map = {
        "pm2.5": "pm25",
        "pm_25": "pm25",
        "pm2_5": "pm25",
        "日期": "date",
        "时间": "date",
    }

    return df.rename(columns=rename_map)


def clip_outliers(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()

    for col in cols:
        low = df[col].quantile(0.01)
        high = df[col].quantile(0.99)

        df[col] = df[col].clip(lower=low, upper=high)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["day_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    return df


def add_lag_rolling_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    df = df.copy()

    df["pm25_lag1"] = df["pm25"].shift(1)
    df["pm25_lag3"] = df["pm25"].shift(3)
    df["pm25_lag7"] = df["pm25"].shift(7)

    df["pm10_lag1"] = df["pm10"].shift(1)
    df["co_lag1"] = df["co"].shift(1)

    df["pm25_roll3"] = df["pm25"].rolling(window=3, min_periods=1).mean()
    df["pm25_roll7"] = df["pm25"].rolling(window=7, min_periods=1).mean()

    df["pm10_roll3"] = df["pm10"].rolling(window=3, min_periods=1).mean()
    df["co_roll3"] = df["co"].rolling(window=3, min_periods=1).mean()

    df["pm25_diff1"] = df["pm25"].diff(1)
    df["pm10_diff1"] = df["pm10"].diff(1)

    df[features] = df[features].ffill().bfill()
    df = df.dropna(subset=features)

    return df


def read_single_csv(
    file_path: Path,
    feature_mode: str,
    seq_length: int,
    pred_days: int,
) -> Optional[pd.DataFrame]:
    try:
        df = read_csv_auto_encoding(file_path)
    except Exception as exc:
        print(f"跳过文件，CSV读取失败: {file_path}，错误：{exc}", flush=True)
        return None

    df = normalize_columns(df)

    if "date" not in df.columns:
        print(f"跳过文件，缺少 date 列: {file_path}", flush=True)
        return None

    missing_cols = [col for col in BASE_FEATURES if col not in df.columns]

    if missing_cols:
        print(f"跳过文件，缺少字段 {missing_cols}: {file_path}", flush=True)
        return None

    df = df[["date"] + BASE_FEATURES].copy()

    df["date"] = (
        df["date"]
        .astype(str)
        .str.strip()
        .str.replace("/", "-", regex=False)
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in BASE_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date"])

    df = (
        df.sort_values("date", ascending=True)
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    df[BASE_FEATURES] = df[BASE_FEATURES].ffill().bfill()
    df = df.dropna(subset=BASE_FEATURES)

    df = clip_outliers(df, BASE_FEATURES)

    if feature_mode == "featured":
        features = FEATURED_FEATURES
        df = add_time_features(df)
        df = add_lag_rolling_features(df, features)
    else:
        features = BASE_FEATURES

    if len(df) < seq_length + pred_days:
        print(f"跳过文件，数据量不足: {file_path}，有效数据 {len(df)} 条", flush=True)
        return None

    return df


def parse_env_dataset_files() -> List[Path]:
    raw_datasets = os.getenv("PM25_DATASETS", "").strip()
    single_dataset = os.getenv("PM25_DATASET", "").strip()
    data_dir = Path(os.getenv("PM25_DATA_DIR", str(DEFAULT_DATA_DIR)))

    paths: List[Path] = []

    if raw_datasets:
        try:
            parsed = json.loads(raw_datasets)
            if isinstance(parsed, list):
                paths = [Path(item) for item in parsed if str(item).strip()]
        except Exception as exc:
            print(f"PM25_DATASETS 解析失败，将尝试 PM25_DATASET：{exc}", flush=True)

    if not paths and single_dataset:
        paths = [Path(single_dataset)]

    if not paths:
        paths = sorted(data_dir.glob("*.csv"))

    return paths


def load_all_station_data(feature_mode: str, seq_length: int, pred_days: int) -> List[pd.DataFrame]:
    csv_files = parse_env_dataset_files()

    if len(csv_files) == 0:
        raise FileNotFoundError("未找到 CSV 文件")

    station_dfs = []

    print(f"共选择 {len(csv_files)} 个训练数据文件", flush=True)

    for file_path in csv_files:
        file_path = Path(file_path)

        print(f"读取文件: {file_path.name}", flush=True)

        df = read_single_csv(file_path, feature_mode, seq_length, pred_days)

        if df is not None:
            station_dfs.append(df)
            print(f"有效数据: {len(df)} 条", flush=True)

    if len(station_dfs) == 0:
        raise ValueError("没有可用站点数据")

    return station_dfs


def fit_scaler(station_dfs: List[pd.DataFrame], features: List[str]) -> MinMaxScaler:
    all_data = pd.concat([df[features] for df in station_dfs], ignore_index=True)

    scaler = MinMaxScaler()
    scaler.fit(all_data)

    return scaler


def transform_station_data(df: pd.DataFrame, scaler: MinMaxScaler, features: List[str]) -> np.ndarray:
    return scaler.transform(df[features])


def create_sequences_for_station(
    df: pd.DataFrame,
    scaler: MinMaxScaler,
    features: List[str],
    seq_length: int,
    pred_days: int,
) -> Tuple[np.ndarray, np.ndarray]:
    data = transform_station_data(df, scaler, features)
    target_index = features.index(TARGET)

    X = []
    y = []

    for i in range(len(data) - seq_length - pred_days + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + pred_days, target_index])

    return np.array(X), np.array(y)


def split_station_sequences(X: np.ndarray, y: np.ndarray):
    total = len(X)

    train_end = int(total * 0.7)
    val_end = int(total * 0.85)

    return (
        X[:train_end],
        y[:train_end],
        X[train_end:val_end],
        y[train_end:val_end],
        X[val_end:],
        y[val_end:],
    )


def inverse_pm25_values(scaler: MinMaxScaler, values: np.ndarray, features: List[str]) -> np.ndarray:
    values = np.array(values)
    original_shape = values.shape
    values_flat = values.reshape(-1)

    dummy = np.zeros((len(values_flat), len(features)))

    target_index = features.index(TARGET)
    dummy[:, target_index] = values_flat

    inversed = scaler.inverse_transform(dummy)

    pm25 = inversed[:, target_index]

    return pm25.reshape(original_shape)


def calculate_metrics(pred_inv: np.ndarray, true_inv: np.ndarray) -> Dict[str, float]:
    true_flat = true_inv.reshape(-1)
    pred_flat = pred_inv.reshape(-1)

    rmse = np.sqrt(mean_squared_error(true_flat, pred_flat))
    mae = mean_absolute_error(true_flat, pred_flat)

    try:
        r2 = r2_score(true_flat, pred_flat)
    except Exception:
        r2 = 0.0

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def evaluate_model(
    model: nn.Module,
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    scaler: MinMaxScaler,
    features: List[str],
) -> Dict[str, float]:
    model.eval()

    with torch.no_grad():
        pred = model(X_tensor).cpu().numpy()
        true = y_tensor.cpu().numpy()

    pred_inv = inverse_pm25_values(scaler, pred, features)
    true_inv = inverse_pm25_values(scaler, true, features)

    return calculate_metrics(pred_inv, true_inv)


def evaluate_baseline(
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: MinMaxScaler,
    features: List[str],
    pred_days: int,
) -> Dict[str, float]:
    target_index = features.index(TARGET)

    last_pm25_scaled = X_test[:, -1, target_index]

    baseline_pred = np.repeat(last_pm25_scaled.reshape(-1, 1), pred_days, axis=1)

    pred_inv = inverse_pm25_values(scaler, baseline_pred, features)
    true_inv = inverse_pm25_values(scaler, y_test, features)

    return calculate_metrics(pred_inv, true_inv)


def evaluate_high_pollution_model(
    model: nn.Module,
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    scaler: MinMaxScaler,
    features: List[str],
    threshold: float = 115,
) -> Dict[str, float]:
    model.eval()

    with torch.no_grad():
        pred = model(X_tensor).cpu().numpy()
        true = y_tensor.cpu().numpy()

    pred_inv = inverse_pm25_values(scaler, pred, features)
    true_inv = inverse_pm25_values(scaler, true, features)

    mask = true_inv.reshape(-1) >= threshold

    if mask.sum() == 0:
        return {
            "high_rmse": 0.0,
            "high_mae": 0.0,
            "high_count": 0,
        }

    high_true = true_inv.reshape(-1)[mask]
    high_pred = pred_inv.reshape(-1)[mask]

    rmse = np.sqrt(mean_squared_error(high_true, high_pred))
    mae = mean_absolute_error(high_true, high_pred)

    return {
        "high_rmse": float(rmse),
        "high_mae": float(mae),
        "high_count": int(mask.sum()),
    }


def make_loss_fn(loss_mode: str, day_weights: torch.Tensor):
    """
    趋势感知损失函数。

    目标：
    1. 惩罚预测值本身误差
    2. 惩罚未来多日预测曲线的趋势误差
    3. 对高污染样本加权，提升模型对污染峰值的响应能力

    环境变量：
    PM25_TREND_LOSS_WEIGHT：趋势损失权重，默认 0.35
    PM25_HIGH_WEIGHT_FACTOR：高污染权重系数，默认 3.0
    """

    trend_weight = float(os.getenv("PM25_TREND_LOSS_WEIGHT", "0.35"))
    high_weight_factor = float(os.getenv("PM25_HIGH_WEIGHT_FACTOR", "3.0"))

    def build_day_weighted_loss(error, true):
        day_weight = day_weights.view(1, -1)

        pollution_weight = 1.0 + true * high_weight_factor

        return (error * day_weight * pollution_weight).mean()

    def trend_loss(pred, true):
        if pred.shape[1] <= 1:
            return torch.tensor(0.0, device=pred.device)

        pred_diff = pred[:, 1:] - pred[:, :-1]
        true_diff = true[:, 1:] - true[:, :-1]

        diff_error = torch.abs(pred_diff - true_diff)

        trend_day_weight = day_weights[1:].view(1, -1)

        return (diff_error * trend_day_weight).mean()

    def mae_day_weighted(pred, true):
        base_error = torch.abs(pred - true)
        base = build_day_weighted_loss(base_error, true)
        trend = trend_loss(pred, true)

        return base + trend_weight * trend

    def warning_mse(pred, true):
        base_error = (pred - true) ** 2
        base = build_day_weighted_loss(base_error, true)
        trend = trend_loss(pred, true)

        return base + trend_weight * trend

    if loss_mode == "warning_mse":
        return warning_mse

    return mae_day_weighted


def format_epoch_log(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    metrics: Dict[str, float],
    lr: float,
) -> str:
    return (
        f"Epoch {epoch}/{total_epochs} - "
        f"loss: {train_loss:.6f} - "
        f"val_loss: {val_loss:.6f} - "
        f"rmse: {metrics['rmse']:.6f} - "
        f"mae: {metrics['mae']:.6f} - "
        f"r2: {metrics['r2']:.6f} - "
        f"lr: {lr:.8f}"
    )


def save_named_copy(model_path: Path, scaler_path: Path, model_name: str):
    sanitized = safe_name(model_name)

    default_model_stem = model_path.stem
    default_scaler_stem = scaler_path.stem

    if sanitized in [default_model_stem, default_scaler_stem]:
        return

    named_model_path = model_path.with_name(f"{sanitized}.pt")
    named_scaler_path = scaler_path.with_name(f"scaler_{sanitized}.save")

    try:
        if model_path.exists():
            named_model_path.write_bytes(model_path.read_bytes())

        if scaler_path.exists():
            named_scaler_path.write_bytes(scaler_path.read_bytes())

        print(f"命名模型副本已保存到: {named_model_path}", flush=True)
        print(f"命名归一化器副本已保存到: {named_scaler_path}", flush=True)
    except Exception as exc:
        print(f"保存命名副本失败: {exc}", flush=True)


def train_model(config: Dict[str, object]):
    torch.manual_seed(42)
    np.random.seed(42)

    model_label = str(config["model_label"])
    feature_mode = str(config["feature_mode"])
    loss_mode = str(config["loss_mode"])
    default_model_file = str(config["model_file"])
    default_scaler_file = str(config["scaler_file"])
    high_pollution_eval = bool(config.get("high_pollution_eval", False))

    features = FEATURED_FEATURES if feature_mode == "featured" else BASE_FEATURES

    seq_length = env_int("PM25_SEQ_LENGTH", 45)
    pred_days = env_int("PM25_PRED_DAYS", 7)

    hidden_size = env_int("PM25_HIDDEN_SIZE", 64)
    num_layers = env_int("PM25_NUM_LAYERS", 2)
    dropout = env_float("PM25_DROPOUT", 0.2)

    num_epochs = env_int("PM25_EPOCHS", 300)
    batch_size = env_int("PM25_BATCH_SIZE", 32)
    learning_rate = env_float("PM25_LEARNING_RATE", 0.0005)
    patience = env_int("PM25_PATIENCE", 40)

    model_name = os.getenv("PM25_MODEL_NAME", Path(default_model_file).stem)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = Path(os.getenv("PM25_MODEL_DIR", str(MODEL_DIR)))
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / default_model_file
    scaler_path = model_dir / default_scaler_file

    base_day_weights = [1.5, 1.3, 1.2, 1.0, 0.9, 0.8, 0.7]

    if pred_days <= len(base_day_weights):
        weights = base_day_weights[:pred_days]
    else:
        weights = base_day_weights + [0.7] * (pred_days - len(base_day_weights))

    day_weights = torch.tensor(
        weights,
        dtype=torch.float32,
    ).to(device)

    criterion = make_loss_fn(loss_mode, day_weights)

    print("=" * 80, flush=True)
    print(f"开始训练：{model_label}", flush=True)
    print(f"模型名称: {model_name}", flush=True)
    print(f"设备: {device}", flush=True)
    print(f"特征模式: {feature_mode}", flush=True)
    print(f"损失函数: {loss_mode}", flush=True)
    print(f"趋势损失权重: {os.getenv('PM25_TREND_LOSS_WEIGHT', '0.35')}", flush=True)
    print(f"高污染权重系数: {os.getenv('PM25_HIGH_WEIGHT_FACTOR', '3.0')}", flush=True)
    print(f"seq_length: {seq_length}", flush=True)
    print(f"pred_days: {pred_days}", flush=True)
    print(f"epochs: {num_epochs}", flush=True)
    print(f"batch_size: {batch_size}", flush=True)
    print(f"learning_rate: {learning_rate}", flush=True)
    print(f"hidden_size: {hidden_size}", flush=True)
    print(f"num_layers: {num_layers}", flush=True)
    print(f"dropout: {dropout}", flush=True)
    print("=" * 80, flush=True)

    print("\n开始读取训练数据...\n", flush=True)

    station_dfs = load_all_station_data(feature_mode, seq_length, pred_days)

    scaler = fit_scaler(station_dfs, features)

    X_train_list = []
    y_train_list = []

    X_val_list = []
    y_val_list = []

    X_test_list = []
    y_test_list = []

    print("\n开始按站点构造时间序列样本...\n", flush=True)

    for df in station_dfs:
        X, y = create_sequences_for_station(df, scaler, features, seq_length, pred_days)

        X_train, y_train, X_val, y_val, X_test, y_test = split_station_sequences(X, y)

        if len(X_train) > 0:
            X_train_list.append(X_train)
            y_train_list.append(y_train)

        if len(X_val) > 0:
            X_val_list.append(X_val)
            y_val_list.append(y_val)

        if len(X_test) > 0:
            X_test_list.append(X_test)
            y_test_list.append(y_test)

    if not X_train_list or not X_val_list or not X_test_list:
        raise ValueError("训练集、验证集或测试集为空，请检查 CSV 数据量是否足够。")

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)

    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    print(f"训练集 X: {X_train.shape}", flush=True)
    print(f"训练集 y: {y_train.shape}", flush=True)
    print(f"验证集 X: {X_val.shape}", flush=True)
    print(f"验证集 y: {y_val.shape}", flush=True)
    print(f"测试集 X: {X_test.shape}", flush=True)
    print(f"测试集 y: {y_test.shape}", flush=True)

    baseline_metrics = evaluate_baseline(X_test, y_test, scaler, features, pred_days)

    print(
        "BASELINE_METRICS - "
        f"rmse: {baseline_metrics['rmse']:.6f} - "
        f"mae: {baseline_metrics['mae']:.6f} - "
        f"r2: {baseline_metrics['r2']:.6f}",
        flush=True,
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = PM25LSTM(
        input_size=len(features),
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=pred_days,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
    )

    best_val_rmse = float("inf")
    no_improve_count = 0

    print(f"\n开始训练 {model_label}...\n", flush=True)

    for epoch in range(1, num_epochs + 1):
        model.train()

        total_loss = 0.0
        batch_count = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            outputs = model(batch_X)

            loss = criterion(outputs, batch_y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += float(loss.item())
            batch_count += 1

        avg_train_loss = total_loss / max(batch_count, 1)

        model.eval()

        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = float(criterion(val_outputs, y_val_tensor).item())

        val_metrics = evaluate_model(
            model,
            X_val_tensor,
            y_val_tensor,
            scaler,
            features,
        )

        scheduler.step(val_metrics["rmse"])

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            format_epoch_log(
                epoch=epoch,
                total_epochs=num_epochs,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                metrics=val_metrics,
                lr=current_lr,
            ),
            flush=True,
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            no_improve_count = 0

            torch.save(model.state_dict(), model_path)
            joblib.dump(scaler, scaler_path)

            print(
                f"保存最佳模型 - epoch: {epoch} - best_val_rmse: {best_val_rmse:.6f}",
                flush=True,
            )

        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"EarlyStopping - patience: {patience} - epoch: {epoch}", flush=True)
            break

    print("\n加载最佳模型进行测试集评估...", flush=True)

    model.load_state_dict(torch.load(model_path, map_location=device))

    test_metrics = evaluate_model(
        model,
        X_test_tensor,
        y_test_tensor,
        scaler,
        features,
    )

    print(
        "FINAL_METRICS - "
        f"test_rmse: {test_metrics['rmse']:.6f} - "
        f"test_mae: {test_metrics['mae']:.6f} - "
        f"test_r2: {test_metrics['r2']:.6f} - "
        f"baseline_rmse: {baseline_metrics['rmse']:.6f} - "
        f"baseline_mae: {baseline_metrics['mae']:.6f} - "
        f"best_val_rmse: {best_val_rmse:.6f}",
        flush=True,
    )

    if high_pollution_eval:
        high_metrics = evaluate_high_pollution_model(
            model,
            X_test_tensor,
            y_test_tensor,
            scaler,
            features,
            threshold=115,
        )

        print(
            "HIGH_POLLUTION_METRICS - "
            f"high_count: {high_metrics['high_count']} - "
            f"high_rmse: {high_metrics['high_rmse']:.6f} - "
            f"high_mae: {high_metrics['high_mae']:.6f}",
            flush=True,
        )

    print(f"模型已保存到: {model_path}", flush=True)
    print(f"归一化器已保存到: {scaler_path}", flush=True)

    save_named_copy(model_path, scaler_path, model_name)

    if test_metrics["rmse"] < baseline_metrics["rmse"]:
        print("模型优于持久性预测 baseline", flush=True)
    else:
        print("模型未明显优于 baseline，建议继续调整特征或模型参数", flush=True)