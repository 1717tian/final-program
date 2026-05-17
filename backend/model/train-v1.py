import os
import glob
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


DATA_DIR = r"D:\final program\pythonProject12\data\raw\pm25_sites"
MODEL_DIR = r"D:\final program\pythonProject12\backend\model"

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lstm_pm25_changchun_featured_direct7.pt"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler_pm25_featured_direct7.save"
)

BASE_FEATURES = [
    "pm25",
    "pm10",
    "o3",
    "no2",
    "so2",
    "co"
]

TIME_FEATURES = [
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos"
]

LAG_FEATURES = [
    "pm25_lag1",
    "pm25_lag3",
    "pm25_lag7",
    "pm10_lag1",
    "co_lag1"
]

ROLL_FEATURES = [
    "pm25_roll3",
    "pm25_roll7",
    "pm10_roll3",
    "co_roll3"
]

DIFF_FEATURES = [
    "pm25_diff1",
    "pm10_diff1"
]

FEATURES = (
    BASE_FEATURES
    + TIME_FEATURES
    + LAG_FEATURES
    + ROLL_FEATURES
    + DIFF_FEATURES
)

TARGET = "pm25"

SEQ_LENGTH = 45
PRED_DAYS = 7

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

NUM_EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
PATIENCE = 40

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DAY_WEIGHTS = torch.tensor(
    [1.5, 1.3, 1.2, 1.0, 0.9, 0.8, 0.7],
    dtype=torch.float32
).to(DEVICE)


class PM25LSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        output_size=7
    ):

        super(PM25LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=DROPOUT
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_size)
        )

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.fc(out)

        return out


def weighted_loss(pred, true):

    loss = torch.abs(pred - true)

    loss = loss * DAY_WEIGHTS

    return loss.mean()


def add_time_features(df):

    df = df.copy()

    df["month"] = df["date"].dt.month

    df["dayofyear"] = df["date"].dt.dayofyear

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    df["day_sin"] = np.sin(
        2 * np.pi * df["dayofyear"] / 365
    )

    df["day_cos"] = np.cos(
        2 * np.pi * df["dayofyear"] / 365
    )

    return df


def add_lag_rolling_features(df):

    df = df.copy()

    df["pm25_lag1"] = df["pm25"].shift(1)
    df["pm25_lag3"] = df["pm25"].shift(3)
    df["pm25_lag7"] = df["pm25"].shift(7)

    df["pm10_lag1"] = df["pm10"].shift(1)
    df["co_lag1"] = df["co"].shift(1)

    df["pm25_roll3"] = df["pm25"].rolling(
        window=3,
        min_periods=1
    ).mean()

    df["pm25_roll7"] = df["pm25"].rolling(
        window=7,
        min_periods=1
    ).mean()

    df["pm10_roll3"] = df["pm10"].rolling(
        window=3,
        min_periods=1
    ).mean()

    df["co_roll3"] = df["co"].rolling(
        window=3,
        min_periods=1
    ).mean()

    df["pm25_diff1"] = df["pm25"].diff(1)
    df["pm10_diff1"] = df["pm10"].diff(1)

    df[FEATURES] = (
        df[FEATURES]
        .ffill()
        .bfill()
    )

    df = df.dropna(
        subset=FEATURES
    )

    return df


def clip_outliers(df):

    df = df.copy()

    for col in BASE_FEATURES:

        low = df[col].quantile(0.01)

        high = df[col].quantile(0.99)

        df[col] = df[col].clip(
            lower=low,
            upper=high
        )

    return df


def read_single_csv(file_path):

    df = pd.read_csv(file_path)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    if "date" not in df.columns:

        print(f"跳过文件，缺少 date 列: {file_path}")

        return None

    missing_cols = [
        col for col in BASE_FEATURES
        if col not in df.columns
    ]

    if missing_cols:

        print(f"跳过文件，缺少字段 {missing_cols}: {file_path}")

        return None

    df = df[
        ["date"] + BASE_FEATURES
    ].copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    for col in BASE_FEATURES:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=["date"]
    )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    df[BASE_FEATURES] = (
        df[BASE_FEATURES]
        .ffill()
        .bfill()
    )

    df = df.dropna(
        subset=BASE_FEATURES
    )

    df = clip_outliers(df)

    df = add_time_features(df)

    df = add_lag_rolling_features(df)

    if len(df) < SEQ_LENGTH + PRED_DAYS:

        print(f"跳过文件，数据量不足: {file_path}")

        return None

    return df


def load_all_station_data():

    csv_files = glob.glob(
        os.path.join(DATA_DIR, "*.csv")
    )

    if len(csv_files) == 0:

        raise FileNotFoundError("未找到 CSV 文件")

    station_dfs = []

    print(f"共发现 {len(csv_files)} 个站点文件")

    for file_path in csv_files:

        print(f"读取文件: {os.path.basename(file_path)}")

        df = read_single_csv(file_path)

        if df is not None:

            station_dfs.append(df)

            print(f"有效数据: {len(df)} 条")

    if len(station_dfs) == 0:

        raise ValueError("没有可用站点数据")

    return station_dfs


def fit_feature_scaler(station_dfs):

    all_feature_data = pd.concat(
        [
            df[FEATURES]
            for df in station_dfs
        ],
        ignore_index=True
    )

    scaler = MinMaxScaler()

    scaler.fit(all_feature_data)

    return scaler


def transform_station_data(
    df,
    scaler
):

    scaled_data = scaler.transform(
        df[FEATURES]
    )

    return scaled_data


def create_sequences_for_station(
    df,
    scaler
):

    data = transform_station_data(
        df,
        scaler
    )

    target_index = FEATURES.index(TARGET)

    X = []
    y = []

    for i in range(
        len(data) - SEQ_LENGTH - PRED_DAYS + 1
    ):

        X.append(
            data[i:i + SEQ_LENGTH]
        )

        y.append(
            data[
                i + SEQ_LENGTH:
                i + SEQ_LENGTH + PRED_DAYS,
                target_index
            ]
        )

    return np.array(X), np.array(y)


def split_station_sequences(
    X,
    y
):

    total = len(X)

    train_end = int(total * 0.7)

    val_end = int(total * 0.85)

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


def inverse_pm25_values(
    scaler,
    values
):

    values = np.array(values)

    original_shape = values.shape

    values_flat = values.reshape(-1)

    dummy = np.zeros(
        (
            len(values_flat),
            len(FEATURES)
        )
    )

    target_index = FEATURES.index(TARGET)

    dummy[:, target_index] = values_flat

    inversed = scaler.inverse_transform(dummy)

    pm25 = inversed[:, target_index]

    return pm25.reshape(original_shape)


def evaluate_model(
    model,
    X_tensor,
    y_tensor,
    scaler
):

    model.eval()

    with torch.no_grad():

        pred = model(
            X_tensor
        ).cpu().numpy()

        true = y_tensor.cpu().numpy()

    pred_inv = inverse_pm25_values(
        scaler,
        pred
    )

    true_inv = inverse_pm25_values(
        scaler,
        true
    )

    rmse = np.sqrt(
        mean_squared_error(
            true_inv.reshape(-1),
            pred_inv.reshape(-1)
        )
    )

    mae = mean_absolute_error(
        true_inv.reshape(-1),
        pred_inv.reshape(-1)
    )

    return rmse, mae


def evaluate_baseline(
    X_test,
    y_test,
    scaler
):

    target_index = FEATURES.index(TARGET)

    last_pm25_scaled = X_test[
        :,
        -1,
        target_index
    ]

    baseline_pred = np.repeat(
        last_pm25_scaled.reshape(-1, 1),
        PRED_DAYS,
        axis=1
    )

    pred_inv = inverse_pm25_values(
        scaler,
        baseline_pred
    )

    true_inv = inverse_pm25_values(
        scaler,
        y_test
    )

    rmse = np.sqrt(
        mean_squared_error(
            true_inv.reshape(-1),
            pred_inv.reshape(-1)
        )
    )

    mae = mean_absolute_error(
        true_inv.reshape(-1),
        pred_inv.reshape(-1)
    )

    return rmse, mae


def train():

    print("\n开始读取长春站点数据...\n")

    station_dfs = load_all_station_data()

    scaler = fit_feature_scaler(
        station_dfs
    )

    X_train_list = []
    y_train_list = []

    X_val_list = []
    y_val_list = []

    X_test_list = []
    y_test_list = []

    print("\n开始按站点构造时间序列样本...\n")

    for df in station_dfs:

        X, y = create_sequences_for_station(
            df,
            scaler
        )

        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test
        ) = split_station_sequences(
            X,
            y
        )

        X_train_list.append(X_train)
        y_train_list.append(y_train)

        X_val_list.append(X_val)
        y_val_list.append(y_val)

        X_test_list.append(X_test)
        y_test_list.append(y_test)

    X_train = np.concatenate(
        X_train_list,
        axis=0
    )

    y_train = np.concatenate(
        y_train_list,
        axis=0
    )

    X_val = np.concatenate(
        X_val_list,
        axis=0
    )

    y_val = np.concatenate(
        y_val_list,
        axis=0
    )

    X_test = np.concatenate(
        X_test_list,
        axis=0
    )

    y_test = np.concatenate(
        y_test_list,
        axis=0
    )

    print("训练集 X:", X_train.shape)
    print("训练集 y:", y_train.shape)
    print("验证集 X:", X_val.shape)
    print("验证集 y:", y_val.shape)
    print("测试集 X:", X_test.shape)
    print("测试集 y:", y_test.shape)

    baseline_rmse, baseline_mae = evaluate_baseline(
        X_test,
        y_test,
        scaler
    )

    print("\n持久性预测 baseline:")
    print(f"Baseline RMSE: {baseline_rmse:.2f}")
    print(f"Baseline MAE : {baseline_mae:.2f}")

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32
    ).to(DEVICE)

    y_val_tensor = torch.tensor(
        y_val,
        dtype=torch.float32
    ).to(DEVICE)

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    ).to(DEVICE)

    y_test_tensor = torch.tensor(
        y_test,
        dtype=torch.float32
    ).to(DEVICE)

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = PM25LSTM(
        input_size=len(FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=PRED_DAYS
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8
    )

    best_val_rmse = float("inf")

    no_improve_count = 0

    print("\n开始训练模型...\n")

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        model.train()

        total_loss = 0

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(batch_X)

            loss = weighted_loss(
                outputs,
                batch_y
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()

        val_rmse, val_mae = evaluate_model(
            model,
            X_val_tensor,
            y_val_tensor,
            scaler
        )

        scheduler.step(val_rmse)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] "
            f"Loss={total_loss:.4f} "
            f"Val_RMSE={val_rmse:.2f} "
            f"Val_MAE={val_mae:.2f} "
            f"LR={current_lr:.6f}"
        )

        if val_rmse < best_val_rmse:

            best_val_rmse = val_rmse

            no_improve_count = 0

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            joblib.dump(
                scaler,
                SCALER_PATH
            )

            print("已保存最佳模型")

        else:

            no_improve_count += 1

        if no_improve_count >= PATIENCE:

            print("\n验证集长期未提升，提前停止训练")

            break

    print("\n加载最佳模型进行测试集评估...")

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    test_rmse, test_mae = evaluate_model(
        model,
        X_test_tensor,
        y_test_tensor,
        scaler
    )

    print("\n训练完成")
    print(f"最佳验证集 RMSE: {best_val_rmse:.2f}")
    print(f"测试集 RMSE: {test_rmse:.2f}")
    print(f"测试集 MAE : {test_mae:.2f}")
    print(f"Baseline RMSE: {baseline_rmse:.2f}")
    print(f"Baseline MAE : {baseline_mae:.2f}")
    print(f"模型已保存到: {MODEL_PATH}")
    print(f"归一化器已保存到: {SCALER_PATH}")

    if test_rmse < baseline_rmse:

        print("\n模型优于持久性预测 baseline")

    else:

        print("\n模型未明显优于 baseline，建议继续调整特征或模型参数")


if __name__ == "__main__":

    train()