import os
import glob
import re
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


DATA_DIR = r"D:\final program\pythonProject12\data\raw\pm25_sites"
MODEL_DIR = r"D:\final program\pythonProject12\backend\model"
PREDICTION_DIR = r"D:\final program\pythonProject12\data\predictions"

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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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


def load_model_and_scaler():

    print("正在加载模型...")

    scaler = joblib.load(
        SCALER_PATH
    )

    model = PM25LSTM(
        input_size=len(FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=PRED_DAYS
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    print("模型加载完成")

    return model, scaler


def safe_filename(name):

    name = os.path.splitext(name)[0]

    name = re.sub(
        r'[\\/:*?"<>|,，（）() ]+',
        "_",
        name
    )

    name = name.strip("_")

    return name


def load_station_data(file_path):

    df = pd.read_csv(
        file_path
    )

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    if "date" not in df.columns:

        raise ValueError(
            "CSV 文件缺少 date 列"
        )

    missing_cols = [
        col for col in BASE_FEATURES
        if col not in df.columns
    ]

    if missing_cols:

        raise ValueError(
            f"CSV 文件缺少字段: {missing_cols}"
        )

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

    df = add_time_features(
        df
    )

    df = add_lag_rolling_features(
        df
    )

    if len(df) < SEQ_LENGTH:

        raise ValueError(
            f"该站点有效数据不足，至少需要 {SEQ_LENGTH} 条"
        )

    return df


def transform_station_data(
    df,
    scaler
):

    scaled_data = scaler.transform(
        df[FEATURES]
    )

    return scaled_data


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

    target_index = FEATURES.index(
        TARGET
    )

    dummy[:, target_index] = values_flat

    inversed = scaler.inverse_transform(
        dummy
    )

    pm25 = inversed[:, target_index]

    return pm25.reshape(
        original_shape
    )


def predict_future_direct7(
    df,
    model,
    scaler
):

    data = transform_station_data(
        df,
        scaler
    )

    seq = data[
        -SEQ_LENGTH:
    ].reshape(
        1,
        SEQ_LENGTH,
        len(FEATURES)
    )

    x_tensor = torch.tensor(
        seq,
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():

        pred_scaled = model(
            x_tensor
        ).cpu().numpy()[0]

    pred_pm25 = inverse_pm25_values(
        scaler,
        pred_scaled
    )

    pred_pm25 = [
        round(float(v), 2)
        for v in pred_pm25
    ]

    last_date = df["date"].max()

    future_dates = [
        last_date + pd.Timedelta(
            days=i + 1
        )
        for i in range(PRED_DAYS)
    ]

    result = pd.DataFrame(
        {
            "date": future_dates,
            "predicted_pm25": pred_pm25
        }
    )

    return result


def predict_history_direct7(
    df,
    model,
    scaler
):

    data = transform_station_data(
        df,
        scaler
    )

    X = []

    for i in range(
        len(data) - SEQ_LENGTH - PRED_DAYS + 1
    ):

        X.append(
            data[i:i + SEQ_LENGTH]
        )

    if len(X) == 0:

        raise ValueError(
            "历史数据不足，无法绘制长期拟合图"
        )

    X = np.array(
        X
    )

    x_tensor = torch.tensor(
        X,
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():

        pred_scaled = model(
            x_tensor
        ).cpu().numpy()

    first_day_pred_scaled = pred_scaled[
        :,
        0
    ]

    hist_pred = inverse_pm25_values(
        scaler,
        first_day_pred_scaled
    )

    hist_dates = df["date"].iloc[
        SEQ_LENGTH:
        SEQ_LENGTH + len(hist_pred)
    ].reset_index(
        drop=True
    )

    return hist_dates, hist_pred


def air_quality_level(pm25):

    if pm25 <= 35:

        return "优"

    elif pm25 <= 75:

        return "良"

    elif pm25 <= 115:

        return "轻度污染"

    elif pm25 <= 150:

        return "中度污染"

    elif pm25 <= 250:

        return "重度污染"

    else:

        return "严重污染"


def save_result(
    result,
    station_name
):

    os.makedirs(
        PREDICTION_DIR,
        exist_ok=True
    )

    file_name = safe_filename(
        station_name
    )

    result_path = os.path.join(
        PREDICTION_DIR,
        f"{file_name}_featured_direct7_future.csv"
    )

    result.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"预测结果已保存到: {result_path}"
    )


def save_figure(
    station_name
):

    os.makedirs(
        PREDICTION_DIR,
        exist_ok=True
    )

    file_name = safe_filename(
        station_name
    )

    figure_path = os.path.join(
        PREDICTION_DIR,
        f"{file_name}_featured_direct7_prediction.png"
    )

    plt.savefig(
        figure_path,
        dpi=300
    )

    print(
        f"预测图已保存到: {figure_path}"
    )


class PM25PredictApp:

    def __init__(
        self,
        root,
        csv_files,
        model,
        scaler
    ):

        self.root = root
        self.csv_files = csv_files
        self.model = model
        self.scaler = scaler

        self.root.title(
            "长春 PM2.5 特征增强 Direct7 预测系统"
        )

        self.fig, (
            self.ax1,
            self.ax2
        ) = plt.subplots(
            2,
            1,
            figsize=(14, 10)
        )

        self.canvas = FigureCanvasTkAgg(
            self.fig,
            master=root
        )

        self.canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )

        toolbar = NavigationToolbar2Tk(
            self.canvas,
            root
        )

        toolbar.update()

        control_frame = tk.Frame(
            root
        )

        control_frame.pack(
            fill=tk.X,
            pady=5
        )

        tk.Label(
            control_frame,
            text="选择站点：",
            font=("SimHei", 11)
        ).pack(
            side=tk.LEFT,
            padx=5
        )

        self.station_var = tk.StringVar()

        self.station_combo = ttk.Combobox(
            control_frame,
            textvariable=self.station_var,
            state="readonly",
            width=55
        )

        self.station_combo["values"] = [
            os.path.basename(file)
            for file in self.csv_files
        ]

        self.station_combo.current(
            0
        )

        self.station_combo.pack(
            side=tk.LEFT,
            padx=5
        )

        self.station_combo.bind(
            "<<ComboboxSelected>>",
            self.on_station_change
        )

        self.info_label = tk.Label(
            root,
            text="",
            font=("SimHei", 11)
        )

        self.info_label.pack(
            pady=5
        )

        self.update_plot()

    def on_station_change(
        self,
        event=None
    ):

        self.update_plot()

    def update_plot(
        self
    ):

        idx = self.station_combo.current()

        file_path = self.csv_files[
            idx
        ]

        station_name = os.path.basename(
            file_path
        )

        print(
            f"\n当前选择站点: {station_name}"
        )

        try:

            df = load_station_data(
                file_path
            )

            result = predict_future_direct7(
                df=df,
                model=self.model,
                scaler=self.scaler
            )

            hist_dates, hist_pred = predict_history_direct7(
                df=df,
                model=self.model,
                scaler=self.scaler
            )

            save_result(
                result,
                station_name
            )

            self.draw_result(
                df,
                result,
                hist_dates,
                hist_pred,
                station_name
            )

            last_date = df["date"].max()

            info_text = (
                f"站点：{station_name}    "
                f"最后日期：{last_date.date()}    "
                f"预测日期：{result['date'].iloc[0].date()} "
                f"至 {result['date'].iloc[-1].date()}"
            )

            self.info_label.config(
                text=info_text
            )

            print("\n未来7天 PM2.5 预测结果:")
            print(result)

            print("\n空气质量预警:")

            for i in range(len(result)):

                pm25 = result.loc[
                    i,
                    "predicted_pm25"
                ]

                level = air_quality_level(
                    pm25
                )

                print(
                    f"{result.loc[i, 'date'].date()} "
                    f"PM2.5={pm25:.2f} "
                    f"空气质量: {level}"
                )

        except Exception as e:

            self.info_label.config(
                text=f"错误：{e}"
            )

            print(
                f"错误：{e}"
            )

    def draw_result(
        self,
        df,
        result,
        hist_dates,
        hist_pred,
        station_name
    ):

        self.ax1.clear()
        self.ax2.clear()

        plt.rcParams[
            "font.sans-serif"
        ] = ["SimHei"]

        plt.rcParams[
            "axes.unicode_minus"
        ] = False

        hist_df = df.tail(
            60
        )

        self.ax1.plot(
            hist_df["date"],
            hist_df["pm25"],
            label="历史PM2.5",
            linewidth=2,
            color="blue"
        )

        last_date = df["date"].iloc[-1]

        last_pm25 = df["pm25"].iloc[-1]

        future_dates_connected = [
            last_date
        ] + list(
            result["date"]
        )

        future_values_connected = [
            last_pm25
        ] + list(
            result["predicted_pm25"]
        )

        self.ax1.plot(
            future_dates_connected,
            future_values_connected,
            label="未来7天预测",
            linestyle="--",
            linewidth=3,
            color="red",
            marker="o"
        )

        for x, y in zip(
            result["date"],
            result["predicted_pm25"]
        ):

            self.ax1.text(
                x,
                y,
                f"{y:.1f}",
                fontsize=9
            )

        self.ax1.set_title(
            f"{station_name} - 未来7天PM2.5预测"
        )

        self.ax1.set_xlabel(
            "日期"
        )

        self.ax1.set_ylabel(
            "PM2.5"
        )

        self.ax1.legend()
        self.ax1.grid(True)

        self.ax2.plot(
            df["date"],
            df["pm25"],
            label="历史真实值",
            linewidth=1.8,
            color="green"
        )

        self.ax2.plot(
            hist_dates,
            hist_pred,
            label="模型历史预测值",
            linestyle="--",
            linewidth=1.8,
            color="orange"
        )

        self.ax2.set_title(
            f"{station_name} - 长期历史拟合效果"
        )

        self.ax2.set_xlabel(
            "日期"
        )

        self.ax2.set_ylabel(
            "PM2.5"
        )

        self.ax2.legend()
        self.ax2.grid(True)

        self.fig.tight_layout()

        self.canvas.draw()

        save_figure(
            station_name
        )


if __name__ == "__main__":

    csv_files = glob.glob(
        os.path.join(DATA_DIR, "*.csv")
    )

    if len(csv_files) == 0:

        raise FileNotFoundError(
            "未找到 CSV 文件"
        )

    model, scaler = load_model_and_scaler()

    root = tk.Tk()

    app = PM25PredictApp(
        root=root,
        csv_files=csv_files,
        model=model,
        scaler=scaler
    )

    root.mainloop()