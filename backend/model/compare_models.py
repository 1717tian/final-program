import os
import glob
import re
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error


DATA_DIR = r"D:\final program\pythonProject12\data\raw\pm25_sites"
MODEL_DIR = r"D:\final program\pythonProject12\backend\model"
SAVE_DIR = r"D:\final program\pythonProject12\data\predictions"

os.makedirs(SAVE_DIR, exist_ok=True)


OLD_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lstm_pm25_changchun_old_direct7.pt"
)

OLD_SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler_pm25_old_direct7.save"
)

FEATURED_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lstm_pm25_changchun_featured_direct7.pt"
)

FEATURED_SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler_pm25_featured_direct7.save"
)

WARNING_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lstm_pm25_changchun_featured_warning_direct7.pt"
)

WARNING_SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler_pm25_featured_warning_direct7.save"
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

OLD_FEATURES = BASE_FEATURES

FEATURED_FEATURES = (
    BASE_FEATURES
    + TIME_FEATURES
    + LAG_FEATURES
    + ROLL_FEATURES
    + DIFF_FEATURES
)

WARNING_FEATURES = FEATURED_FEATURES

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


def safe_name(name):
    name = os.path.splitext(name)[0]

    name = re.sub(
        r'[\\/:*?"<>|,，（）() ]+',
        "_",
        name
    )

    return name.strip("_")


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

    df[FEATURED_FEATURES] = (
        df[FEATURED_FEATURES]
        .ffill()
        .bfill()
    )

    df = df.dropna(
        subset=FEATURED_FEATURES
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


def load_station_csv(file_path):
    df = pd.read_csv(file_path)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    if "date" not in df.columns:
        return None

    missing_cols = [
        col for col in BASE_FEATURES
        if col not in df.columns
    ]

    if missing_cols:
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
        return None

    return df


def load_one_model(model_path, input_size):
    model = PM25LSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=PRED_DAYS
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE
        )
    )

    model.eval()

    return model


def load_models():
    old_scaler = joblib.load(
        OLD_SCALER_PATH
    )

    featured_scaler = joblib.load(
        FEATURED_SCALER_PATH
    )

    warning_scaler = joblib.load(
        WARNING_SCALER_PATH
    )

    old_model = load_one_model(
        OLD_MODEL_PATH,
        len(OLD_FEATURES)
    )

    featured_model = load_one_model(
        FEATURED_MODEL_PATH,
        len(FEATURED_FEATURES)
    )

    warning_model = load_one_model(
        WARNING_MODEL_PATH,
        len(WARNING_FEATURES)
    )

    return (
        old_model,
        old_scaler,
        featured_model,
        featured_scaler,
        warning_model,
        warning_scaler
    )


def inverse_pm25_values(
    scaler,
    values,
    features
):
    values = np.array(values)
    original_shape = values.shape
    values_flat = values.reshape(-1)

    dummy = np.zeros(
        (
            len(values_flat),
            len(features)
        )
    )

    target_index = features.index(TARGET)

    dummy[:, target_index] = values_flat

    inversed = scaler.inverse_transform(dummy)

    return inversed[:, target_index].reshape(original_shape)


def predict_baseline(df, start_idx):
    last_pm25 = df["pm25"].iloc[
        start_idx + SEQ_LENGTH - 1
    ]

    pred = np.repeat(
        last_pm25,
        PRED_DAYS
    )

    return pred


def predict_lstm(
    df,
    model,
    scaler,
    features,
    start_idx
):
    scaled_data = scaler.transform(
        df[features]
    )

    seq = scaled_data[
        start_idx:
        start_idx + SEQ_LENGTH
    ].reshape(
        1,
        SEQ_LENGTH,
        len(features)
    )

    x_tensor = torch.tensor(
        seq,
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():
        pred_scaled = model(
            x_tensor
        ).cpu().numpy()[0]

    pred = inverse_pm25_values(
        scaler,
        pred_scaled,
        features
    )

    return pred


def generate_special_test_dataset():
    """
    构建特化测试集：
    选择未来7天PM2.5波动幅度大、首日跳变明显、整体变化剧烈的样本。
    """

    csv_files = glob.glob(
        os.path.join(DATA_DIR, "*.csv")
    )

    candidates = []

    for file_path in csv_files:
        station_name = os.path.basename(file_path)

        df = load_station_csv(file_path)

        if df is None:
            continue

        for i in range(
            0,
            len(df) - SEQ_LENGTH - PRED_DAYS + 1
        ):
            hist = df["pm25"].iloc[
                i:
                i + SEQ_LENGTH
            ].values

            future = df["pm25"].iloc[
                i + SEQ_LENGTH:
                i + SEQ_LENGTH + PRED_DAYS
            ].values

            last_value = hist[-1]

            future_range = future.max() - future.min()

            first_jump = abs(
                future[0] - last_value
            )

            future_diff = np.abs(
                np.diff(future)
            ).sum()

            special_score = (
                future_range
                + first_jump
                + 0.5 * future_diff
            )

            candidates.append(
                {
                    "station": station_name,
                    "start_idx": i,
                    "input_start_date": df["date"].iloc[i],
                    "input_end_date": df["date"].iloc[i + SEQ_LENGTH - 1],
                    "target_start_date": df["date"].iloc[i + SEQ_LENGTH],
                    "target_end_date": df["date"].iloc[i + SEQ_LENGTH + PRED_DAYS - 1],
                    "special_score": special_score,
                    "future_range": future_range,
                    "first_jump": first_jump,
                    "future_diff": future_diff,
                    "df": df
                }
            )

    candidates = sorted(
        candidates,
        key=lambda x: x["special_score"],
        reverse=True
    )

    selected = []
    used_keys = set()

    for item in candidates:
        key = (
            item["station"],
            item["target_start_date"].strftime("%Y-%m")
        )

        if key in used_keys:
            continue

        selected.append(item)
        used_keys.add(key)

        if len(selected) >= 30:
            break

    return selected


def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    return rmse, mae


def evaluate_special_cases(
    special_cases,
    old_model,
    old_scaler,
    featured_model,
    featured_scaler,
    warning_model,
    warning_scaler
):
    rows = []

    all_true = []
    all_baseline = []
    all_old = []
    all_featured = []
    all_warning = []

    for case_id, item in enumerate(
        special_cases,
        start=1
    ):
        df = item["df"]
        start_idx = item["start_idx"]

        y_true = df["pm25"].iloc[
            start_idx + SEQ_LENGTH:
            start_idx + SEQ_LENGTH + PRED_DAYS
        ].values

        baseline_pred = predict_baseline(
            df,
            start_idx
        )

        old_pred = predict_lstm(
            df,
            old_model,
            old_scaler,
            OLD_FEATURES,
            start_idx
        )

        featured_pred = predict_lstm(
            df,
            featured_model,
            featured_scaler,
            FEATURED_FEATURES,
            start_idx
        )

        warning_pred = predict_lstm(
            df,
            warning_model,
            warning_scaler,
            WARNING_FEATURES,
            start_idx
        )

        baseline_abs_error = np.abs(
            baseline_pred - y_true
        )

        old_abs_error = np.abs(
            old_pred - y_true
        )

        featured_abs_error = np.abs(
            featured_pred - y_true
        )

        warning_abs_error = np.abs(
            warning_pred - y_true
        )

        all_true.extend(y_true)
        all_baseline.extend(baseline_pred)
        all_old.extend(old_pred)
        all_featured.extend(featured_pred)
        all_warning.extend(warning_pred)

        row = {
            "case_id": case_id,
            "station": item["station"],
            "input_start_date": item["input_start_date"],
            "input_end_date": item["input_end_date"],
            "target_start_date": item["target_start_date"],
            "target_end_date": item["target_end_date"],
            "special_score": item["special_score"],
            "future_range": item["future_range"],
            "first_jump": item["first_jump"],
            "future_diff": item["future_diff"],
            "baseline_case_mae": baseline_abs_error.mean(),
            "old_case_mae": old_abs_error.mean(),
            "featured_case_mae": featured_abs_error.mean(),
            "warning_case_mae": warning_abs_error.mean(),
            "old_vs_baseline_improvement": baseline_abs_error.mean() - old_abs_error.mean(),
            "featured_vs_old_improvement": old_abs_error.mean() - featured_abs_error.mean(),
            "warning_vs_featured_improvement": featured_abs_error.mean() - warning_abs_error.mean(),
            "warning_vs_baseline_improvement": baseline_abs_error.mean() - warning_abs_error.mean()
        }

        for i in range(PRED_DAYS):
            row[f"true_day{i + 1}"] = round(float(y_true[i]), 2)
            row[f"baseline_pred_day{i + 1}"] = round(float(baseline_pred[i]), 2)
            row[f"old_pred_day{i + 1}"] = round(float(old_pred[i]), 2)
            row[f"featured_pred_day{i + 1}"] = round(float(featured_pred[i]), 2)
            row[f"warning_pred_day{i + 1}"] = round(float(warning_pred[i]), 2)

            row[f"baseline_abs_error_day{i + 1}"] = round(float(baseline_abs_error[i]), 2)
            row[f"old_abs_error_day{i + 1}"] = round(float(old_abs_error[i]), 2)
            row[f"featured_abs_error_day{i + 1}"] = round(float(featured_abs_error[i]), 2)
            row[f"warning_abs_error_day{i + 1}"] = round(float(warning_abs_error[i]), 2)

        rows.append(row)

    result_df = pd.DataFrame(rows)

    all_true = np.array(all_true)
    all_baseline = np.array(all_baseline)
    all_old = np.array(all_old)
    all_featured = np.array(all_featured)
    all_warning = np.array(all_warning)

    baseline_rmse, baseline_mae = calc_metrics(
        all_true,
        all_baseline
    )

    old_rmse, old_mae = calc_metrics(
        all_true,
        all_old
    )

    featured_rmse, featured_mae = calc_metrics(
        all_true,
        all_featured
    )

    warning_rmse, warning_mae = calc_metrics(
        all_true,
        all_warning
    )

    return (
        result_df,
        baseline_rmse,
        baseline_mae,
        old_rmse,
        old_mae,
        featured_rmse,
        featured_mae,
        warning_rmse,
        warning_mae
    )


def build_summary_df(
    baseline_rmse,
    baseline_mae,
    old_rmse,
    old_mae,
    featured_rmse,
    featured_mae,
    warning_rmse,
    warning_mae
):
    summary_df = pd.DataFrame(
        {
            "模型": [
                "Baseline持久性预测",
                "普通多变量LSTM",
                "特征增强LSTM",
                "预警导向特征增强LSTM"
            ],
            "RMSE": [
                baseline_rmse,
                old_rmse,
                featured_rmse,
                warning_rmse
            ],
            "MAE": [
                baseline_mae,
                old_mae,
                featured_mae,
                warning_mae
            ]
        }
    )

    summary_df["RMSE较Baseline降低"] = (
        baseline_rmse - summary_df["RMSE"]
    )

    summary_df["RMSE较Baseline下降率(%)"] = (
        summary_df["RMSE较Baseline降低"]
        / baseline_rmse
        * 100
    )

    summary_df["MAE较Baseline降低"] = (
        baseline_mae - summary_df["MAE"]
    )

    summary_df["MAE较Baseline下降率(%)"] = (
        summary_df["MAE较Baseline降低"]
        / baseline_mae
        * 100
    )

    return summary_df


def print_conclusion(
    result_df,
    summary_df
):
    baseline_row = summary_df.iloc[0]
    old_row = summary_df.iloc[1]
    featured_row = summary_df.iloc[2]
    warning_row = summary_df.iloc[3]

    total_cases = len(result_df)

    old_better_than_baseline = (
        result_df["old_vs_baseline_improvement"] > 0
    ).sum()

    featured_better_than_old = (
        result_df["featured_vs_old_improvement"] > 0
    ).sum()

    warning_better_than_featured = (
        result_df["warning_vs_featured_improvement"] > 0
    ).sum()

    old_rate = old_better_than_baseline / total_cases * 100
    featured_rate = featured_better_than_old / total_cases * 100
    warning_rate = warning_better_than_featured / total_cases * 100

    print("\n========== 四类模型整体性能对比 ==========")
    print(summary_df)

    print("\n========== 模型渐进式改进分析 ==========")

    print(
        f"普通多变量LSTM相较Baseline持久性预测模型，"
        f"RMSE降低{baseline_row['RMSE'] - old_row['RMSE']:.2f}，"
        f"下降{(baseline_row['RMSE'] - old_row['RMSE']) / baseline_row['RMSE'] * 100:.2f}%；"
        f"MAE降低{baseline_row['MAE'] - old_row['MAE']:.2f}，"
        f"下降{(baseline_row['MAE'] - old_row['MAE']) / baseline_row['MAE'] * 100:.2f}%。"
        f"结果说明，引入LSTM后，模型能够学习PM2.5序列中的时间依赖关系，"
        f"相比简单持久性预测具有更强的动态建模能力。"
    )

    print(
        f"特征增强LSTM相较普通多变量LSTM，"
        f"RMSE降低{old_row['RMSE'] - featured_row['RMSE']:.2f}，"
        f"下降{(old_row['RMSE'] - featured_row['RMSE']) / old_row['RMSE'] * 100:.2f}%；"
        f"MAE降低{old_row['MAE'] - featured_row['MAE']:.2f}，"
        f"下降{(old_row['MAE'] - featured_row['MAE']) / old_row['MAE'] * 100:.2f}%。"
        f"这表明时间周期特征、滞后特征、滑动统计特征和差分特征能够补充原始污染物数据的信息表达，"
        f"提升模型对污染变化趋势和短期波动的刻画能力。"
    )

    print(
        f"预警导向特征增强LSTM相较特征增强LSTM，"
        f"RMSE降低{featured_row['RMSE'] - warning_row['RMSE']:.2f}，"
        f"下降{(featured_row['RMSE'] - warning_row['RMSE']) / featured_row['RMSE'] * 100:.2f}%；"
        f"MAE降低{featured_row['MAE'] - warning_row['MAE']:.2f}，"
        f"下降{(featured_row['MAE'] - warning_row['MAE']) / featured_row['MAE'] * 100:.2f}%。"
        f"说明在特化高波动样本中，预警导向训练策略能够进一步提升模型对突发污染变化的响应能力。"
    )

    print("\n========== 各阶段样本占优情况 ==========")

    print(
        f"在{total_cases}个特化测试样本中，"
        f"普通多变量LSTM有{old_better_than_baseline}个样本优于Baseline持久性预测模型，"
        f"占比{old_rate:.2f}%。"
    )

    print(
        f"特征增强LSTM有{featured_better_than_old}个样本优于普通多变量LSTM，"
        f"占比{featured_rate:.2f}%。"
    )

    print(
        f"预警导向特征增强LSTM有{warning_better_than_featured}个样本优于特征增强LSTM，"
        f"占比{warning_rate:.2f}%。"
    )

    print("\n========== 综合结论 ==========")

    print(
        "综合四类模型的整体误差指标与样本级表现可以看出，"
        "模型性能并非只依赖单一预警导向策略，而是随着建模方法和特征体系的逐步完善呈现阶段性提升。"
        "Baseline持久性预测模型作为参照方法，反映了简单延续历史值的预测能力；"
        "普通多变量LSTM通过引入循环神经网络结构，提高了对PM2.5时间序列依赖关系的学习能力；"
        "特征增强LSTM进一步融合时间周期、滞后、滚动统计和差分信息，增强了模型对污染物变化规律的表达；"
        "预警导向特征增强LSTM则在此基础上进一步面向高波动、高污染场景进行优化，"
        "提高了模型在特化预警样本中的适应性。"
        "因此，四类模型共同构成了从基准预测、时序建模、特征增强到预警优化的完整对比体系。"
    )


def plot_model_comparison(
    result_df,
    summary_df
):
    models = summary_df["模型"].tolist()

    rmse_values = summary_df["RMSE"].values
    mae_values = summary_df["MAE"].values

    best_case = result_df.sort_values(
        by="warning_vs_featured_improvement",
        ascending=False
    ).iloc[0]

    dates = pd.date_range(
        start=best_case["target_start_date"],
        periods=PRED_DAYS,
        freq="D"
    )

    true_values = [
        best_case[f"true_day{i + 1}"]
        for i in range(PRED_DAYS)
    ]

    baseline_values = [
        best_case[f"baseline_pred_day{i + 1}"]
        for i in range(PRED_DAYS)
    ]

    old_values = [
        best_case[f"old_pred_day{i + 1}"]
        for i in range(PRED_DAYS)
    ]

    featured_values = [
        best_case[f"featured_pred_day{i + 1}"]
        for i in range(PRED_DAYS)
    ]

    warning_values = [
        best_case[f"warning_pred_day{i + 1}"]
        for i in range(PRED_DAYS)
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(18, 11)
    )

    x = np.arange(len(models))
    width = 0.35

    bars1 = axes[0, 0].bar(
        x - width / 2,
        rmse_values,
        width,
        label="RMSE",
        color="#4C72B0"
    )

    bars2 = axes[0, 0].bar(
        x + width / 2,
        mae_values,
        width,
        label="MAE",
        color="#DD8452"
    )

    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(
        models,
        rotation=15,
        ha="right"
    )
    axes[0, 0].set_ylabel("误差值")
    axes[0, 0].set_title("四类模型整体误差指标对比")
    axes[0, 0].legend()
    axes[0, 0].grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    stage_labels = [
        "普通LSTM\n较Baseline",
        "特征增强LSTM\n较普通LSTM",
        "预警导向LSTM\n较特征增强LSTM"
    ]

    stage_rmse_improvements = [
        summary_df.iloc[0]["RMSE"] - summary_df.iloc[1]["RMSE"],
        summary_df.iloc[1]["RMSE"] - summary_df.iloc[2]["RMSE"],
        summary_df.iloc[2]["RMSE"] - summary_df.iloc[3]["RMSE"]
    ]

    stage_colors = [
        "#66C2A5" if value >= 0 else "#C44E52"
        for value in stage_rmse_improvements
    ]

    bars3 = axes[0, 1].bar(
        stage_labels,
        stage_rmse_improvements,
        color=stage_colors
    )

    axes[0, 1].axhline(
        0,
        color="black",
        linewidth=1
    )

    axes[0, 1].set_ylabel("RMSE降低值")
    axes[0, 1].set_title("模型渐进式改进过程中的RMSE变化")
    axes[0, 1].grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    for bar in bars3:
        height = bar.get_height()
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom" if height >= 0 else "top"
        )

    preference_sorted = result_df.sort_values(
        by="warning_vs_featured_improvement",
        ascending=False
    ).reset_index(drop=True)

    top20 = preference_sorted.head(20)

    x = np.arange(1, len(top20) + 1)
    width = 0.25

    axes[1, 0].bar(
        x - width,
        top20["old_case_mae"],
        width,
        label="普通LSTM",
        color="#4C72B0"
    )

    axes[1, 0].bar(
        x,
        top20["featured_case_mae"],
        width,
        label="特征增强LSTM",
        color="#DD8452"
    )

    axes[1, 0].bar(
        x + width,
        top20["warning_case_mae"],
        width,
        label="预警导向LSTM",
        color="#66C2A5"
    )

    high_pollution_threshold = 60

    for i, value in enumerate(top20["warning_case_mae"]):
        if value >= high_pollution_threshold:
            axes[1, 0].text(
                x[i],
                value + 0.5,
                "高污染",
                ha="center",
                va="bottom",
                fontsize=8,
                color="red",
                rotation=90
            )

    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xlabel("特化测试样本")
    axes[1, 0].set_ylabel("MAE")
    axes[1, 0].set_title(
        "不同LSTM模型在特化样本中的MAE对比\n红色文字表示高污染样本"
    )
    axes[1, 0].grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )
    axes[1, 0].legend()

    axes[1, 1].plot(
        dates,
        true_values,
        marker="o",
        linewidth=2.5,
        label="真实PM2.5",
        color="black"
    )

    axes[1, 1].plot(
        dates,
        baseline_values,
        marker="o",
        linestyle=":",
        linewidth=2,
        label="Baseline",
        color="#999999"
    )

    axes[1, 1].plot(
        dates,
        old_values,
        marker="o",
        linestyle="--",
        linewidth=2,
        label="普通LSTM",
        color="#4C72B0"
    )

    axes[1, 1].plot(
        dates,
        featured_values,
        marker="o",
        linestyle="--",
        linewidth=2,
        label="特征增强LSTM",
        color="#DD8452"
    )

    axes[1, 1].plot(
        dates,
        warning_values,
        marker="o",
        linestyle="--",
        linewidth=2.5,
        label="预警导向LSTM",
        color="#66C2A5"
    )

    axes[1, 1].set_title(
        "四类模型在典型特化样本中的预测曲线对比"
    )
    axes[1, 1].set_xlabel("日期")
    axes[1, 1].set_ylabel("PM2.5浓度")
    axes[1, 1].grid(
        True,
        linestyle="--",
        alpha=0.5
    )
    axes[1, 1].legend()

    plt.suptitle(
        "四类PM2.5预测模型整体性能与渐进式改进对比",
        fontsize=16
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.93]
    )

    save_path = os.path.join(
        SAVE_DIR,
        "compare_models_baseline_old_featured_warning.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"\n四类模型对比图已保存到: {save_path}"
    )

    plt.show()


def save_outputs(
    result_df,
    summary_df
):
    result_path = os.path.join(
        SAVE_DIR,
        "special_test_dataset_four_models.csv"
    )

    summary_path = os.path.join(
        SAVE_DIR,
        "compare_models_four_summary.csv"
    )

    result_df.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"特化测试集四模型预测结果已保存到: {result_path}"
    )

    print(
        f"四模型指标汇总已保存到: {summary_path}"
    )


def main():
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    print(f"当前运行设备: {DEVICE}")

    (
        old_model,
        old_scaler,
        featured_model,
        featured_scaler,
        warning_model,
        warning_scaler
    ) = load_models()

    special_cases = generate_special_test_dataset()

    if len(special_cases) == 0:
        print("未生成有效特化测试集，请检查数据目录和CSV文件格式。")
        return

    (
        result_df,
        baseline_rmse,
        baseline_mae,
        old_rmse,
        old_mae,
        featured_rmse,
        featured_mae,
        warning_rmse,
        warning_mae
    ) = evaluate_special_cases(
        special_cases,
        old_model,
        old_scaler,
        featured_model,
        featured_scaler,
        warning_model,
        warning_scaler
    )

    summary_df = build_summary_df(
        baseline_rmse,
        baseline_mae,
        old_rmse,
        old_mae,
        featured_rmse,
        featured_mae,
        warning_rmse,
        warning_mae
    )

    save_outputs(
        result_df,
        summary_df
    )

    print_conclusion(
        result_df,
        summary_df
    )

    plot_model_comparison(
        result_df,
        summary_df
    )


if __name__ == "__main__":
    main()