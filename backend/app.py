import hashlib
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Dict, Any

import joblib
import numpy as np
import pandas as pd
import psycopg2
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from psycopg2.errors import UniqueViolation
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel

from backend.admin_api import (
    admin_router,
    load_model_access_state,
    load_trained_models,
    load_model_manage_state,
    backfill_trained_model_params,
)


# =========================================================
# 路径配置
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "raw" / "pm25_sites"
PREDICTION_DIR = BASE_DIR / "data" / "predictions"
MODEL_DIR = BASE_DIR / "backend" / "model"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_local_env(env_path: Path):
    """读取项目根目录 .env，允许启动器或手动启动共享数据库配置。"""

    if not env_path.exists():
        return

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value
    except Exception as exc:
        print(f"读取 .env 失败：{exc}", flush=True)


load_local_env(BASE_DIR / ".env")


# =========================================================
# 默认模型参数
# =========================================================

SEQ_LENGTH = 45
PRED_DAYS = 7

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 特征配置
# =========================================================

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

MODEL_CONFIGS = {
    "old_direct7": {
        "label": "普通多变量LSTM",
        "model_file": "lstm_pm25_changchun_old_direct7.pt",
        "scaler_files": [
            "scaler_pm25_old_direct7.save",
        ],
        "features": BASE_FEATURES,
        "description": "使用 pm25、pm10、o3、no2、so2、co 六个基础污染物特征。",
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "seq_length": 45,
        "pred_days": 7,
        "input_size": 6,
    },
    "featured_direct7": {
        "label": "特征增强LSTM",
        "model_file": "lstm_pm25_changchun_featured_direct7.pt",
        "scaler_files": [
            "scaler_pm25_featured_direct7.save",
        ],
        "features": FEATURED_FEATURES,
        "description": "在基础污染物特征上加入时间周期、滞后、滑动均值和差分特征。",
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "seq_length": 45,
        "pred_days": 7,
        "input_size": 21,
    },
    "featured_warning_direct7": {
        "label": "预警导向特征增强LSTM",
        "model_file": "lstm_pm25_changchun_featured_warning_direct7.pt",
        "scaler_files": [
            "scaler_pm25_featured_warning_direct7.save",
            "scaler_pm25_featured_warning.save",
        ],
        "features": FEATURED_FEATURES,
        "description": "面向污染等级预警优化的特征增强LSTM模型。",
        # 该模型历史权重的 lstm.weight_ih_l0 为 [512, 21]，即 hidden_size=128。
        # 运行时仍会以 state_dict 自动反推结果为准，避免后续模型参数再次变化导致加载失败。
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "seq_length": 45,
        "pred_days": 7,
        "input_size": 21,
    },
}

DEFAULT_MODEL_KEY = "featured_direct7"


def get_effective_model_configs() -> Dict[str, Dict[str, Any]]:
    configs = dict(MODEL_CONFIGS)
    trained_models = load_trained_models()

    for key, meta in trained_models.items():
        base_key = meta.get("base_model_key")

        if base_key not in MODEL_CONFIGS:
            continue

        base_config = dict(MODEL_CONFIGS[base_key])

        base_config["label"] = meta.get("label", key)
        base_config["model_file"] = meta.get("model_file", base_config["model_file"])
        base_config["scaler_files"] = meta.get(
            "scaler_files",
            base_config["scaler_files"],
        )
        base_config["description"] = meta.get(
            "description",
            f"管理员训练生成模型，基础模型：{MODEL_CONFIGS[base_key]['label']}",
        )
        base_config["base_model_key"] = base_key
        base_config["base_model_label"] = MODEL_CONFIGS[base_key]["label"]
        base_config["is_custom"] = True

        for param_name in [
            "hidden_size",
            "num_layers",
            "dropout",
            "seq_length",
            "pred_days",
            "input_size",
        ]:
            if param_name in meta:
                base_config[param_name] = meta[param_name]

        configs[key] = base_config

    return configs


# =========================================================
# FastAPI
# =========================================================

app = FastAPI(title="PM2.5监控预警平台")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(admin_router)


@app.on_event("startup")
def startup_backfill_trained_model_params():
    try:
        result = backfill_trained_model_params(save=True)

        if result.get("changed"):
            print("旧模型参数已自动补全：", result.get("updated_models"), flush=True)

        try:
            load_model_and_scaler.cache_clear()
        except Exception:
            pass

    except Exception as exc:
        print(f"旧模型参数自动补全失败：{exc}", flush=True)


# =========================================================
# PostgreSQL 配置
# =========================================================

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "db1")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")


def get_conn():
    conn_str = (
        f"host={DB_HOST} "
        f"port={DB_PORT} "
        f"dbname={DB_NAME} "
        f"user={DB_USER} "
        f"password={DB_PASSWORD}"
    )
    return psycopg2.connect(conn_str, cursor_factory=RealDictCursor)


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            role VARCHAR(20) NOT NULL DEFAULT 'user'
        );
        """
    )

    conn.commit()
    cur.close()
    conn.close()


init_db()


# =========================================================
# Pydantic
# =========================================================

class UserRequest(BaseModel):
    username: str
    password: str
    role: Optional[str] = "user"


class PredictRequest(BaseModel):
    station: Optional[str] = None
    site: Optional[str] = None
    station_id: Optional[str] = None
    model_key: Optional[str] = DEFAULT_MODEL_KEY
    days: Optional[int] = 7
    history_days: Optional[int] = 14


# =========================================================
# LSTM 模型结构
# =========================================================

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


# =========================================================
# 用户注册 / 登录
# =========================================================

@app.post("/api/register")
def register(user: UserRequest):
    username = user.username.strip()
    password = user.password.strip()

    if not username or not password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")

    role = user.role or "user"

    if role not in ["user", "admin"]:
        role = "user"

    password_hash = hashlib.md5(password.encode("utf-8")).hexdigest()

    conn = get_conn()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            INSERT INTO users (username, password, role)
            VALUES (%s, %s, %s)
            """,
            (username, password_hash, role),
        )
        conn.commit()
    except UniqueViolation:
        conn.rollback()
        raise HTTPException(status_code=400, detail="用户名已存在")
    finally:
        cur.close()
        conn.close()

    return {
        "success": True,
        "msg": "注册成功",
        "username": username,
        "role": role,
    }


@app.post("/api/login")
def login(user: UserRequest):
    username = user.username.strip()
    password = user.password.strip()

    if not username or not password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")

    password_hash = hashlib.md5(password.encode("utf-8")).hexdigest()

    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, username, password, role
        FROM users
        WHERE username = %s
        """,
        (username,),
    )

    db_user = cur.fetchone()

    cur.close()
    conn.close()

    if not db_user:
        raise HTTPException(status_code=400, detail="用户不存在")

    if db_user["password"] != password_hash:
        raise HTTPException(status_code=400, detail="密码错误")

    return {
        "success": True,
        "msg": "登录成功",
        "username": db_user["username"],
        "role": db_user["role"],
    }


# =========================================================
# CSV 与特征处理
# =========================================================

def read_csv_auto_encoding(file_path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb2312"]

    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"CSV读取失败：{file_path.name}，错误：{last_error}")


def normalize_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [str(col).strip().lower() for col in df.columns]

    rename_map = {
        "pm2.5": "pm25",
        "pm_25": "pm25",
        "pm2_5": "pm25",
        "日期": "date",
        "时间": "date",
    }

    df = df.rename(columns=rename_map)

    if "date" not in df.columns:
        raise ValueError("CSV缺少 date 日期列")

    missing_cols = [col for col in BASE_FEATURES if col not in df.columns]

    if missing_cols:
        raise ValueError(f"CSV缺少字段：{missing_cols}")

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
        df
        .sort_values("date", ascending=True)
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    df[BASE_FEATURES] = df[BASE_FEATURES].ffill().bfill()

    df = df.dropna(subset=BASE_FEATURES)

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


def add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
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

    df[FEATURED_FEATURES] = df[FEATURED_FEATURES].ffill().bfill()
    df = df.dropna(subset=FEATURED_FEATURES)

    return df


def build_features_dataframe(
    df: pd.DataFrame,
    features: List[str],
    required_seq_length: int = SEQ_LENGTH,
) -> pd.DataFrame:
    df = normalize_base_dataframe(df)

    if set(features) == set(FEATURED_FEATURES):
        df = add_time_features(df)
        df = add_lag_rolling_features(df)

    df[features] = df[features].ffill().bfill()
    df = df.dropna(subset=features)

    if len(df) < required_seq_length:
        raise ValueError(
            f"有效数据不足，至少需要 {required_seq_length} 条，当前只有 {len(df)} 条"
        )

    return df


# =========================================================
# 模型加载与预测
# =========================================================

def resolve_scaler_path(scaler_files: List[str]) -> Path:
    for filename in scaler_files:
        path = MODEL_DIR / filename

        if path.exists():
            return path

    raise FileNotFoundError(f"未找到 scaler 文件：{scaler_files}")


def resolve_model_config(model_key: str) -> Dict[str, Any]:
    backfill_trained_model_params(save=True)

    configs = get_effective_model_configs()

    if model_key not in configs:
        raise HTTPException(status_code=400, detail=f"未知模型：{model_key}")

    return configs[model_key]


def check_model_enabled(model_key: str):
    manage_state = load_model_manage_state()
    hidden = set(manage_state.get("hidden_models", []))

    if model_key in hidden:
        raise HTTPException(status_code=403, detail="该模型已被删除或隐藏")

    trained_models = load_trained_models()

    if model_key in trained_models:
        enabled = bool(trained_models[model_key].get("enabled", True))
    else:
        state = load_model_access_state()
        enabled = state.get(model_key, True)

    if not enabled:
        raise HTTPException(status_code=403, detail="该模型已被管理员禁用")



def safe_torch_load(model_path: Path):
    """兼容 PyTorch 新旧版本的 torch.load。"""
    try:
        return torch.load(model_path, map_location=DEVICE)
    except Exception as exc:
        # PyTorch 2.6+ 对 weights_only 的默认行为更严格；本项目模型文件来自本地训练流程，
        # 如果因为 weights_only 限制导致加载失败，再回退到旧行为。
        if "weights_only" in str(exc):
            try:
                return torch.load(model_path, map_location=DEVICE, weights_only=False)
            except TypeError:
                pass
        raise


def looks_like_state_dict(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False

    has_lstm_key = False
    has_tensor_value = False

    for key, value in obj.items():
        if isinstance(value, torch.Tensor):
            has_tensor_value = True
            key_text = str(key)
            if key_text.endswith("lstm.weight_ih_l0") or "lstm.weight_ih_l0" in key_text:
                has_lstm_key = True

    return has_tensor_value and has_lstm_key


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """去掉 DataParallel / torch.compile / 自定义封装常见前缀。"""
    prefixes = ["module.", "_orig_mod.", "model.", "net."]
    normalized = {}

    for key, value in state_dict.items():
        new_key = str(key)

        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True

        normalized[new_key] = value

    return normalized


def extract_model_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """从多种 checkpoint 保存格式中提取 PM25LSTM 的 state_dict。"""
    if looks_like_state_dict(checkpoint):
        return normalize_state_dict_keys(checkpoint)

    if isinstance(checkpoint, dict):
        for key in [
            "model_state_dict",
            "state_dict",
            "net_state_dict",
            "network_state_dict",
            "weights",
        ]:
            value = checkpoint.get(key)
            if looks_like_state_dict(value):
                return normalize_state_dict_keys(value)

        for key in ["model", "net", "module"]:
            value = checkpoint.get(key)
            if hasattr(value, "state_dict"):
                candidate = value.state_dict()
                if looks_like_state_dict(candidate):
                    return normalize_state_dict_keys(candidate)

    if hasattr(checkpoint, "state_dict"):
        candidate = checkpoint.state_dict()
        if looks_like_state_dict(candidate):
            return normalize_state_dict_keys(candidate)

    raise ValueError(
        "无法从模型文件中识别 PM25LSTM 权重。请确认保存的是 state_dict，"
        "或 checkpoint 中包含 model_state_dict / state_dict 字段。"
    )


def parse_index_from_key(key: str, prefix: str, suffix: str) -> Optional[int]:
    if not key.startswith(prefix) or not key.endswith(suffix):
        return None

    middle = key[len(prefix): -len(suffix)]

    try:
        return int(middle)
    except Exception:
        return None


def infer_model_params_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    hidden_size = HIDDEN_SIZE
    num_layers = NUM_LAYERS
    pred_days = PRED_DAYS
    input_size = 0

    weight_ih_l0 = state_dict.get("lstm.weight_ih_l0")

    if isinstance(weight_ih_l0, torch.Tensor) and weight_ih_l0.ndim == 2:
        hidden_size = int(weight_ih_l0.shape[0] // 4)
        input_size = int(weight_ih_l0.shape[1])

    layer_indexes = []

    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue

        layer_index = parse_index_from_key(str(key), "lstm.weight_ih_l", "")
        if layer_index is not None:
            layer_indexes.append(layer_index)

    if layer_indexes:
        num_layers = max(layer_indexes) + 1

    # 默认结构是 fc.0 -> ReLU -> Dropout -> fc.3；这里按最后一个 fc Linear 权重推断输出天数，
    # 以兼容后续 fc 层索引轻微调整的模型文件。
    fc_weight_candidates = []

    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue

        key_text = str(key)
        layer_index = parse_index_from_key(key_text, "fc.", ".weight")

        if layer_index is not None and value.ndim == 2:
            fc_weight_candidates.append((layer_index, key_text, value))

    if fc_weight_candidates:
        _, _, output_weight = sorted(fc_weight_candidates, key=lambda item: item[0])[-1]
        pred_days = int(output_weight.shape[0])

    return {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "pred_days": pred_days,
        "input_size": input_size,
    }


@lru_cache(maxsize=32)
def load_model_and_scaler(model_key: str):
    config = resolve_model_config(model_key)

    model_path = MODEL_DIR / config["model_file"]
    scaler_path = resolve_scaler_path(config["scaler_files"])

    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型文件：{model_path}")

    features = config["features"]

    scaler = joblib.load(scaler_path)

    checkpoint = safe_torch_load(model_path)
    state_dict = extract_model_state_dict(checkpoint)

    inferred_params = infer_model_params_from_state_dict(state_dict)

    # 关键修复：
    # hidden_size / num_layers / pred_days / input_size 必须以模型权重为准
    hidden_size = int(inferred_params.get("hidden_size") or config.get("hidden_size", HIDDEN_SIZE))
    num_layers = int(inferred_params.get("num_layers") or config.get("num_layers", NUM_LAYERS))
    pred_days = int(inferred_params.get("pred_days") or config.get("pred_days", PRED_DAYS))
    input_size = int(inferred_params.get("input_size") or config.get("input_size", len(features)))

    # 这两个不能从 state_dict 准确反推
    dropout = float(config.get("dropout", DROPOUT))
    seq_length = int(config.get("seq_length", SEQ_LENGTH))

    if input_size and input_size != len(features):
        raise ValueError(
            f"模型输入维度与特征数量不一致：模型 input_size={input_size}，当前特征数量={len(features)}。"
            f"请检查 base_model_key 或模型类型是否正确。"
        )

    scaler_feature_count = getattr(scaler, "n_features_in_", None)

    if scaler_feature_count is not None and int(scaler_feature_count) != len(features):
        raise ValueError(
            f"Scaler 输入维度与特征数量不一致：scaler n_features_in_={int(scaler_feature_count)}，"
            f"当前特征数量={len(features)}。请检查 scaler 文件是否与模型类型匹配。"
        )

    model = PM25LSTM(
        input_size=input_size or len(features),
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=pred_days,
        dropout=dropout,
    ).to(DEVICE)

    model.load_state_dict(state_dict)
    model.eval()

    runtime_config = dict(config)
    runtime_config["hidden_size"] = hidden_size
    runtime_config["num_layers"] = num_layers
    runtime_config["dropout"] = dropout
    runtime_config["pred_days"] = pred_days
    runtime_config["seq_length"] = seq_length
    runtime_config["input_size"] = input_size

    return model, scaler, runtime_config, str(model_path), str(scaler_path)


def inverse_pm25_values(scaler, values, features: List[str]):
    values = np.array(values)

    original_shape = values.shape
    values_flat = values.reshape(-1)

    dummy = np.zeros((len(values_flat), len(features)))

    target_index = features.index(TARGET)
    dummy[:, target_index] = values_flat

    dummy_df = pd.DataFrame(dummy, columns=features)
    inversed = scaler.inverse_transform(dummy_df)

    pm25 = inversed[:, target_index]

    return pm25.reshape(original_shape)


def transform_station_data(df: pd.DataFrame, scaler, features: List[str]):
    feature_df = df[features].astype(float)
    return scaler.transform(feature_df)


def predict_future_direct7(
    df: pd.DataFrame,
    model,
    scaler,
    features: List[str],
    seq_length: int,
    pred_days: int,
):
    df = df.sort_values("date", ascending=True).reset_index(drop=True)

    data = transform_station_data(df, scaler, features)

    if len(data) < seq_length:
        raise ValueError(f"有效数据不足，至少需要 {seq_length} 条，当前只有 {len(data)} 条")

    seq = data[-seq_length:].reshape(1, seq_length, len(features))

    x_tensor = torch.tensor(seq, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        pred_scaled = model(x_tensor).cpu().numpy()[0]

    pred_pm25 = inverse_pm25_values(scaler, pred_scaled, features)
    pred_pm25 = [round(float(v), 2) for v in pred_pm25]

    last_date = df["date"].max()

    future_dates = [
        last_date + pd.Timedelta(days=i + 1)
        for i in range(pred_days)
    ]

    result = []

    for date_value, pm25 in zip(future_dates, pred_pm25):
        result.append(
            {
                "date": date_value.strftime("%Y-%m-%d"),
                "pm25": pm25,
                "predicted_pm25": pm25,
                "level": air_quality_level(pm25),
                "warning": build_warning_message(pm25),
            }
        )

    return result


def predict_history_direct7(
    df: pd.DataFrame,
    model,
    scaler,
    features: List[str],
    seq_length: int,
    pred_days: int,
):
    df = df.sort_values("date", ascending=True).reset_index(drop=True)

    data = transform_station_data(df, scaler, features)

    x_list = []
    date_list = []
    actual_list = []

    max_start = len(data) - seq_length - pred_days + 1

    if max_start <= 0:
        raise ValueError(
            f"历史数据不足，无法生成长期拟合图。至少需要 {seq_length + pred_days} 条，当前只有 {len(data)} 条"
        )

    for i in range(max_start):
        x_list.append(data[i:i + seq_length])

        target_index = i + seq_length

        date_list.append(df.loc[target_index, "date"])
        actual_list.append(df.loc[target_index, "pm25"])

    x_array = np.array(x_list)

    x_tensor = torch.tensor(x_array, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        pred_scaled = model(x_tensor).cpu().numpy()

    first_day_pred_scaled = pred_scaled[:, 0]

    pred_pm25 = inverse_pm25_values(
        scaler,
        first_day_pred_scaled,
        features,
    )

    long_fit = []

    for date_value, actual_value, pred_value in zip(date_list, actual_list, pred_pm25):
        long_fit.append(
            {
                "date": date_value.strftime("%Y-%m-%d"),
                "actual_pm25": round(float(actual_value), 2),
                "predicted_pm25": round(float(pred_value), 2),
            }
        )

    return long_fit


# =========================================================
# 站点工具
# =========================================================

def get_station_files() -> List[Path]:
    return sorted(DATA_DIR.glob("*.csv"))


def safe_station_name(file_path: Path) -> str:
    name = file_path.stem

    for word in [
        "-air-quality",
        "_air_quality",
        "-pm25",
        "_pm25",
    ]:
        name = name.replace(word, "")

    return name


def get_station_file_by_id(station_id: str) -> Path:
    station_id = str(station_id).strip()

    files = get_station_files()

    if not files:
        raise HTTPException(status_code=404, detail="未找到任何站点CSV文件")

    for file_path in files:
        if station_id in [
            file_path.stem,
            file_path.name,
            safe_station_name(file_path),
        ]:
            return file_path

    return files[0]


def get_csv_last_date(df: pd.DataFrame) -> str:
    last_date = df["date"].max()
    return last_date.strftime("%Y-%m-%d")


def air_quality_level(pm25: float) -> str:
    if pm25 <= 35:
        return "优"
    if pm25 <= 75:
        return "良"
    if pm25 <= 115:
        return "轻度污染"
    if pm25 <= 150:
        return "中度污染"
    if pm25 <= 250:
        return "重度污染"
    return "严重污染"


def build_warning_message(pm25: float) -> str:
    level = air_quality_level(pm25)

    if level in ["优", "良"]:
        return "空气质量较好，暂无明显污染风险。"

    if level == "轻度污染":
        return "轻度污染风险，敏感人群建议减少长时间户外活动。"

    if level == "中度污染":
        return "中度污染预警，建议减少户外运动，外出做好防护。"

    if level == "重度污染":
        return "重度污染预警，建议减少外出，儿童、老人及敏感人群避免户外活动。"

    return "严重污染预警，建议尽量避免外出，并采取必要防护措施。"


def build_history_data(df: pd.DataFrame, history_days: int = 14):
    df = df.sort_values("date", ascending=True).reset_index(drop=True)

    history_df = df.tail(history_days)

    result = []

    for _, row in history_df.iterrows():
        pm25 = float(row["pm25"])

        result.append(
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "pm25": round(pm25, 2),
                "level": air_quality_level(pm25),
                "warning": build_warning_message(pm25),
            }
        )

    return result


def build_forecast_summary(forecast: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not forecast:
        return {
            "max_pm25": None,
            "max_level": None,
            "warning_level": "未知",
            "warning_message": "暂无预测结果。",
        }

    max_item = max(forecast, key=lambda item: item["pm25"])

    values = [float(item["pm25"]) for item in forecast]

    increasing_count = 0

    for i in range(1, len(values)):
        if values[i] > values[i - 1]:
            increasing_count += 1

    max_pm25 = max_item["pm25"]
    max_level = air_quality_level(max_pm25)

    if max_pm25 > 150:
        warning_level = "重污染风险"
    elif max_pm25 > 115:
        warning_level = "中度污染风险"
    elif max_pm25 > 75:
        warning_level = "轻度污染风险"
    elif increasing_count >= 3 and max_pm25 > 60:
        warning_level = "上升型污染风险"
    else:
        warning_level = "低风险"

    return {
        "max_pm25": max_pm25,
        "max_date": max_item["date"],
        "max_level": max_level,
        "warning_level": warning_level,
        "warning_message": build_warning_message(max_pm25),
    }


# =========================================================
# 站点列表接口
# =========================================================

@app.get("/stations")
def get_stations():
    files = get_station_files()

    stations = []

    for file_path in files:
        try:
            df = read_csv_auto_encoding(file_path)
            df = normalize_base_dataframe(df)

            last_date = get_csv_last_date(df)

            stations.append(
                {
                    "id": file_path.stem,
                    "name": safe_station_name(file_path),
                    "file": file_path.name,
                    "last_date": last_date,
                    "version": f"数据版本：{last_date}",
                    "rows": int(len(df)),
                }
            )

        except Exception as e:
            stations.append(
                {
                    "id": file_path.stem,
                    "name": safe_station_name(file_path),
                    "file": file_path.name,
                    "last_date": None,
                    "version": "数据版本：读取失败",
                    "rows": 0,
                    "error": str(e),
                }
            )

    return {
        "success": True,
        "stations": stations,
    }



@app.get("/api/stations")
def get_stations_api():
    return get_stations()


# =========================================================
# 预测接口
# =========================================================

@app.post("/predict")
def predict_pm25(req: PredictRequest):
    station_id = req.station or req.site or req.station_id

    if not station_id:
        raise HTTPException(status_code=400, detail="缺少站点参数 station")

    model_key = req.model_key or DEFAULT_MODEL_KEY

    check_model_enabled(model_key)

    try:
        model, scaler, config, model_path, scaler_path = load_model_and_scaler(model_key)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"模型加载失败：{e}")

    seq_length = int(config.get("seq_length", SEQ_LENGTH))
    pred_days = int(config.get("pred_days", PRED_DAYS))

    file_path = get_station_file_by_id(station_id)

    try:
        raw_df = read_csv_auto_encoding(file_path)
        df = build_features_dataframe(
            raw_df,
            config["features"],
            required_seq_length=seq_length,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV读取或预处理失败：{e}")

    last_date = get_csv_last_date(df)

    try:
        history = build_history_data(df, req.history_days or 14)

        forecast = predict_future_direct7(
            df,
            model,
            scaler,
            config["features"],
            seq_length,
            pred_days,
        )

        long_fit = predict_history_direct7(
            df,
            model,
            scaler,
            config["features"],
            seq_length,
            pred_days,
        )

        forecast_summary = build_forecast_summary(forecast)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"预测失败：{e}")

    output_file = PREDICTION_DIR / f"{file_path.stem}_{model_key}_prediction_{last_date}.csv"

    output_df = pd.DataFrame(
        [
            {
                "date": item["date"],
                "type": "history",
                "pm25": item["pm25"],
                "level": item["level"],
                "warning": item["warning"],
            }
            for item in history
        ]
        +
        [
            {
                "date": item["date"],
                "type": "forecast",
                "pm25": item["pm25"],
                "level": item["level"],
                "warning": item["warning"],
            }
            for item in forecast
        ]
    )

    output_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    return {
        "success": True,
        "station": {
            "id": file_path.stem,
            "name": safe_station_name(file_path),
            "file": file_path.name,
        },
        "model": {
            "key": model_key,
            "label": config["label"],
            "description": config["description"],
            "model_file": Path(model_path).name,
            "scaler_file": Path(scaler_path).name,
            "hidden_size": int(config.get("hidden_size", HIDDEN_SIZE)),
            "num_layers": int(config.get("num_layers", NUM_LAYERS)),
            "dropout": float(config.get("dropout", DROPOUT)),
            "seq_length": seq_length,
            "pred_days": pred_days,
            "input_size": int(config.get("input_size", 0)),
        },
        "last_date": last_date,
        "version": f"数据版本：{last_date}",
        "history": history,
        "forecast": forecast,
        "forecast_summary": forecast_summary,
        "long_fit": long_fit,
        "output_file": str(output_file),
    }



@app.post("/api/predict")
def predict_pm25_api(req: PredictRequest):
    return predict_pm25(req)


# =========================================================
# CSV 上传兼容接口
# =========================================================

@app.post("/upload_csv")
@app.post("/api/upload_csv")
async def upload_csv_legacy(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="只支持上传CSV文件")

    save_path = DATA_DIR / Path(file.filename).name

    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        df = read_csv_auto_encoding(save_path)
        df = normalize_base_dataframe(df)
        last_date = get_csv_last_date(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV保存成功，但解析失败：{e}")

    return {
        "success": True,
        "msg": "CSV上传成功",
        "file": file.filename,
        "station_id": save_path.stem,
        "station_name": safe_station_name(save_path),
        "last_date": last_date,
        "version": f"数据版本：{last_date}",
        "rows": int(len(df)),
    }