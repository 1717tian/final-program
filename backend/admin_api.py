import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel


# =========================================================
# 路径配置
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
DATASET_DIR = BASE_DIR / "data" / "raw" / "pm25_sites"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "training_jobs"
MODEL_DIR = BACKEND_DIR / "model"
CONFIG_DIR = BASE_DIR / "config"
LOG_DIR = BASE_DIR / "logs"

DATASET_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ACCESS_FILE = CONFIG_DIR / "model_access.json"
TRAIN_JOB_FILE = LOG_DIR / "train_jobs.json"
TRAINED_MODEL_FILE = CONFIG_DIR / "trained_models.json"
MODEL_MANAGE_FILE = CONFIG_DIR / "model_manage.json"

admin_router = APIRouter()

RUNNING_PROCESSES: Dict[str, subprocess.Popen] = {}
PROCESS_LOCK = threading.Lock()


# =========================================================
# 固定模型配置
# =========================================================

KNOWN_MODELS = {
    "old_direct7": {
        "label": "普通多变量LSTM",
        "model_file": "lstm_pm25_changchun_old_direct7.pt",
        "scaler_files": ["scaler_pm25_old_direct7.save"],
        "train_script": "train-old.py",
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
        "scaler_files": ["scaler_pm25_featured_direct7.save"],
        "train_script": "train.py",
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
        "train_script": "train-warning.py",
        "description": "面向污染等级预警优化的特征增强 LSTM 模型。",
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "seq_length": 45,
        "pred_days": 7,
        "input_size": 21,
    },
}


# =========================================================
# 通用工具
# =========================================================

def model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def safe_model_name(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"[^\w\u4e00-\u9fa5.-]+", "_", text)
    text = text.strip("._")
    return text or "pm25_lstm"


def safe_csv_filename(filename: str) -> str:
    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    name = Path(filename).name.strip()

    if not name:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    if not name.lower().endswith(".csv"):
        name += ".csv"

    if name in [".csv", "csv"]:
        raise HTTPException(status_code=400, detail="非法文件名")

    return name


def safe_dataset_path(filename: str) -> Path:
    name = safe_csv_filename(filename)
    path = DATASET_DIR / name

    resolved = path.resolve()
    root = DATASET_DIR.resolve()

    if not str(resolved).startswith(str(root)):
        raise HTTPException(status_code=400, detail="非法路径")

    return resolved


def read_csv_auto_encoding(file_path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb2312"]
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"CSV读取失败：{last_error}")


# =========================================================
# 数据集管理工具
# =========================================================

def get_dataset_info(file_path: Path) -> Dict[str, Any]:
    info = {
        "filename": file_path.name,
        "name": file_path.stem,
        "size": file_path.stat().st_size,
        "rows": 0,
        "columns": [],
        "last_date": None,
        "version": "数据版本：未知",
        "error": None,
    }

    try:
        df = read_csv_auto_encoding(file_path)
        df.columns = [str(col).strip() for col in df.columns]

        info["rows"] = int(len(df))
        info["columns"] = list(df.columns)

        date_col = None

        for col in df.columns:
            if str(col).strip().lower() in ["date", "日期", "time", "时间"]:
                date_col = col
                break

        if date_col:
            date_series = (
                df[date_col]
                .astype(str)
                .str.strip()
                .str.replace("/", "-", regex=False)
            )

            parsed_dates = pd.to_datetime(date_series, errors="coerce")
            parsed_dates = parsed_dates.dropna()

            if not parsed_dates.empty:
                last_date = parsed_dates.max().strftime("%Y-%m-%d")
                info["last_date"] = last_date
                info["version"] = f"数据版本：{last_date}"

    except Exception as exc:
        info["error"] = str(exc)

    return info


def prepare_training_dataset(job_id: str, filenames: List[str]) -> Path:
    if not filenames:
        raise HTTPException(status_code=400, detail="至少选择一个训练数据集")

    dataset_paths = []

    for filename in filenames:
        path = safe_dataset_path(filename)

        if not path.exists():
            raise HTTPException(status_code=404, detail=f"训练数据集不存在：{filename}")

        dataset_paths.append(path)

    if len(dataset_paths) == 1:
        return dataset_paths[0]

    frames = []

    for path in dataset_paths:
        df = read_csv_auto_encoding(path)
        df.columns = [str(col).strip() for col in df.columns]
        df["source_station_file"] = path.name
        frames.append(df)

    merged_df = pd.concat(frames, ignore_index=True)

    date_col = None

    for col in merged_df.columns:
        if str(col).strip().lower() in ["date", "日期", "time", "时间"]:
            date_col = col
            break

    if date_col:
        temp_date = (
            merged_df[date_col]
            .astype(str)
            .str.strip()
            .str.replace("/", "-", regex=False)
        )

        parsed_date = pd.to_datetime(temp_date, errors="coerce")
        merged_df["_sort_date"] = parsed_date
        merged_df = merged_df.sort_values("_sort_date").drop(columns=["_sort_date"])

    output_path = PROCESSED_DIR / f"{job_id}_merged.csv"
    merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return output_path


# =========================================================
# 模型注册表 / 权限 / 管理状态
# =========================================================

def load_model_access_state() -> Dict[str, bool]:
    if not MODEL_ACCESS_FILE.exists():
        default_state = {key: True for key in KNOWN_MODELS.keys()}
        save_model_access_state(default_state)
        return default_state

    try:
        with MODEL_ACCESS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            return data

        return {}

    except Exception:
        return {}


def save_model_access_state(state: Dict[str, bool]):
    with MODEL_ACCESS_FILE.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_trained_models() -> Dict[str, Dict[str, Any]]:
    if not TRAINED_MODEL_FILE.exists():
        return {}

    try:
        with TRAINED_MODEL_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            return data

        return {}

    except Exception:
        return {}


def save_trained_models(models: Dict[str, Dict[str, Any]]):
    with TRAINED_MODEL_FILE.open("w", encoding="utf-8") as f:
        json.dump(models, f, ensure_ascii=False, indent=2)


def load_model_manage_state() -> Dict[str, Any]:
    default_state = {
        "label_overrides": {},
        "hidden_models": [],
    }

    if not MODEL_MANAGE_FILE.exists():
        save_model_manage_state(default_state)
        return default_state

    try:
        with MODEL_MANAGE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return default_state

        data.setdefault("label_overrides", {})
        data.setdefault("hidden_models", [])

        return data

    except Exception:
        return default_state


def save_model_manage_state(state: Dict[str, Any]):
    state.setdefault("label_overrides", {})
    state.setdefault("hidden_models", [])

    with MODEL_MANAGE_FILE.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def get_all_model_configs(include_hidden: bool = False) -> Dict[str, Dict[str, Any]]:
    all_configs = dict(KNOWN_MODELS)
    trained_models = load_trained_models()

    for key, config in trained_models.items():
        all_configs[key] = config

    if include_hidden:
        return all_configs

    manage_state = load_model_manage_state()
    hidden = set(manage_state.get("hidden_models", []))

    return {
        key: value
        for key, value in all_configs.items()
        if key not in hidden
    }


def get_model_display_label(model_key: str, config: Dict[str, Any]) -> str:
    manage_state = load_model_manage_state()
    label_overrides = manage_state.get("label_overrides", {})

    return label_overrides.get(model_key) or config.get("label") or model_key


def resolve_existing_scaler_file(scaler_files: List[str]) -> Optional[str]:
    for filename in scaler_files:
        if (MODEL_DIR / filename).exists():
            return filename

    return None


def build_model_item(model_key: str, config: Dict[str, Any], enabled: bool) -> Dict[str, Any]:
    model_path = MODEL_DIR / config["model_file"]
    scaler_file = resolve_existing_scaler_file(config["scaler_files"])

    model_exists = model_path.exists()
    scaler_exists = scaler_file is not None

    is_custom = model_key.startswith("custom_")

    return {
        "key": model_key,
        "label": get_model_display_label(model_key, config),
        "description": config.get("description", ""),
        "model_file": config["model_file"],
        "scaler_file": scaler_file or config["scaler_files"][0],
        "scaler_files": config["scaler_files"],
        "train_script": config.get("train_script", ""),
        "enabled": bool(enabled),
        "model_exists": model_exists,
        "scaler_exists": scaler_exists,
        "available": bool(enabled and model_exists and scaler_exists),
        "is_custom": is_custom,
        "base_model_key": config.get("base_model_key"),
        "base_model_label": config.get("base_model_label"),
        "created_at": config.get("created_at"),
        "metrics": config.get("metrics", {}),
        "hidden_size": config.get("hidden_size"),
        "num_layers": config.get("num_layers"),
        "dropout": config.get("dropout"),
        "seq_length": config.get("seq_length"),
        "pred_days": config.get("pred_days"),
        "input_size": config.get("input_size"),
    }


# =========================================================
# 旧模型参数自动补全
# =========================================================

def infer_model_params_from_pt(model_path: Path) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    if not model_path.exists():
        return params

    try:
        import torch

        checkpoint = torch.load(model_path, map_location="cpu")

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            return params

        weight_ih_l0 = state_dict.get("lstm.weight_ih_l0")

        if weight_ih_l0 is not None:
            params["hidden_size"] = int(weight_ih_l0.shape[0] // 4)
            params["input_size"] = int(weight_ih_l0.shape[1])

        layer_indexes = []

        for key in state_dict.keys():
            if key.startswith("lstm.weight_ih_l"):
                try:
                    layer_index = int(key.replace("lstm.weight_ih_l", ""))
                    layer_indexes.append(layer_index)
                except Exception:
                    pass

        if layer_indexes:
            params["num_layers"] = int(max(layer_indexes) + 1)

        fc_out_weight = state_dict.get("fc.3.weight")

        if fc_out_weight is not None:
            params["pred_days"] = int(fc_out_weight.shape[0])

        input_size = params.get("input_size")

        if input_size == 6:
            params["feature_mode"] = "base"
        elif input_size == 21:
            params["feature_mode"] = "featured"
        else:
            params["feature_mode"] = "unknown"

    except Exception as exc:
        print(f"从模型权重推断参数失败：{model_path}，错误：{exc}", flush=True)
        return params

    return params


def backfill_trained_model_params(save: bool = True) -> Dict[str, Any]:
    trained_models = load_trained_models()

    if not trained_models:
        return {
            "success": True,
            "changed": False,
            "message": "没有自定义模型需要补全",
            "updated_models": [],
        }

    changed = False
    updated_models = []

    for model_key, meta in trained_models.items():
        if not isinstance(meta, dict):
            continue

        model_file = meta.get("model_file", "")

        if not model_file:
            continue

        model_path = MODEL_DIR / model_file
        pt_params = infer_model_params_from_pt(model_path)

        if not pt_params:
            continue

        filled_params = {}

        for name in ["hidden_size", "num_layers", "pred_days", "input_size"]:
            if name in pt_params:
                if name not in meta or meta.get(name) in [None, ""]:
                    meta[name] = pt_params[name]
                    filled_params[name] = pt_params[name]
                    changed = True

        input_size = pt_params.get("input_size")

        if input_size == 6:
            expected_base_model_key = "old_direct7"
            expected_base_model_label = "普通多变量LSTM"
        elif input_size == 21:
            old_base = meta.get("base_model_key", "")

            if old_base == "featured_warning_direct7":
                expected_base_model_key = "featured_warning_direct7"
                expected_base_model_label = "预警导向特征增强LSTM"
            else:
                expected_base_model_key = "featured_direct7"
                expected_base_model_label = "特征增强LSTM"
        else:
            expected_base_model_key = meta.get("base_model_key", "featured_direct7")
            expected_base_model_label = meta.get("base_model_label", "特征增强LSTM")

        if not meta.get("base_model_key"):
            meta["base_model_key"] = expected_base_model_key
            filled_params["base_model_key"] = expected_base_model_key
            changed = True

        if not meta.get("base_model_label"):
            meta["base_model_label"] = expected_base_model_label
            filled_params["base_model_label"] = expected_base_model_label
            changed = True

        if "seq_length" not in meta or meta.get("seq_length") in [None, ""]:
            if model_key.startswith("custom_"):
                meta["seq_length"] = 21
            else:
                meta["seq_length"] = 45

            filled_params["seq_length"] = meta["seq_length"]
            changed = True

        if "dropout" not in meta or meta.get("dropout") in [None, ""]:
            meta["dropout"] = 0.2
            filled_params["dropout"] = 0.2
            changed = True

        if filled_params:
            updated_models.append(
                {
                    "key": model_key,
                    "label": meta.get("label", model_key),
                    "model_file": meta.get("model_file", ""),
                    "filled_params": filled_params,
                    "pt_inferred_params": pt_params,
                }
            )

    if changed and save:
        save_trained_models(trained_models)

    return {
        "success": True,
        "changed": changed,
        "message": "旧模型参数补全完成" if changed else "没有发现缺失参数的旧模型",
        "updated_models": updated_models,
    }


def register_trained_model(req, metrics: Dict[str, Any]):
    base_config = KNOWN_MODELS.get(req.model_key)

    if not base_config:
        return

    safe_name = safe_model_name(req.model_name)

    model_file = f"{safe_name}.pt"
    scaler_file = f"scaler_{safe_name}.save"

    model_path = MODEL_DIR / model_file
    scaler_path = MODEL_DIR / scaler_file

    if not model_path.exists() or not scaler_path.exists():
        return

    custom_key = f"custom_{safe_name}"

    trained_models = load_trained_models()

    trained_models[custom_key] = {
        "label": req.model_name,
        "base_model_key": req.model_key,
        "base_model_label": base_config["label"],
        "model_file": model_file,
        "scaler_files": [scaler_file],
        "train_script": base_config["train_script"],
        "description": f"管理员训练生成模型，基础模型类型：{base_config['label']}",
        "enabled": True,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics or {},
        "hidden_size": int(req.hidden_size),
        "num_layers": int(req.num_layers),
        "dropout": float(req.dropout),
        "seq_length": int(req.seq_length),
        "pred_days": int(req.pred_days),
    }

    save_trained_models(trained_models)

    access_state = load_model_access_state()
    access_state[custom_key] = True
    save_model_access_state(access_state)

    manage_state = load_model_manage_state()
    hidden = set(manage_state.get("hidden_models", []))
    hidden.discard(custom_key)
    manage_state["hidden_models"] = list(hidden)
    save_model_manage_state(manage_state)

    backfill_trained_model_params(save=True)


# =========================================================
# 训练任务记录 / 日志解析
# =========================================================

def load_train_jobs() -> List[Dict[str, Any]]:
    if not TRAIN_JOB_FILE.exists():
        return []

    try:
        with TRAIN_JOB_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        return []

    except Exception:
        return []


def save_train_jobs(jobs: List[Dict[str, Any]]):
    with TRAIN_JOB_FILE.open("w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2)


def find_train_job(job_id: str) -> Optional[Dict[str, Any]]:
    jobs = load_train_jobs()

    for job in jobs:
        if job.get("job_id") == job_id:
            return job

    return None


def update_train_job(job_id: str, **kwargs):
    jobs = load_train_jobs()

    for job in jobs:
        if job.get("job_id") == job_id:
            job.update(kwargs)
            break

    save_train_jobs(jobs)


def parse_metrics_from_text(text: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    epoch_matches = re.findall(
        r"(?:epoch|Epoch|EPOCH|迭代|轮次|第)\s*[:：\[]?\s*(\d+)(?:\s*/\s*(\d+))?",
        text,
        flags=re.IGNORECASE,
    )

    if epoch_matches:
        last_epoch = epoch_matches[-1]

        try:
            metrics["epoch"] = int(last_epoch[0])
        except Exception:
            pass

        if len(last_epoch) > 1 and last_epoch[1]:
            try:
                metrics["total_epoch"] = int(last_epoch[1])
            except Exception:
                pass

    number = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"

    patterns = {
        "loss": rf"\bloss\b\s*[:=：]\s*{number}",
        "val_loss": rf"\bval[_\s-]*loss\b\s*[:=：]\s*{number}",
        "rmse": rf"(?<![_A-Za-z0-9])rmse\b\s*[:=：]\s*{number}",
        "mae": rf"(?<![_A-Za-z0-9])mae\b\s*[:=：]\s*{number}",
        "mape": rf"(?<![_A-Za-z0-9])mape\b\s*[:=：]\s*{number}",
        "r2": rf"(?:\br2\b|\br\^2\b|R²|决定系数)\s*[:=：]\s*{number}",
        "lr": rf"\blr\b\s*[:=：]\s*{number}",
        "test_rmse": rf"\btest[_\s-]*rmse\b\s*[:=：]\s*{number}",
        "test_mae": rf"\btest[_\s-]*mae\b\s*[:=：]\s*{number}",
        "test_r2": rf"\btest[_\s-]*r2\b\s*[:=：]\s*{number}",
        "baseline_rmse": rf"\bbaseline[_\s-]*rmse\b\s*[:=：]\s*{number}",
        "baseline_mae": rf"\bbaseline[_\s-]*mae\b\s*[:=：]\s*{number}",
        "best_val_rmse": rf"\bbest[_\s-]*val[_\s-]*rmse\b\s*[:=：]\s*{number}",
        "high_rmse": rf"\bhigh[_\s-]*rmse\b\s*[:=：]\s*{number}",
        "high_mae": rf"\bhigh[_\s-]*mae\b\s*[:=：]\s*{number}",
        "high_count": rf"\bhigh[_\s-]*count\b\s*[:=：]\s*(\d+)",
    }

    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, flags=re.IGNORECASE)

        if matches:
            try:
                if key == "high_count":
                    metrics[key] = int(matches[-1])
                else:
                    metrics[key] = round(float(matches[-1]), 6)
            except Exception:
                pass

    return metrics


def parse_score_summary(metrics: Dict[str, Any]) -> str:
    if not metrics:
        return "--"

    if "test_rmse" in metrics or "test_mae" in metrics or "test_r2" in metrics:
        parts = []

        if "test_rmse" in metrics:
            parts.append(f"Test RMSE={metrics['test_rmse']}")

        if "test_mae" in metrics:
            parts.append(f"Test MAE={metrics['test_mae']}")

        if "test_r2" in metrics:
            parts.append(f"Test R²={metrics['test_r2']}")

        return "，".join(parts) if parts else "--"

    parts = []

    if "rmse" in metrics:
        parts.append(f"RMSE={metrics['rmse']}")

    if "mae" in metrics:
        parts.append(f"MAE={metrics['mae']}")

    if "r2" in metrics:
        parts.append(f"R²={metrics['r2']}")

    if "val_loss" in metrics:
        parts.append(f"ValLoss={metrics['val_loss']}")

    if "loss" in metrics and "val_loss" not in metrics:
        parts.append(f"Loss={metrics['loss']}")

    return "，".join(parts) if parts else "--"


def get_log_text(log_file: Optional[str], tail_lines: int = 300) -> str:
    if not log_file:
        return ""

    path = Path(log_file)

    if not path.exists():
        return ""

    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        if tail_lines and tail_lines > 0:
            lines = lines[-tail_lines:]

        return "".join(lines)

    except Exception:
        return ""


def enrich_jobs_with_metrics(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []

    for job in jobs:
        item = dict(job)

        log_text = get_log_text(item.get("log_file"), tail_lines=10000)

        if log_text:
            metrics = parse_metrics_from_text(log_text)

            if metrics:
                item["metrics"] = metrics
                item["score_summary"] = parse_score_summary(metrics)

        if "metrics" not in item:
            item["metrics"] = {}

        if "score_summary" not in item:
            item["score_summary"] = parse_score_summary(item.get("metrics") or {})

        result.append(item)

    return result


def normalize_return_code(code: Optional[int]) -> Dict[str, Any]:
    if code is None:
        return {
            "raw": None,
            "signed": None,
            "hex": None,
        }

    signed = code

    if code > 2147483647:
        signed = code - 4294967296

    unsigned = code if code >= 0 else code + 4294967296

    return {
        "raw": code,
        "signed": signed,
        "hex": f"0x{unsigned:08X}",
    }


def training_artifacts_exist(model_key: str) -> bool:
    config = KNOWN_MODELS.get(model_key)

    if not config:
        return False

    model_path = MODEL_DIR / config["model_file"]
    scaler_ok = resolve_existing_scaler_file(config["scaler_files"]) is not None

    return model_path.exists() and scaler_ok


def custom_training_artifacts_exist(req) -> bool:
    safe_name = safe_model_name(req.model_name)

    model_path = MODEL_DIR / f"{safe_name}.pt"
    scaler_path = MODEL_DIR / f"scaler_{safe_name}.save"

    return model_path.exists() and scaler_path.exists()


def has_final_metrics(metrics: Dict[str, Any]) -> bool:
    return "test_rmse" in metrics or "test_mae" in metrics or "test_r2" in metrics


def terminate_process_by_job_id(job_id: str) -> bool:
    with PROCESS_LOCK:
        process = RUNNING_PROCESSES.get(job_id)

    if not process:
        return False

    if process.poll() is not None:
        with PROCESS_LOCK:
            RUNNING_PROCESSES.pop(job_id, None)
        return False

    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            process.terminate()

        try:
            process.wait(timeout=5)
        except Exception:
            process.kill()

        with PROCESS_LOCK:
            RUNNING_PROCESSES.pop(job_id, None)

        return True

    except Exception:
        return False


# =========================================================
# 请求模型
# =========================================================

class CreateDatasetRequest(BaseModel):
    filename: str
    columns: Optional[List[str]] = None


class RenameDatasetRequest(BaseModel):
    new_filename: str


class SaveDatasetContentRequest(BaseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]


class ModelEnableRequest(BaseModel):
    enabled: bool


class RenameModelRequest(BaseModel):
    new_label: str


class TrainRequest(BaseModel):
    model_name: str
    model_key: str

    dataset_filenames: Optional[List[str]] = None
    dataset_filename: Optional[str] = None

    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    seq_length: int = 45
    pred_days: int = 7

    remark: Optional[str] = ""


def get_selected_dataset_filenames(req: TrainRequest) -> List[str]:
    if req.dataset_filenames and len(req.dataset_filenames) > 0:
        return [safe_csv_filename(item) for item in req.dataset_filenames]

    if req.dataset_filename:
        return [safe_csv_filename(req.dataset_filename)]

    return []


# =========================================================
# 数据集接口
# =========================================================

@admin_router.get("/admin/datasets")
def list_datasets():
    files = sorted(DATASET_DIR.glob("*.csv"))
    datasets = [get_dataset_info(file_path) for file_path in files]

    return {
        "success": True,
        "datasets": datasets,
    }


@admin_router.post("/admin/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    filename = safe_csv_filename(file.filename)
    save_path = safe_dataset_path(filename)

    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    info = get_dataset_info(save_path)

    return {
        "success": True,
        "message": "CSV导入成功",
        "dataset": info,
    }


@admin_router.post("/admin/datasets/create")
def create_dataset(req: CreateDatasetRequest):
    filename = safe_csv_filename(req.filename)
    file_path = safe_dataset_path(filename)

    if file_path.exists():
        raise HTTPException(status_code=400, detail="该CSV文件已存在")

    columns = req.columns or ["date", "pm25", "pm10", "o3", "no2", "so2", "co"]

    df = pd.DataFrame(columns=columns)
    df.to_csv(file_path, index=False, encoding="utf-8-sig")

    info = get_dataset_info(file_path)

    return {
        "success": True,
        "message": "CSV新建成功",
        "dataset": info,
    }


@admin_router.put("/admin/datasets/{filename:path}/rename")
def rename_dataset(filename: str, req: RenameDatasetRequest):
    old_path = safe_dataset_path(filename)

    if not old_path.exists():
        raise HTTPException(status_code=404, detail="原CSV文件不存在")

    new_filename = safe_csv_filename(req.new_filename)
    new_path = safe_dataset_path(new_filename)

    if new_path.exists():
        raise HTTPException(status_code=400, detail="目标文件名已存在")

    old_path.rename(new_path)

    info = get_dataset_info(new_path)

    return {
        "success": True,
        "message": "重命名成功",
        "dataset": info,
    }


@admin_router.delete("/admin/datasets/{filename:path}")
def delete_dataset(filename: str):
    file_path = safe_dataset_path(filename)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="CSV文件不存在")

    file_path.unlink()

    return {
        "success": True,
        "message": "CSV删除成功",
    }


@admin_router.get("/admin/datasets/{filename:path}/preview")
def preview_dataset(filename: str, limit: int = 100):
    file_path = safe_dataset_path(filename)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="CSV文件不存在")

    try:
        df = read_csv_auto_encoding(file_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV读取失败：{exc}")

    df = df.head(limit)
    df = df.where(pd.notnull(df), "")

    columns = [str(col) for col in df.columns]
    rows = df.to_dict(orient="records")

    return {
        "success": True,
        "filename": file_path.name,
        "columns": columns,
        "rows": rows,
        "limit": limit,
    }


@admin_router.put("/admin/datasets/{filename:path}/content")
def save_dataset_content(filename: str, req: SaveDatasetContentRequest):
    file_path = safe_dataset_path(filename)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="CSV文件不存在")

    raw_columns = req.columns or []

    columns = []
    used = set()

    for index, col in enumerate(raw_columns):
        name = str(col).strip()

        if not name:
            name = f"column_{index + 1}"

        original_name = name
        suffix = 2

        while name in used:
            name = f"{original_name}_{suffix}"
            suffix += 1

        used.add(name)
        columns.append(name)

    if not columns:
        raise HTTPException(status_code=400, detail="CSV至少需要一列")

    clean_rows = []

    for row in req.rows or []:
        clean_row = {}

        for col in columns:
            value = row.get(col, "")

            if value is None:
                value = ""

            clean_row[col] = value

        clean_rows.append(clean_row)

    df = pd.DataFrame(clean_rows, columns=columns)

    try:
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"CSV保存失败：{exc}")

    info = get_dataset_info(file_path)

    return {
        "success": True,
        "message": "CSV保存成功",
        "dataset": info,
    }


@admin_router.post("/admin/upload_csv")
async def legacy_upload_csv(file: UploadFile = File(...)):
    return await upload_dataset(file)


# =========================================================
# 模型管理接口
# =========================================================

@admin_router.get("/admin/models")
def list_admin_models():
    backfill_trained_model_params(save=True)

    trained_models = load_trained_models()
    access_state = load_model_access_state()
    all_configs = get_all_model_configs(include_hidden=False)

    models = []

    for key, config in all_configs.items():
        if key in trained_models:
            enabled = bool(trained_models[key].get("enabled", True))
        else:
            enabled = bool(access_state.get(key, True))

        models.append(build_model_item(key, config, enabled))

    return {
        "success": True,
        "models": models,
    }


@admin_router.post("/admin/models/backfill-params")
def backfill_model_params_api():
    return backfill_trained_model_params(save=True)


@admin_router.put("/admin/models/{model_key}/enabled")
def update_model_enabled(model_key: str, req: ModelEnableRequest):
    all_configs = get_all_model_configs(include_hidden=True)

    if model_key not in all_configs:
        raise HTTPException(status_code=404, detail="模型不存在")

    trained_models = load_trained_models()

    if model_key in trained_models:
        trained_models[model_key]["enabled"] = bool(req.enabled)
        save_trained_models(trained_models)

        item = build_model_item(model_key, trained_models[model_key], bool(req.enabled))

        return {
            "success": True,
            "message": "模型权限已更新",
            "model": item,
        }

    state = load_model_access_state()
    state[model_key] = bool(req.enabled)
    save_model_access_state(state)

    item = build_model_item(model_key, all_configs[model_key], bool(req.enabled))

    return {
        "success": True,
        "message": "模型权限已更新",
        "model": item,
    }


@admin_router.put("/admin/models/{model_key}/rename")
def rename_model(model_key: str, req: RenameModelRequest):
    new_label = str(req.new_label or "").strip()

    if not new_label:
        raise HTTPException(status_code=400, detail="模型名称不能为空")

    all_configs = get_all_model_configs(include_hidden=True)

    if model_key not in all_configs:
        raise HTTPException(status_code=404, detail="模型不存在")

    trained_models = load_trained_models()

    if model_key in trained_models:
        trained_models[model_key]["label"] = new_label
        save_trained_models(trained_models)

        enabled = bool(trained_models[model_key].get("enabled", True))
        item = build_model_item(model_key, trained_models[model_key], enabled)

        return {
            "success": True,
            "message": "模型重命名成功",
            "model": item,
        }

    manage_state = load_model_manage_state()
    manage_state.setdefault("label_overrides", {})
    manage_state["label_overrides"][model_key] = new_label
    save_model_manage_state(manage_state)

    access_state = load_model_access_state()
    enabled = bool(access_state.get(model_key, True))
    item = build_model_item(model_key, all_configs[model_key], enabled)

    return {
        "success": True,
        "message": "模型重命名成功",
        "model": item,
    }


@admin_router.delete("/admin/models/{model_key}")
def delete_model(model_key: str, delete_files: bool = Query(True)):
    all_configs = get_all_model_configs(include_hidden=True)

    if model_key not in all_configs:
        raise HTTPException(status_code=404, detail="模型不存在")

    trained_models = load_trained_models()
    access_state = load_model_access_state()
    manage_state = load_model_manage_state()

    if model_key in trained_models:
        config = trained_models[model_key]

        if delete_files:
            model_path = MODEL_DIR / config.get("model_file", "")

            if model_path.exists():
                try:
                    model_path.unlink()
                except Exception:
                    pass

            for scaler_file in config.get("scaler_files", []):
                scaler_path = MODEL_DIR / scaler_file

                if scaler_path.exists():
                    try:
                        scaler_path.unlink()
                    except Exception:
                        pass

        trained_models.pop(model_key, None)
        save_trained_models(trained_models)

        access_state.pop(model_key, None)
        save_model_access_state(access_state)

        manage_state.setdefault("label_overrides", {})
        manage_state["label_overrides"].pop(model_key, None)

        hidden = set(manage_state.get("hidden_models", []))
        hidden.discard(model_key)
        manage_state["hidden_models"] = list(hidden)
        save_model_manage_state(manage_state)

        return {
            "success": True,
            "message": "自定义模型已删除",
            "model_key": model_key,
        }

    hidden = set(manage_state.get("hidden_models", []))
    hidden.add(model_key)
    manage_state["hidden_models"] = list(hidden)

    manage_state.setdefault("label_overrides", {})
    manage_state["label_overrides"].pop(model_key, None)

    save_model_manage_state(manage_state)

    access_state[model_key] = False
    save_model_access_state(access_state)

    return {
        "success": True,
        "message": "内置模型已从模型管理中隐藏",
        "model_key": model_key,
    }


@admin_router.get("/models")
def list_user_models():
    backfill_trained_model_params(save=True)

    trained_models = load_trained_models()
    access_state = load_model_access_state()
    all_configs = get_all_model_configs(include_hidden=False)

    all_models = []

    for key, config in all_configs.items():
        if key in trained_models:
            enabled = bool(trained_models[key].get("enabled", True))
        else:
            enabled = bool(access_state.get(key, True))

        all_models.append(build_model_item(key, config, enabled))

    user_models = [
        item for item in all_models
        if item["enabled"] and item["model_exists"] and item["scaler_exists"]
    ]

    default_model = user_models[0]["key"] if user_models else None

    return {
        "success": True,
        "default_model": default_model,
        "models": user_models,
    }


# =========================================================
# 模型训练接口
# =========================================================

def run_training_job(job: Dict[str, Any], req: TrainRequest):
    job_id = job["job_id"]

    update_train_job(
        job_id,
        status="running",
        started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    config = KNOWN_MODELS.get(req.model_key)

    if not config:
        update_train_job(
            job_id,
            status="failed",
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            error="未知模型类型",
        )
        return

    script_path = MODEL_DIR / config["train_script"]

    if not script_path.exists():
        update_train_job(
            job_id,
            status="failed",
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            error=f"训练脚本不存在：{script_path}",
        )
        return

    selected_filenames = get_selected_dataset_filenames(req)

    try:
        training_dataset_path = prepare_training_dataset(job_id, selected_filenames)
    except Exception as exc:
        update_train_job(
            job_id,
            status="failed",
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            error=str(exc),
        )
        return

    env = os.environ.copy()

    env.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "PM25_MODEL_NAME": req.model_name,
            "PM25_MODEL_KEY": req.model_key,
            "PM25_DATASET": str(training_dataset_path),
            "PM25_DATASETS": json.dumps(
                [str(safe_dataset_path(name)) for name in selected_filenames],
                ensure_ascii=False,
            ),
            "PM25_DATASET_FILENAMES": json.dumps(selected_filenames, ensure_ascii=False),
            "PM25_EPOCHS": str(req.epochs),
            "PM25_BATCH_SIZE": str(req.batch_size),
            "PM25_LEARNING_RATE": str(req.learning_rate),
            "PM25_HIDDEN_SIZE": str(req.hidden_size),
            "PM25_NUM_LAYERS": str(req.num_layers),
            "PM25_DROPOUT": str(req.dropout),
            "PM25_SEQ_LENGTH": str(req.seq_length),
            "PM25_PRED_DAYS": str(req.pred_days),
            "PM25_TREND_LOSS_WEIGHT": "0.35",
            "PM25_HIGH_WEIGHT_FACTOR": "3.0",
        }
    )

    log_file = LOG_DIR / f"train_{job_id}.log"

    update_train_job(job_id, log_file=str(log_file))

    cmd = [
        sys.executable,
        "-u",
        str(script_path),
    ]

    try:
        with log_file.open("w", encoding="utf-8", errors="replace") as f:
            f.write("训练任务启动\n")
            f.write(f"任务ID：{job_id}\n")
            f.write(f"命令：{' '.join(cmd)}\n")
            f.write(f"模型名称：{req.model_name}\n")
            f.write(f"模型类型：{req.model_key}\n")
            f.write(f"训练数据集数量：{len(selected_filenames)}\n")
            f.write(f"训练数据集：{', '.join(selected_filenames)}\n")
            f.write(f"实际传入训练脚本的数据文件：{training_dataset_path}\n")
            f.write(f"训练参数：{model_to_dict(req)}\n")
            f.write("趋势损失权重：0.35\n")
            f.write("高污染权重系数：3.0\n")
            f.write("=" * 80 + "\n\n")
            f.flush()

            process = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            with PROCESS_LOCK:
                RUNNING_PROCESSES[job_id] = process

            full_log_parts = []

            if process.stdout:
                for line in process.stdout:
                    f.write(line)
                    f.flush()

                    full_log_parts.append(line)

                    recent_text = "".join(full_log_parts[-100:])
                    metrics = parse_metrics_from_text(recent_text)

                    if metrics:
                        update_train_job(
                            job_id,
                            metrics=metrics,
                            score_summary=parse_score_summary(metrics),
                        )

            process.wait()
            return_code = process.returncode

            with PROCESS_LOCK:
                RUNNING_PROCESSES.pop(job_id, None)

        final_log = get_log_text(str(log_file), tail_lines=20000)
        final_metrics = parse_metrics_from_text(final_log)
        final_score_summary = parse_score_summary(final_metrics)
        return_info = normalize_return_code(return_code)

        current_job = find_train_job(job_id)

        if current_job and current_job.get("status") == "terminated":
            update_train_job(
                job_id,
                finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                log_file=str(log_file),
                metrics=final_metrics,
                score_summary=final_score_summary,
                message="任务已被管理员终止",
            )
            return

        success_by_artifacts = (
            has_final_metrics(final_metrics)
            and (
                custom_training_artifacts_exist(req)
                or training_artifacts_exist(req.model_key)
            )
        )

        if return_code == 0 or success_by_artifacts:
            register_trained_model(req, final_metrics)

            update_train_job(
                job_id,
                status="finished",
                finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                log_file=str(log_file),
                metrics=final_metrics,
                score_summary=final_score_summary,
                return_code=return_info,
                message="训练完成",
                error=None,
            )
        else:
            update_train_job(
                job_id,
                status="failed",
                finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                log_file=str(log_file),
                metrics=final_metrics,
                score_summary=final_score_summary,
                return_code=return_info,
                error=f"训练脚本返回错误码：{return_info['raw']} ({return_info['hex']})",
            )

    except Exception as exc:
        with PROCESS_LOCK:
            RUNNING_PROCESSES.pop(job_id, None)

        update_train_job(
            job_id,
            status="failed",
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            log_file=str(log_file),
            error=str(exc),
        )


@admin_router.post("/admin/train")
def start_training(req: TrainRequest):
    if req.model_key not in KNOWN_MODELS:
        raise HTTPException(status_code=400, detail="未知模型类型")

    if not req.model_name.strip():
        raise HTTPException(status_code=400, detail="模型名称不能为空")

    selected_filenames = get_selected_dataset_filenames(req)

    if not selected_filenames:
        raise HTTPException(status_code=400, detail="请至少选择一个训练数据集")

    for filename in selected_filenames:
        dataset_path = safe_dataset_path(filename)

        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail=f"训练数据集不存在：{filename}")

    job_id = uuid.uuid4().hex[:12]

    job = {
        "job_id": job_id,
        "model_name": req.model_name,
        "model_key": req.model_key,
        "model_label": KNOWN_MODELS[req.model_key]["label"],
        "dataset_filenames": selected_filenames,
        "dataset_filename": selected_filenames[0] if selected_filenames else "",
        "dataset_count": len(selected_filenames),
        "status": "pending",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "started_at": None,
        "finished_at": None,
        "params": {
            "epochs": req.epochs,
            "batch_size": req.batch_size,
            "learning_rate": req.learning_rate,
            "hidden_size": req.hidden_size,
            "num_layers": req.num_layers,
            "dropout": req.dropout,
            "seq_length": req.seq_length,
            "pred_days": req.pred_days,
            "trend_loss_weight": 0.35,
            "high_weight_factor": 3.0,
        },
        "remark": req.remark or "",
        "log_file": None,
        "metrics": {},
        "score_summary": "--",
        "return_code": None,
        "error": None,
        "message": "",
    }

    jobs = load_train_jobs()
    jobs.insert(0, job)
    save_train_jobs(jobs)

    thread = threading.Thread(
        target=run_training_job,
        args=(job, req),
        daemon=True,
    )
    thread.start()

    return {
        "success": True,
        "message": "训练任务已提交",
        "job": job,
    }


@admin_router.get("/admin/train/jobs")
def list_train_jobs():
    jobs = load_train_jobs()
    jobs = enrich_jobs_with_metrics(jobs)

    return {
        "success": True,
        "jobs": jobs,
    }


@admin_router.get("/admin/train/jobs/{job_id}/log")
def get_train_job_log(job_id: str, tail_lines: int = 300):
    jobs = load_train_jobs()

    target_job = None

    for job in jobs:
        if job.get("job_id") == job_id:
            target_job = job
            break

    if not target_job:
        raise HTTPException(status_code=404, detail="训练任务不存在")

    log_text = get_log_text(target_job.get("log_file"), tail_lines=tail_lines)
    metrics = parse_metrics_from_text(log_text)

    return {
        "success": True,
        "job_id": job_id,
        "status": target_job.get("status"),
        "log": log_text,
        "metrics": metrics,
        "score_summary": parse_score_summary(metrics),
    }


@admin_router.post("/admin/train/jobs/{job_id}/terminate")
def terminate_train_job(job_id: str):
    job = find_train_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="训练任务不存在")

    if job.get("status") not in ["pending", "running"]:
        return {
            "success": False,
            "message": "该任务当前状态不可终止",
            "job": job,
        }

    killed = terminate_process_by_job_id(job_id)

    update_train_job(
        job_id,
        status="terminated",
        finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        error=None,
        message="任务已被管理员终止" if killed else "任务已标记为终止；未找到活动进程，可能任务已结束或服务已重启。",
    )

    return {
        "success": True,
        "message": "终止任务请求已执行",
        "killed": killed,
        "job_id": job_id,
    }


@admin_router.delete("/admin/train/jobs/{job_id}")
def delete_train_job(job_id: str, delete_log: bool = Query(False)):
    jobs = load_train_jobs()

    target_job = None
    new_jobs = []

    for job in jobs:
        if job.get("job_id") == job_id:
            target_job = job
        else:
            new_jobs.append(job)

    if not target_job:
        raise HTTPException(status_code=404, detail="训练任务不存在")

    if target_job.get("status") in ["pending", "running"]:
        terminate_process_by_job_id(job_id)

    if delete_log and target_job.get("log_file"):
        log_path = Path(target_job["log_file"])

        if log_path.exists():
            try:
                log_path.unlink()
            except Exception:
                pass

    save_train_jobs(new_jobs)

    return {
        "success": True,
        "message": "训练任务记录已删除",
        "job_id": job_id,
    }