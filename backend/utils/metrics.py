# backend/utils/metrics.py
import numpy as np

def rmse(y_true, y_pred):
    """
    均方根误差 (Root Mean Squared Error)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    """
    平均绝对误差 (Mean Absolute Error)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))