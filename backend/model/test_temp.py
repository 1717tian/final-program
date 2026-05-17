# backend/model/test_temp.py
import os
import pandas as pd
import torch
import joblib
from backend.model.predict import predict_pm25

DATA_PATH = r'D:\final program\pythonProject12\data\raw\changchun_pm25.csv'
MODEL_DIR = 'backend/model/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

df = pd.read_csv(DATA_PATH, skipinitialspace=True)
indices, y_true, y_pred = predict_pm25(df)

# -----------------------------
# 计算指标
# -----------------------------
from sklearn.metrics import mean_squared_error, mean_absolute_error
rmse = mean_squared_error(y_true, y_pred, squared=False)
mae = mean_absolute_error(y_true, y_pred)
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# -----------------------------
# 可视化
# -----------------------------
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot(indices, y_true, label='真实值')
plt.plot(indices, y_pred, label='预测值')
plt.xlabel('样本索引')
plt.ylabel('PM2.5')
plt.title('PM2.5 预测对比')
plt.legend()
plt.show()