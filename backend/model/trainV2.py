# train.py
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -----------------------------
# 配置路径和参数
# -----------------------------
DATA_DIR = r'D:\final program\pythonProject12\data\raw\pm25_sites'
MODEL_DIR = r'D:\final program\pythonProject12\backend\model'
os.makedirs(MODEL_DIR, exist_ok=True)

SEQ_LENGTH = 14
HIDDEN_SIZE = 128   # 可调
NUM_EPOCHS = 340    # 可调
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PM25LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128):  # 与train.py一致
        super(PM25LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.3)
        self.norm1 = nn.LayerNorm(hidden_size*2)
        self.lstm2 = nn.LSTM(hidden_size*2, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size//2, batch_first=True)
        self.dropout3 = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size//2, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out = self.norm1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = self.norm2(out)
        out, _ = self.lstm3(out)
        out = self.dropout3(out[:, -1, :])
        out = self.fc(out)
        return out
# -----------------------------
# 构建滑动窗口
# -----------------------------
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# -----------------------------
# 评价函数
# -----------------------------
def evaluate_rmse_mae(rmse, mae):
    if rmse < 50 and mae < 50:
        return "好"
    elif rmse < 150 and mae < 150:
        return "中"
    else:
        return "差"

# -----------------------------
# 数据准备
# -----------------------------
def load_data():
    csv_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()
        # 只保留 PM2.5 列·
        if 'pm25' not in df.columns:
            continue
        # 填充缺失值
        df['pm25'] = pd.to_numeric(df['pm25'], errors='coerce')
        df['pm25'] = df['pm25'].ffill().fillna(0)
        dfs.append(df)
    return dfs

# -----------------------------
# 训练函数
# -----------------------------
def train():
    dfs = load_data()
    if len(dfs) == 0:
        print("没有有效 CSV 数据！")
        return

    # 合并所有站点数据进行训练
    pm25_values = np.concatenate([df['pm25'].values.reshape(-1,1) for df in dfs], axis=0)

    # 归一化
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    pm25_scaled = scaler_X.fit_transform(pm25_values)
    y_scaled = scaler_y.fit_transform(pm25_values)

    # 构建训练序列
    X_seq, y_seq = create_sequences(pm25_scaled, SEQ_LENGTH)
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32).to(DEVICE)

    X_tensor = X_tensor.reshape(-1, SEQ_LENGTH, 1)
    y_tensor = y_tensor.reshape(-1, 1)

    # 模型
    model = PM25LSTM(input_size=1, hidden_size=HIDDEN_SIZE).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练
    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        # 计算评价
        with torch.no_grad():
            y_pred = outputs.cpu().numpy()
            y_true = y_tensor.cpu().numpy()
            y_pred_inv = scaler_y.inverse_transform(y_pred)
            y_true_inv = scaler_y.inverse_transform(y_true)
            rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
            mae = mean_absolute_error(y_true_inv, y_pred_inv)
            rating = evaluate_rmse_mae(rmse, mae)

        print(f"Epoch {epoch}/{NUM_EPOCHS} - Loss: {loss.item():.6f} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, 评价: {rating}")

    # 保存模型和归一化器
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'lstm_pm25_best.pt'))
    joblib.dump(scaler_X, os.path.join(MODEL_DIR, 'scaler_X.save'))
    joblib.dump(scaler_y, os.path.join(MODEL_DIR, 'scaler_y.save'))
    print("训练完成，模型已保存。")

# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    train()