import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------
# 配置路径
# -----------------------------
MODEL_DIR = r'D:\final program\pythonProject12\backend\model'
MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_pm25_best-V3.pt')
SCALER_X_PATH = os.path.join(MODEL_DIR, 'scaler_X.save')
SCALER_Y_PATH = os.path.join(MODEL_DIR, 'scaler_y.save')
DATA_DIR = r'D:\final program\pythonProject12\data\raw\pm25_sites'

SEQ_LENGTH = 14
HIST_DAYS_SHORT = 14
FUTURE_DAYS_SHORT = 7
FUTURE_DAYS_LONG = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LSTM 模型
# -----------------------------
class PM25LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128):
        super(PM25LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.3)
        self.norm1 = nn.LayerNorm(hidden_size*2)
        self.lstm2 = nn.LSTM(hidden_size*2, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size//2, batch_first=True)
        self.dropout3 = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size//2,1)

    def forward(self,x):
        out,_ = self.lstm1(x)
        out = self.dropout1(out)
        out = self.norm1(out)
        out,_ = self.lstm2(out)
        out = self.dropout2(out)
        out = self.norm2(out)
        out,_ = self.lstm3(out)
        out = self.dropout3(out[:,-1,:])
        out = self.fc(out)
        return out

# -----------------------------
# 加载模型和归一化器
# -----------------------------
scaler_X = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)

model = PM25LSTM(input_size=1, hidden_size=128).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE))
model.eval()

# -----------------------------
# 滑动窗口
# -----------------------------
def create_sequences(data, seq_length):
    X_seq = []
    for i in range(len(data)-seq_length):
        X_seq.append(data[i:i+seq_length])
    return np.array(X_seq)

# -----------------------------
# 模型评价
# -----------------------------
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae

# -----------------------------
# 长期预测
# -----------------------------
def predict_long_term(df, future_days=FUTURE_DAYS_LONG):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    if 'pm25' not in df.columns:
        raise ValueError("CSV 文件缺少 'pm25' 列")
    df['pm25'] = pd.to_numeric(df['pm25'], errors='coerce').ffill().fillna(0)
    pm25_vals = df['pm25'].values.astype(np.float32).reshape(-1,1)

    # 历史预测
    X_scaled = scaler_X.transform(pm25_vals)
    X_seq = create_sequences(X_scaled, SEQ_LENGTH)
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()

    # 未来预测
    last_seq = X_scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH,1)
    future_pred = []
    seq = last_seq.copy()
    for _ in range(future_days):
        x_tensor = torch.tensor(seq,dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            y_scaled = model(x_tensor).cpu().numpy()
        y_next = scaler_y.inverse_transform(y_scaled.reshape(-1,1)).flatten()[0]
        future_pred.append(y_next)
        next_scaled = scaler_X.transform(np.array([[y_next]],dtype=np.float32))
        seq = np.concatenate([seq[:,1:,:], next_scaled.reshape(1,1,1)], axis=1)

    rmse, mae = evaluate_model(pm25_vals[SEQ_LENGTH:].flatten(), y_pred)
    return y_pred, future_pred, rmse, mae

# -----------------------------
# 短期预测
# -----------------------------
def predict_short_term(df, hist_days=HIST_DAYS_SHORT, future_days=FUTURE_DAYS_SHORT):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    if 'pm25' not in df.columns:
        raise ValueError("CSV 文件缺少 'pm25' 列")
    df['pm25'] = pd.to_numeric(df['pm25'], errors='coerce').ffill().fillna(0)
    pm25_vals = df['pm25'].values.astype(np.float32).reshape(-1,1)

    hist_pred = []
    for i in range(hist_days):
        end_idx = len(pm25_vals) - hist_days + i
        start_idx = end_idx - SEQ_LENGTH
        if start_idx < 0:
            pad_len = -start_idx
            seq_input = np.vstack([np.repeat(pm25_vals[0:1], pad_len, axis=0), pm25_vals[0:end_idx]])
        else:
            seq_input = pm25_vals[start_idx:end_idx]
        assert seq_input.shape[0] == SEQ_LENGTH
        seq_scaled = scaler_X.transform(seq_input).reshape(1,SEQ_LENGTH,1)
        seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            y_scaled = model(seq_tensor).cpu().numpy()
        y_pred_val = scaler_y.inverse_transform(y_scaled.reshape(-1,1)).flatten()[0]
        hist_pred.append(y_pred_val)

    # 未来预测
    last_seq = scaler_X.transform(pm25_vals[-SEQ_LENGTH:]).reshape(1,SEQ_LENGTH,1)
    future_pred = []
    seq = last_seq.copy()
    for _ in range(future_days):
        x_tensor = torch.tensor(seq,dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            y_scaled = model(x_tensor).cpu().numpy()
        y_next = scaler_y.inverse_transform(y_scaled.reshape(-1,1)).flatten()[0]
        future_pred.append(y_next)
        next_scaled = scaler_X.transform(np.array([[y_next]],dtype=np.float32))
        seq = np.concatenate([seq[:,1:,:], next_scaled.reshape(1,1,1)], axis=1)

    return np.array(hist_pred), np.array(future_pred)

# -----------------------------
# 绘制两张图
# -----------------------------
def plot_two_axes(ax1, ax2, df, long_pred, long_future, short_hist, short_future, title):
    df_sorted = df.copy()
    df_sorted.columns = df_sorted.columns.str.strip().str.lower()
    if 'pm25' not in df_sorted.columns or 'date' not in df_sorted.columns:
        print(f"{title} 缺少 'pm25' 或 'date' 列")
        return
    df_sorted['date'] = pd.to_datetime(df_sorted['date'], errors='coerce')
    df_sorted = df_sorted.sort_values('date').reset_index(drop=True)
    pm25_vals = pd.to_numeric(df_sorted['pm25'], errors='coerce').ffill().fillna(0)

    # 长期
    ax1.clear()
    ax1.plot(df_sorted['date'], pm25_vals, label='历史真实', color='green', linewidth=2)
    start_idx = len(df_sorted)-len(long_pred)
    hist_pred_full = np.full(len(df_sorted), np.nan)
    if start_idx >=0:
        hist_pred_full[start_idx:] = long_pred
    else:
        hist_pred_full = long_pred[-len(df_sorted):]
    ax1.plot(df_sorted['date'], hist_pred_full, label='历史预测', color='blue', linestyle='--', linewidth=2)
    last_date = df_sorted['date'].iloc[-1]
    future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(len(long_future))]
    ax1.plot(future_dates, long_future, label='未来预测', color='red', linestyle='-.', linewidth=2)
    ax1.set_title(title+' - 长期', fontsize=14)
    ax1.set_ylabel('PM2.5')
    ax1.grid(True)
    ax1.legend(fontsize=10)

    # 短期
    ax2.clear()
    hist_dates = df_sorted['date'].iloc[-HIST_DAYS_SHORT:]
    ax2.plot(hist_dates, df_sorted['pm25'].iloc[-HIST_DAYS_SHORT:], label='历史真实', color='green', linewidth=2)
    ax2.plot(hist_dates, short_hist, label='历史预测', color='blue', linestyle='--', linewidth=2)
    future_dates = [hist_dates.iloc[-1] + pd.Timedelta(days=i+1) for i in range(len(short_future))]
    ax2.plot(future_dates, short_future, label='未来预测', color='red', linestyle='-.', linewidth=2)
    ax2.set_title(title+' - 短期', fontsize=14)
    ax2.set_ylabel('PM2.5')
    ax2.grid(True)
    ax2.legend(fontsize=10)

# -----------------------------
# GUI
# -----------------------------
class PM25App:
    def __init__(self, root, csv_files):
        self.root = root
        self.csv_files = csv_files
        self.dfs = [pd.read_csv(f) for f in csv_files]
        self.idx = 0

        self.fig, (self.ax1, self.ax2) = plt.subplots(2,1,figsize=(12,8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill='both',expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, root)
        toolbar.update()
        self.canvas._tkcanvas.pack(fill='both',expand=True)

        self.site_var = tk.StringVar()
        self.site_menu = ttk.Combobox(root, textvariable=self.site_var, state='readonly', width=40)
        self.site_menu['values'] = [os.path.basename(f) for f in csv_files]
        self.site_menu.current(0)
        self.site_menu.bind("<<ComboboxSelected>>", self.on_site_change)
        self.site_menu.pack(pady=5)

        self.info_label = tk.Label(root,text='',font=('SimHei',12))
        self.info_label.pack(pady=5)

        self.update_plot()

    def on_site_change(self,event=None):
        self.idx = self.site_menu.current()
        self.update_plot()

    def update_plot(self):
        df = self.dfs[self.idx]
        long_pred, long_future, rmse, mae = predict_long_term(df)
        short_hist, short_future = predict_short_term(df)
        plot_two_axes(self.ax1, self.ax2, df, long_pred, long_future, short_hist, short_future, os.path.basename(self.csv_files[self.idx]))
        self.info_label.config(text=f"RMSE={rmse:.2f}  MAE={mae:.2f}")
        self.canvas.draw()

# -----------------------------
# 主程序
# -----------------------------
if __name__=="__main__":
    csv_files = glob.glob(os.path.join(DATA_DIR,'*.csv'))
    if len(csv_files)==0:
        print("没有找到 CSV 文件！")
    else:
        root = tk.Tk()
        root.title("PM2.5 模型预测评估系统")
        app = PM25App(root,csv_files)
        root.mainloop()