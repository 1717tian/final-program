import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Preprocessor:
    """数据预处理模块：归一化"""

    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit_transform(self, X):
        # X 必须是二维 (samples, features)
        return self.scaler.fit_transform(X)

    def transform(self, X):
        return self.scaler.transform(X)

def create_sequences(data, seq_length=3):
    """
    构建滑动窗口时间序列
    data: np.array, shape=(n_samples, n_features)
    返回:
        X_seq: (n_samples - seq_length, seq_length, n_features)
        y_seq: (n_samples - seq_length,)
    """
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i:i+seq_length])
        y_seq.append(data[i+seq_length][0])  # PM2.5 为预测目标
    return np.array(X_seq), np.array(y_seq)