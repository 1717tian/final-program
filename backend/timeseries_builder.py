# backend/timeseries_builder.py
import numpy as np


def create_sequences(data, seq_length, target_index=0, return_index=False):
    """
    构建时间序列滑动窗口数据。

    参数:
        data (np.ndarray): 原始特征数组，形状 (样本数, 特征数)
        seq_length (int): 滑动窗口长度
        target_index (int): 目标列索引（默认 0）
        return_index (bool): 是否返回目标值对应的索引

    返回:
        X_seq (np.ndarray): 特征序列, 形状 (样本数-序列长度, seq_length, 特征数)
        y_seq (np.ndarray): 目标值序列, 形状 (样本数-序列长度,)
        indices (np.ndarray, optional): 如果 return_index=True，则返回目标值索引
    """
    X_seq, y_seq, indices = [], [], []

    for i in range(len(data) - seq_length):
        seq_x = data[i:i + seq_length]
        seq_y = data[i + seq_length, target_index]
        X_seq.append(seq_x)
        y_seq.append(seq_y)
        if return_index:
            indices.append(i + seq_length)

    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)

    if return_index:
        return X_seq, y_seq, np.array(indices, dtype=int)
    return X_seq, y_seq