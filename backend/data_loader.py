import pandas as pd
import numpy as np

class DataLoader:
    """读取空气质量数据"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path, skipinitialspace=True)

        # 转 datetime 并排序
        self.df['date'] = pd.to_datetime(self.df['date'], format='%Y/%m/%d')
        self.df.sort_values('date', inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # 缺失值填充
        self.df.ffill(inplace=True)  # 前向填充
        self.df.fillna(0, inplace=True)  # 首行缺失填0

        return self.df

    def get_features_labels(self):
        features = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
        labels = self.df['pm25'].values
        return self.df[features].values.astype(np.float32), labels.astype(np.float32)