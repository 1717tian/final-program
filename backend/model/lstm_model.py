import torch
import torch.nn as nn

class PM25LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=7):
        super(PM25LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out