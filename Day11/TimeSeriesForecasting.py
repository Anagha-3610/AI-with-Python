#Example 2:  Time Series Forecasting with LSTM
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Sample synthetic sine wave time series
time_steps = np.linspace(0, 100, 200)
data = np.sin(time_steps)

plt.plot(time_steps, data)


# Prepare sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)

seq_length = 10
X, y = create_sequences(data, seq_length)
X = X.unsqueeze(-1)  # (batch, seq_len, input_size)
print(X.shape)
print(y.shape)
# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=100, num_layers=1)
        self.linear = nn.Linear(100, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.linear(last_out)

model = LSTMModel()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    model.train()
    output = model(X)
    loss = loss_fn(output.squeeze(), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")