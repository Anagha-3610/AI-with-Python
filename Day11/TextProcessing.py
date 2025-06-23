#Example 1: Text Processing with RNN & LSTM
import torch
import torch.nn as nn

# Example input: batch of 2 sequences, each with 5 time steps, each step with 10 features
input_data = torch.randn(5, 2, 6)  # (seq_len, batch_size, input_size)

# RNN
rnn = nn.RNN(input_size=6, hidden_size=30, num_layers=1)
rnn_out, hidden = rnn(input_data)

# LSTM
lstm = nn.LSTM(input_size=6, hidden_size=20, num_layers=1)
lstm_out, (h_n, c_n) = lstm(input_data)

print("RNN output shape:", rnn_out.shape)
print("hidden output shape:", hidden.shape)
print("LSTM output shape:", lstm_out.shape)
print("h_n output shape:", h_n.shape)
print("c_n output shape:", c_n.shape)