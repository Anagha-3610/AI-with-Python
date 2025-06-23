#MSE
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

z = 0.2
y_true = np.array([0.5])
y_pred = np.array([sigmoid(z)])
print("MSE Loss:", mse_loss(y_true,y_pred))


# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Binary cross-entropy loss function
def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)  # Avoid log(0). Ensures y-pred is never exactly 0 or 1.
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Example input
z = 2
y_true = np.array([1])
y_pred = np.array([sigmoid(z)])

# Output the loss
print("Binary Cross Entropy Loss:", binary_cross_entropy(y_true,y_pred))