import numpy as np
# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
z=0.51
print("Sigmoid Output:",sigmoid(z))


#b. ReLU (Rectified Linear Unit)
def relu(x):
    return np.maximum(0, x)
z=0.51
print("ReLU Output:", relu(z))


#c. Tanh
def tanh(x):
    return np.tanh(x)
z=0.51
print("Tanh Output:",tanh(z))