import numpy as np

# Inputs and weights
inputs = np.array([0.5, 0.3])
weights = np.array([0.4, 0.7])
bias = 0.1

# Linear combination
z = np.dot(inputs, weights) + bias
print("Z (weighted sum):",z)