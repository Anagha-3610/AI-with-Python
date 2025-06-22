import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

epochs = list(range(1, 11))
loss = [0.9, 0.7, 0.5, 0.4, 0.35, 0.3, 0.25, 0.22, 0.21, 0.2]

plt.plot(epochs, loss, marker='o')
plt.title("Model Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.show()