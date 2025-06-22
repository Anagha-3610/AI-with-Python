#SineWave
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# Visualization Examples
# Line Plot (Matplotlib)
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("Sine Wave")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid(True)
plt.show()