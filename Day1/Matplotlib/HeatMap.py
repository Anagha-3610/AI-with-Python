import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

corr = df.drop('species', axis=1).corr() #select_dtypes(include=[np.number])
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix Heatmap")
plt.show()