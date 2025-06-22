#Histogram
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
data = np.random.normal(size=1000)
sns.histplot(data, kde=True, color='green')
plt.title("Histogram with KDE")
plt.show()