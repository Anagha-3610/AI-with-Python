import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
sns.pairplot(df, hue='species')
plt.show()