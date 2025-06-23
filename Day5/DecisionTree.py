# Import necessary libraries
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt

# Sample dataset
data = {
    'Age': [25, 30, 45, 35, 22, 55, 40, 60],
    'Income': [40000, 60000, 80000, 120000, 30000, 70000, 100000, 90000],
    'Buys_Computer': ['No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Define features (X) and target (y)
X = df[['Age', 'Income']]
y = df['Buys_Computer']

# Create Decision Tree Classifier
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X, y)

# Make a prediction
sample = [[67, 70000]]  # Age 28, Income 50000
prediction = model.predict(sample)
print("Prediction for sample [28, 70000]:", prediction[0])

# Visualize the decision tree
plt.figure(figsize=(5, 6))
tree.plot_tree(model, feature_names=['Age', 'Income'])
plt.title("Decision Tree for Buying a Computer\n")
plt.show()