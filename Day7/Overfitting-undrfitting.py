from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Underfitting: very shallow tree
underfit_model = DecisionTreeClassifier(max_depth=1)
underfit_model.fit(X_train, y_train)
print("Underfitting - Train Accuracy:", underfit_model.score(X_train, y_train))
print("Underfitting - Test Accuracy:", underfit_model.score(X_test, y_test))

# Overfitting: very deep tree
overfit_model = DecisionTreeClassifier(max_depth=None)
overfit_model.fit(X_train, y_train)
print("Overfitting - Train Accuracy:", overfit_model.score(X_train, y_train))
print("Overfitting - Test Accuracy:", overfit_model.score(X_test, y_test))

