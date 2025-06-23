import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1.2, 1.9, 3.2, 3.9, 5.1])


# Transform to polynomial features
degree = 2
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(X)

# Fit polynomial regression
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_poly_pred = poly_reg.predict(X_poly)

# Metrics
mse_poly = mean_squared_error(y, y_poly_pred)
rmse_poly = np.sqrt(mse_poly)
r2_poly = r2_score(y, y_poly_pred)

# Output
print(f"\nPolynomial Regression (degree={degree}) Results:")
print(f"MSE: {mse_poly:.4f}")
print(f"RMSE: {rmse_poly:.4f}")
print(f"R²: {r2_poly:.4f}")

# Plot
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_poly_pred, color='green', label='Polynomial Fit')
plt.title(f'Polynomial Regression (Degree {degree})')
plt.legend()
plt.grid(True)
plt.show()