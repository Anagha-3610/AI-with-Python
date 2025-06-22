import matplotlib.pyplot as plt
import numpy as np
# Define a function
def f(x):
    return x**2 + 2*x + 1

# Derivative of f(x) = 2x + 2
def f_prime(x):
    return 2*x + 2

# Plot function and derivative
x = np.linspace(-10, 10, 100)
y = f(x)
y_prime = f_prime(x)

plt.plot(x, y, label ="f(x)")
plt.plot(x, y_prime,label="f'(x)")
plt.legend()
plt.title("Function and Its Derivative")
plt.grid()
plt.show()