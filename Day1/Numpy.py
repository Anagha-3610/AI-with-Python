import numpy as np

arr = np.array([1, 2, 3, 4])
print(arr)               # [1 2 3 4]
print(arr.dtype)         # int64

arr2 = np.array([10, 20, 30, 40])
print(arr + arr2)        # [11 22 33 44]
print(arr * 2)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A)
print(np.dot(A, B))
print(np.mean(A))      # Mean
print(np.std(A))  

print(np.zeros((2,2)))   # 2x2 matrix of zeros
print(np.eye(3))
print(np.ones((2,2)))    # 2x2 matrix of ones