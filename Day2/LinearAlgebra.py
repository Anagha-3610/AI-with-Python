import numpy as np

v=np.array([1,2,3])

M=np.array([[1,2],[3,4]])

v2=np.array([1,1,1])

v_sum=v+v2


A=np.array([[1,0],[0,1]])
B=np.array([[4,1],[2,2]])
Mul=np.dot(A,B)

print("Vector Sum:", v_sum)
print("Matrix Product:\n",Mul)