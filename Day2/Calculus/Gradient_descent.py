#Gradient Descent to minimize f(x)=x^2

x=10 #initial guess
learning_rate=0.1

for i in range(10):
  grad=2*x
  x-=learning_rate*grad
  print(f"Step {i+1} : x={x}")