import numpy as np
#Probability
flips=np.random.choice([0,1],size=1000)
prob_of_heads=np.mean(flips)


#Statistics
data=np.array([1,2,3,4,5])
mean=np.mean(data)
variance=np.var(data)
std_dev=np.std(data)


print("Probability of heads:", prob_of_heads)
print("Mean:", mean, "Variance:", variance, "Std Dev:", std_dev)