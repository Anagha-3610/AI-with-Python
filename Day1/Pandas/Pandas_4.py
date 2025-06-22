import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler

#Sample data
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 45, 35, 50],
    'Salary': [50000, 80000, 60000, 120000]
}

# Create DataFrame
df = pd.DataFrame(data)
print(df)
#Initialize the MinMaxScaler
scaler = MinMaxScaler()

#Apply scaling only to numerical columns
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])

# Show the result
print("After Min-Max Scaling:\n", df)