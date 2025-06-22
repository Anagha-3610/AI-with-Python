import pandas as pd
import numpy as np

#Sample DataFrame with missing values
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, np.nan, 30, np.nan, 22],
    'City': ['New York', 'Los Angeles', np.nan, 'Chicago', 'Houston']
}

df = pd.DataFrame(data)

print("Original DataFrame with missing values:\n")
print(df)

# Drop rows with any missing values
df_dropped = df.dropna()
print("\nDataFrame after dropping rows with missing values:\n")
print(df_dropped)

# Fill missing values in 'Age' with a specific value (e.g., 0)
df_fill_zero = df.copy()
df_fill_zero['Age'].fillna(0, inplace=True)
print("\nFilled missing 'Age' with 0:\n")
print(df_fill_zero)

# Fill missing values in 'Age' with the mean age
df_fill_mean = df.copy()
mean_age = df_fill_mean['Age'].mean()
df_fill_mean['Age'].fillna(mean_age, inplace=True)
print(f"\nFilled missing 'Age' with mean ({mean_age:.2f}):\n")
print(df_fill_mean)

# Forward fill (propagate last valid value forward)
df_ffill = df.copy()
df_ffill.fillna(method='ffill', inplace=True)
print("\nDataFrame after forward fill:\n")
print(df_ffill)