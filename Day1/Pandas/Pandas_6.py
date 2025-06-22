import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Create the sample DataFrame
data = {
    'Name': ['John', 'Alice', 'Bob'],
    'Gender': ['Male', 'Female', 'Male'],
    'Department': ['Sales', 'HR', 'IT'],
    'Category': ['Beginner', 'Intermediate', 'Expert']
}

df = pd.DataFrame(data)

# One-Hot Encode Gender and Department (nominal)
df = pd.get_dummies(df, columns=['Department'])

# Label Encode Category (ordinal)
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Category'] = le.fit_transform(df['Category'])

# Final Data Check
print("Data Types and Missing Values:")
print(df.info())  # shows data types and if null values exist

print("\nStatistical Summary of Numeric Columns:")
print(df.describe())  # shows mean, std, min, max, etc. for numeric columns

# Final DataFrame
print("\nFinal Encoded DataFrame:")
print(df)