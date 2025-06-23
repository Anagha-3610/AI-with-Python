import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder


df = sns.load_dataset('titanic')
print("THE FIRST 5 ROWS OF TITANIC :=>")
print(df.head())
print()
print()
print("STATISTICS AND INFO :=>")
print(df.info())
print(df.describe())
print()
print()
df_fill_med_age=df.copy()
median_age=df_fill_med_age['age'].median()
df_fill_med_age['age'].fillna(median_age,inplace=True)
print("FILLED MISSING AGE VALUES WITH THE MEDIAN AGE :=>")
print(df_fill_med_age)
print()
print()
df_fill_embarked=df.copy()
most_popular=df_fill_embarked['embarked'].mode()
df_fill_embarked['embarked'].fillna(most_popular,inplace=True)
print("FILLED MISSING EMBARKED VALUES WITH MOST COMMON PORT(MODE) :=>")
print(df_fill_embarked)
print()
print()
df_dropped=df.dropna()
print("DROPPED ALL THE DECK COLUMNS :=>")
print(df_dropped)
print()
print()
le = LabelEncoder()
df['class'] = le.fit_transform(df['class'])
df['deck'] = le.fit_transform(df['deck'])
print(df)
print()
print()
print()

y=df.index
x=df['survived']
plt.ylabel("Index")
plt.xlabel("Survived people")
sns.histplot(x=x,y=y,data=df,color='yellow')
#plt.plot(x,y,label="survival_count")
plt.legend
plt.grid
plt.show()

x_2=df['survived']
y_2=df['age']
plt.xlabel("Survival")
plt.ylabel("Age")
sns.histplot(x=x_2,y=y_2,data=df,color='green')
plt.legend
plt.grid
plt.show()

x_2=df['survived']
y_3=df['pclass']
plt.ylabel("Passenger class")
plt.xlabel("Survival")
sns.histplot(x=x_2,y=y_3,data=df,color='red')
plt.legend
plt.grid
plt.show()

x_2=df['survived']
y_3=df['sex']
plt.ylabel("Sex")
plt.xlabel("Survival")
sns.histplot(x=x_2,y=y_3,data=df,color='blue')
plt.legend
plt.grid
plt.show()

