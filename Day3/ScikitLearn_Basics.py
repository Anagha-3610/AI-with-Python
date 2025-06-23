#Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
iris=load_iris()
x=iris.data
y=iris.target

#Split Data (train/test)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)

#Train a model
model=DecisionTreeClassifier()
model.fit(x_train,y_train)

#Predictions
y_pred=model.predict(x_test)

#Accuracy
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy of the model is :{accuracy:.2f}")

# Step 4: Train a Model
model = RandomForestClassifier()
model.fit(x_train, y_train)  # Train the model

# Step 5: Predict
y_pred = model.predict(x_test)

# Step 6: Evaluate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Model Accuracy on Wine Dataset: {accuracy:.2f}")