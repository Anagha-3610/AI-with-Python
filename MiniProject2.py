#classify emails as spam or not

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

df=pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv", sep='\t', names=['label', 'message'])

df['label_num']=df.label.map({'ham':0, 'spam':1})

X=df['message']
y=df['label_num']

vec = CountVectorizer()
X_vec = vec.fit_transform(X)

X_train, X_test, y_train, y_test=train_test_split(X_vec, y, test_size=0.2)
model = MultinomialNB()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print(classification_report(y_test,y_pred))



#Predict house prices

from re import L
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

df=pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')

X=df[["crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","b","lstat"]]
y=df['medv']

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2,random_state=0)

model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print(r2_score(y_test,y_pred))