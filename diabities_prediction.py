import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,r2_score,classification_report
df= pd.read_csv("diabities.csv")
df.head(5)
df.info()
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
x.fillna(0, inplace=True)
y.fillna(0, inplace=True)
print(x)
df = df.reset_index()
x=x.astype('float')
y=y.astype('float')
print(x)
x.info()
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=20, random_state=0)

x_train = x_train.astype('float')
y_train = y_train.astype('float')
logreg= LogisticRegression(solver='lbfgs', max_iter= 1000)
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)
print("Accuracy = ", accuracy_score(y_pred,y_test))

