# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Yuvaraj B
RegisterNumber:  212222040186
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/LENOVO/Downloads/student_scores.csv")
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

Y_pred

Y_test

plt.scatter(X_train,Y_train,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)

![Screenshot (127)](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122221735/bbb11b17-cc75-4c6f-bff8-00f050de21fc)
![Screenshot (128)](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122221735/ed973e0f-018c-4cca-bac2-cee7f229b2ba)
![Screenshot (129)](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122221735/4ffa126a-0949-4d61-b0bb-ba3b17798bd3)
![Screenshot (130)](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122221735/5687a9f9-bb63-4c61-94fc-4cb743417180)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
