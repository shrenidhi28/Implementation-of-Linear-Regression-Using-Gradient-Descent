# EXP-3 Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize weights randomly.

2.Compute predicted values.

3.Compute gradient of loss function.

4.Update weights using gradient descent

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: C.Shrenidhi
RegisterNumber:  212223040196
*/
Program to implement the linear regression using gradient descent.
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
X=np.c_[np.ones(len(X1)),X1]
theta=np.zeros(X.shape[1]).reshape(-1,1)
for _ in range(num_iters):
predictions=(X).dot(theta).reshape(-1,1)
errors=(predictions-y).reshape(-1,1)
theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
return theta
data=pd.read_csv("/content/50_Startups.csv",header=None)
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```





## Output:


<img width="177" alt="image" src="https://github.com/shrenidhi28/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/155261096/42ed5540-057b-422d-b652-b426d3cd4d72"> <br><br><br>








data information
<br>
<br>
<img width="263" alt="image" src="https://github.com/shrenidhi28/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/155261096/f87c11a4-a68a-46da-8cb4-085e35100079"> <br><br><br>

<img width="291" alt="image" src="https://github.com/shrenidhi28/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/155261096/cf37c9f2-e3ea-4c79-b8dc-7cde204ce91b">




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
