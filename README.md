<img width="787" height="649" alt="image" src="https://github.com/user-attachments/assets/ee0f2f0f-4727-4a37-9c5a-4ace0a6e5c98" /># Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

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
Developed by: Naveen
RegisterNumber: 21222504076
*import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn .model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("student_scores.csv")

print("First 5 rows of the dataset:")
print(df.head())

print("Last 5 rows of the dataset:")
print(df.tail())

X = df.iloc[:, :-1].values
Y = df.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)


model = LinearRegression()
model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)

print("Predicted values:")
print(Y_pred)
print("Actual Values")
print(Y_test)

plt.scatter(X_train,Y_train, color="blue", label ="Actual scores")
plt.title("Hours vs Scores - Training Set")
plt.xlabel("Hours Studied")
plt.ylabel("Scores Achieved")
plt.legend()
plt.show()

plt.scatter(X_test, Y_test, color='green', label="Actual Scores")
plt.plot(X_train, model.predict(X_train), color='red', label="Fitted Line")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores Achieved")
plt.legend()
plt.show()


mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print('Mean Squared Error (MSE) =', mse)
print('Mean Absolute Error (MAE) =', mae)
print('Root Mean Squared Error (RMSE) =', rmse)
```


## Output:
<img 
<img width="1058" height="709" alt="Screenshot 2026-01-31 162155" src="https://github.com/user-attachments/assets/8eb98d04-bff5-4b64-998b-1bf5dc5754a2" />
<img width="787" height="649" alt="Screenshot 2026-02-02 114434" src="https://github.com/user-attachments/assets/abdc968b-843a-46bb-9820-099fd092e1cc" />

<img width="1058" height="709" alt="Screenshot 2026-01-31 162155" src="https://github.com/user-attachments/assets/2c44a13d-1196-4b7d-944e-a97945822857" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
