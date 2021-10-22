# This project is a Multivaraite Linear Regression
# We will fit a model to the data then be able to make 
# predictions based on that model.
# 
# The dataset here is red wine quality.

# Dataset available at: 
# https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/


# 0 - Setup & data import

import pandas as pd
pd.options.mode.chained_assignment = None

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# contains dictionaries that let me change from month & day to number:
# from helper import monthToNum, dayToNum 

data = pd.read_csv("wine-quality.txt", sep=";")
print(data.head()) 

X = data.drop(['quality'], axis='columns')
y = data.quality

# 1 - Data Pre- processing






# 2 - Model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

model = LinearRegression()
model.fit(X_train, y_train)


# 3 - Predictions & accuracy

y_pred = model.predict(X_test)

score = model.score(X_test, y_test) # score makes predictions with X_test and compares it to y_test
formatted_score = "{:.2f}".format(score)
print('Model score is:' , formatted_score)

error = mean_squared_error(y_test, y_pred)
formatted_error = "{:.2f}".format(error)
print('The model mean squared error is:', formatted_error)
