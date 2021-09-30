# This project is a Multivaraite Linear Regression
# We will fit a model to the data then be able to make 
# predictions based on that model.
# 
# The dataset here is about from forest fires in Portugal.
# Our model will allow us to predict the area affected by 
# the fire.

# 0 - Setup & data import

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from helper import monthToNum, dayToNum

data = pd.read_csv("forestfires.csv")
#print(data.head()) 

# I noticed that when we print the data, the months are written  
# out in letters, this is kind if useless to our model so I'll 
# express it as a number and use that as an extra feature


# 1 - Data cleaning

months_original = data['month']
days_original = data['day']

for i, month in enumerate(months_original):
    month_number = monthToNum(month)
    months_original[i] = month_number

for i, day in enumerate(days_original):
    day_number = dayToNum(day)
    print(i, day, day_number)
    days_original[i] = day_number

data['month'] = months_original
data['day'] = days_original

# print(data.head())

# If you uncomment the above line, you can see the days and 
# months are now in numerical format

X = data.drop(['area'], axis='columns')
y = data['area']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 


# 2 - Model

model = LinearRegression()
model.fit(X_train, y_train)


# 3 - Predictions & accuracy

#y_pred = linear_regression.predict(X_sample)

score = model.score(X_test, y_test) # test our model on the test set, then give it an accuracy score
format_score = "{:.2f}".format(score)
print('The model accuracy is currently:', format_score) # print out the score of the model

