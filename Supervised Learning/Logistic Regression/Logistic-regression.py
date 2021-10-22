# This is a logistic regression.
# It will determine if candy is 
# chocolate (1) or not (0) based on 
# some features like if it has peanuts,
# if it tastes like fruit etc.

# Just in time for Halloween !

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


data = pd.read_csv("candy-data.csv")
print(data.head())

X = data.drop(["chocolate", "competitorname"], axis = 'columns')
y = data.chocolate

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = LogisticRegression()

model.fit(X_train, y_train)


score = model.score(X_test, y_test)
print(format(score, ".4f"))