# This is a logistic regression.
# It will determine if candy is 
# chocolate (1) or not (0) based on 
# some features like if it has peanuts,
# if it tastes like fruit etc.

# Just in time for Halloween !

# Available at: 
# https://www.kaggle.com/fivethirtyeight/the-ultimate-halloween-candy-power-ranking/

# 0 - Imports


# new comment
# another comment
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

data = pd.read_csv("candy-data.csv")
#print(data.head())


# 1 - Separate data

X = data.drop(["chocolate", "competitorname"], axis = 'columns')
y = data.chocolate

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# 2 - Model 

model = LogisticRegression()
model.fit(X_train, y_train)

# 3 - Evaluate Model

score = model.score(X_test, y_test)
print("The score of the model is", format(score, ".4f"))

# 4 - Predicting

#(i) - this block generates a random transaction
# first 8 are binary
# next 2 are 0<x<1
# last 0<x<100

c1 = np.random.choice([0, 1], size=(8,))
c2 = np.random.uniform(size=2)
c3 = np.random.randint(low=1, high=100, size=1)

user_candy = np.concatenate((c1, c2, c3), axis=None)
user_candy = [user_candy]

print('Randomly generated candy:\n', user_candy)



# (ii) - this asks user for array input instead
'''
user_candy=[]  
print("Please enter the 11 feature values")
for i in range(11):  
    Xnew.append(float(input())) 
Xnew = np.array([user_candy])
print('User generated candy:\n', user_candy)
'''

ynew = model.predict(user_candy)

if ynew == [0]:
    print('This is not chocolate.')

else:
    print('This is chocolate.')