# This project is a Multivaraite Linear Regression
# We will fit a model to the data then be able to make 
# predictions based on that model.
# 
# The dataset here is red wine quality.

# Dataset available at: 
# https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/

# 0 - Imports

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

data = pd.read_csv("winequality-red.txt", sep=";")
#print(data.head()) 

# 1 - Split data

X = data.drop(['quality'], axis='columns')
y = data.quality

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

# 2 - Model

model = LinearRegression()
model.fit(X_train, y_train)

# 3 - Accuracy

score = model.score(X_test, y_test) # score makes predictions with X_test and compares it to y_test
formatted_score = "{:.2f}".format(score)
print('Model score is:' , formatted_score)

# 4 - Ask user for imput

'''
user_wine=[]  
print("Please enter the 11 feature values")
for i in range(11):  
    Xnew.append(float(input())) 
Xnew = np.array([user_wine])
print('User generated wine:\n', user_wine)
'''

new_wine, _ = make_blobs(n_samples=1, centers=2, n_features=11, random_state=1) 
#print('Randomly generated wine:\n', new_wine)

y_pred = model.predict(new_wine)
print("Predicted wine quality: ", y_pred)
