# Support Vector Machine
#
# it classifies data into true (1) or false (0)
# in this case, we are looking at credit card transactions
# and we are trying to classify if they are fraudulent (1) or not (0)
#
# The scikit learn library is used
#
# dataset is available at: https://www.kaggle.com/mlg-ulb/creditcardfraud


import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# 1 - Import Data
data = pd.read_csv("creditcard.csv")


# 2 - Separate features from classifications
X = data.drop(['Class'], axis='columns') # assign eveything but Class to X (features)
y = data.Class  # assign the class column (1's and 0's) to y

# inbuilt function separates data into train and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 


# 3 - Modelling/scoring
model = SVC()

model.fit(X_train, y_train)

score = model.score(X_test, y_test)*100 # test our model on the test set, then give it an accuracy score
format_score = "{:.2f}".format(score)
print('The model accuracy is currently:', format_score, '%') # print out the score of the model


# 4 - Predicting

#(i) - this block generates a random transaction
from sklearn.datasets import make_blobs
Xnew, _ = make_blobs(n_samples=1, centers=2, n_features=30, random_state=1) 
print('Randomly generated transaction:\n', Xnew)


# (ii) - this asks user for array input instead
'''
import numpy as np

Xnew=[]  
print("Please enter the 30 feature values")
for i in range(30):  
    Xnew.append(float(input())) 

Xnew = np.array([Xnew])
print('User generated transaction:\n', Xnew)
'''

ynew = model.predict(Xnew)

if ynew == [0]:
    print('This is not a fraudulent transaction.')

else:
    print('Fraud detected.')

