# Univariate Linear Regression
#
# Done using PyTorch


import torch 
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 1 - Import Data
data = pd.read_csv("heightsandweights.csv")
# print(data) # uncomment to see data

X_numpy, y_numpy = data.Height, data.Weight
X_numpy, y_numpy = np.array(X_numpy), np.array(y_numpy)

plt.plot(X_numpy, y_numpy, 'ro')
#plt.show() # uncomment to see the plot before line fitting

# 2 - Make tensors from np.arrays
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
X = X.view(X.shape[0], 1)

n_samples, n_features = X.shape

# 3 - Model
input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size)


# 4 - Loss and optimiser
learning_rate = 0.005
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 5 - Training Loop
num_epochs = 500
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    # backward pass
    loss.backward()

    # update
    optimizer.step()

    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.2f}')

    

# 6 - Plot

predicted = model(X).detach()
plt.plot(X_numpy, predicted, 'b')
plt.show()


