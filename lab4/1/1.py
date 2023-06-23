import random

import numpy as np
import torch

from lab2.examples.linear_regression import twod_line

learning_rate = 0.01
epochs = 1000

X, Y = twod_line.generate_linear_regression_2d(20, shift=(4, 4))
X = X[:, 1]

y_tensor = torch.from_numpy(Y.reshape(-1, 1)).float()
X_tensor = torch.from_numpy(X.reshape(-1, 1)).float()


class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize, _x, _y):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)
        self.linear.weight = torch.nn.Parameter(torch.tensor([[1.]], requires_grad=True))
        self.linear.bias = torch.nn.Parameter(torch.tensor([1.], requires_grad=True))

    def forward(self, X):
        predictions = self.linear(X)
        return predictions


model = LinearRegression(1, 1, 0.1, 1.0)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def take_batch(use_batch=True):
    select = []
    while len(select) < 5:
        t = random.randint(0, 20 - 1)
        if t not in select:
            select.append(t)
    return torch.tensor([[X_tensor[i]] for i in select]), torch.tensor([[y_tensor[i]] for i in select])


_x, _y = take_batch()
print(X_tensor)
print(_x)

print(y_tensor)
print(_y)

for epoch in range(epochs):
    optimizer.zero_grad()
    _x, _y = take_batch()
    predictions = model(_x)
    loss = criterion(predictions, _y)
    # get gradients
    loss.backward()
    # update parameters
    optimizer.step()
    if loss.item() < 0.001:
        break

print(epoch)
print(model.linear.weight.item())
print(model.linear.bias.item())
