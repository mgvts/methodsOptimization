import random

import numpy as np

from lab2.linear_regression import LinearRegression

import torch


class _LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize, _x, _y):
        super(_LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)
        self.linear.weight = torch.nn.Parameter(torch.tensor([[float(_x)]], requires_grad=True))
        self.linear.bias = torch.nn.Parameter(torch.tensor([float(_y)], requires_grad=True))

    def forward(self, X):
        predictions = self.linear(X)
        return predictions


class TorchLinearRegression(LinearRegression):
    def __init__(self, X, Y, start_point, batch=None):
        super().__init__(X, Y, start_point, batch)
        self.y_tensor = torch.from_numpy(self.Y.reshape(-1, 1)).float()
        self.X_tensor = torch.from_numpy(self.X[:, 1].reshape(-1, 1)).float()

        self.losses = []

    def _create_model(self):
        model = _LinearRegression(1, 1, self.start_point[0, 0], self.start_point[1, 0])
        return model

    def _take_batch(self, use_batch=True):
        if use_batch and self.batch is not None:
            select = []
            while len(select) < self.batch:
                t = random.randint(0, len(self.X) - 1)
                if t not in select:
                    select.append(t)
            return torch.tensor([[self.X_tensor[i]] for i in select]), torch.tensor(
                [[self.y_tensor[i]] for i in select])
        return self.X_tensor, self.y_tensor

    def _pytorch_grad(self, model, optimizer, runs):
        criterion = torch.nn.MSELoss()

        for epoch in range(runs):
            optimizer.zero_grad()
            _x, _y = self._take_batch()
            predictions = model(_x)
            loss = criterion(predictions, _y)
            loss.backward()
            self.losses.append(loss.item())
            optimizer.step()
        return np.matrix([model.linear.bias.item(), model.linear.weight.item()]).T

    def _pytorch_grad_points(self, model, optimizer, runs, eps):
        criterion = torch.nn.MSELoss()
        paths = [np.matrix([model.linear.bias.item(), model.linear.weight.item()]).T]
        for epoch in range(runs):
            optimizer.zero_grad()
            _x, _y = self._take_batch()
            predictions = model(_x)
            loss = criterion(predictions, _y)
            loss.backward()
            self.losses.append(loss.item())
            optimizer.step()
            if loss.item() < eps:
                break
            paths.append(np.matrix([model.linear.bias.item(), model.linear.weight.item()]).T)
        return paths

    def stochastic_grad_down(self, alpha=0.001, runs=1000):
        model = self._create_model()
        return self._pytorch_grad(
            model, torch.optim.SGD(model.parameters(), lr=alpha), runs
        )

    def momentum_stochastic_grad_down(self, y=0.9, alpha=0.001, runs=1000):
        model = self._create_model()
        return self._pytorch_grad(
            model, torch.optim.SGD(model.parameters(), lr=alpha, momentum=y), runs
        )

    def nesterov_stochastic_grad_down(self, y=0.9, alpha=0.001, runs=1000):
        model = self._create_model()
        return self._pytorch_grad(
            model, torch.optim.SGD(model.parameters(), lr=alpha, momentum=y, nesterov=True), runs
        )

    def adagrad_stochastic_grad_down(self, alpha=0.7, runs=1000):
        model = self._create_model()
        return self._pytorch_grad(
            model, torch.optim.Adagrad(model.parameters(), lr=alpha), runs
        )

    def rms_stochastic_grad_down(self, W=4, alpha=0.7, runs=1000):
        model = self._create_model()
        return self._pytorch_grad(
            model, torch.optim.RMSprop(model.parameters(), lr=alpha, alpha=W), runs
        )

    def adam_stochastic_grad_down(self, b1=0.9, b2=0.9, alpha=0.01, runs=1000):
        model = self._create_model()
        return self._pytorch_grad(
            model, torch.optim.Adam(model.parameters(), lr=alpha, betas=(b1, b2)), runs
        )

    def stochastic_grad_down_points(self, alpha=0.001, runs=1000, eps=0.0001):
        model = self._create_model()
        return self._pytorch_grad_points(
            model, torch.optim.SGD(model.parameters(), lr=alpha), runs, eps
        )

    def momentum_stochastic_grad_down_points(self, y=0.9, alpha=0.001, runs=1000, eps=0.0001):
        model = self._create_model()
        return self._pytorch_grad_points(
            model, torch.optim.SGD(model.parameters(), lr=alpha, momentum=y), runs, eps
        )

    def nesterov_stochastic_grad_down_points(self, y=0.9, alpha=0.001, runs=1000, eps=0.0001):
        model = self._create_model()
        return self._pytorch_grad_points(
            model, torch.optim.SGD(model.parameters(), lr=alpha, momentum=y, nesterov=True), runs, eps
        )

    def adagrad_stochastic_grad_down_points(self, alpha=0.7, runs=1000, eps=0.0001):
        model = self._create_model()
        return self._pytorch_grad_points(
            model, torch.optim.Adagrad(model.parameters(), lr=alpha), runs. runs, eps
        )

    def rms_stochastic_grad_down_points(self, W=4, alpha=0.7, runs=1000, eps=0.0001):
        model = self._create_model()
        return self._pytorch_grad_points(
            model, torch.optim.RMSprop(model.parameters(), lr=alpha, alpha=W), runs, eps
        )

    def adam_stochastic_grad_down_points(self, b1=0.9, b2=0.9, alpha=0.01, runs=1000, eps=0.0001):
        model = self._create_model()
        return self._pytorch_grad_points(
            model, torch.optim.Adam(model.parameters(), lr=alpha, betas=(b1, b2)), runs, eps
        )

