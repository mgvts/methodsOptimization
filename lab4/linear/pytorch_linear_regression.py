import random

import torch


def take_batch(x_tensor, y_tensor, batch):
    size = x_tensor.size()[0]
    select = []
    while len(select) < batch:
        t = random.randint(0, size - 1)
        if t not in select:
            select.append(t)
    return torch.tensor([[x_tensor[i]] for i in select]), torch.tensor([[y_tensor[i]] for i in select])


class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize, _x, _y):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)
        self.linear.weight = torch.nn.Parameter(torch.tensor([[10.]], requires_grad=True))
        self.linear.bias = torch.nn.Parameter(torch.tensor([10.], requires_grad=True))

    def forward(self, X):
        predictions = self.linear(X)
        return predictions

    def get_points(self):
        return [self.linear.bias.data.item(), self.linear.weight.item()]


class PyTorchLinearRegression:
    """
    :param X [x0, x1 ... xn] xi - [xi0, xi1 ... xik]
    :param Y [y0, y1 ... yn]
    :param batch set None for don't use batch
    """

    def __init__(self, X, Y, start_point, batch=None):
        self.X = X[:, 1]
        self.Y = Y
        self.start_point = start_point
        self.batch = batch

    def _grad_down(self, model, optimizer, runs, eps):
        x_tensor = torch.from_numpy(self.X.reshape(-1, 1)).float()
        y_tensor = torch.from_numpy(self.Y.reshape(-1, 1)).float()
        criterion = torch.nn.MSELoss()

        _x, _y = take_batch(x_tensor, y_tensor, self.batch)

        for epoch in range(runs):
            optimizer.zero_grad()
            # _x, _y = take_batch(x_tensor, y_tensor, self.batch)
            predictions = model(x_tensor)
            loss = criterion(predictions, y_tensor)
            loss.backward()
            optimizer.step()
            # if loss.item() < eps:
            #     break
        return model.get_points()

    def _grad_down_points(self, model, optimizer, runs, eps):
        x_tensor = torch.from_numpy(self.X.reshape(-1, 1)).float()
        y_tensor = torch.from_numpy(self.Y.reshape(-1, 1)).float()
        criterion = torch.nn.MSELoss()

        _x, _y = take_batch(x_tensor, y_tensor, self.batch)

        points = [model.get_points()]

        for epoch in range(runs):
            optimizer.zero_grad()
            _x, _y = take_batch(x_tensor, y_tensor, self.batch)
            predictions = model(_x)
            loss = criterion(predictions, _y)
            loss.backward()
            optimizer.step()
            points.append(model.get_points())
            if loss.item() < eps:
                break
        return points

    def stochastic_grad_down(self, alpha=0.001, runs=1000, eps=0.0001):
        model = LinearRegression(1, 1, self.start_point[0], self.start_point[1])
        optimizer = torch.optim.SGD(model.parameters(), lr=alpha, momentum=False, nesterov=False)
        return self._grad_down(model, optimizer, runs, eps)

    def stochastic_grad_down_points(self, alpha=0.001, runs=1000, eps=0.0001):
        model = LinearRegression(1, 1, self.start_point[0], self.start_point[1])
        optimizer = torch.optim.SGD(model.parameters(), lr=alpha, momentum=False, nesterov=False)
        return self._grad_down_points(model, optimizer, runs, eps)

    def momentum_stochastic_grad_down(self, y=0.9, alpha=0.001, runs=1000, eps=0.0001):
        model = LinearRegression(1, 1, self.start_point[0], self.start_point[1])
        optimizer = torch.optim.SGD(model.parameters(), lr=alpha, momentum=y, nesterov=False)
        return self._grad_down(model, optimizer, runs, eps)

    def momentum_stochastic_grad_down_points(self, y=0.9, alpha=0.001, runs=1000, eps=0.0001):
        model = LinearRegression(1, 1, self.start_point[0], self.start_point[1])
        optimizer = torch.optim.SGD(model.parameters(), lr=alpha, momentum=y, nesterov=False)
        return self._grad_down_points(model, optimizer, runs, eps)

    def nesterov_stochastic_grad_down(self, y=0.9, alpha=0.001, runs=1000, eps=0.0001):
        model = LinearRegression(1, 1, self.start_point[0], self.start_point[1])
        optimizer = torch.optim.SGD(model.parameters(), lr=alpha, momentum=y, nesterov=True)
        return self._grad_down(model, optimizer, runs, eps)

    def nesterov_stochastic_grad_down_points(self, y=0.9, alpha=0.001, runs=1000, eps=0.0001):
        model = LinearRegression(1, 1, self.start_point[0], self.start_point[1])
        optimizer = torch.optim.SGD(model.parameters(), lr=alpha, momentum=y, nesterov=True)
        return self._grad_down_points(model, optimizer, runs, eps)

    def adagrad_stochastic_grad_down(self, alpha=0.7, runs=1000, eps=0.0001):
        model = LinearRegression(1, 1, self.start_point[0], self.start_point[1])
        optimizer = torch.optim.Adagrad(model.parameters(), lr=alpha)
        return self._grad_down(model, optimizer, runs, eps)

    def adagrad_stochastic_grad_down_points(self, alpha=0.7, runs=1000, eps=0.0001):
        model = LinearRegression(1, 1, self.start_point[0], self.start_point[1])
        optimizer = torch.optim.Adagrad(model.parameters(), lr=alpha)
        return self._grad_down_points(model, optimizer, runs, eps)

    def rms_stochastic_grad_down(self, W=4, alpha=0.7, runs=1000, eps=0.0001):
        model = LinearRegression(1, 1, self.start_point[0], self.start_point[1])
        optimizer = torch.optim.RMSprop(model.parameters(), lr=alpha, alpha=W)
        return self._grad_down(model, optimizer, runs, eps)

    def rms_stochastic_grad_down_points(self, W=4, alpha=0.7, runs=1000, eps=0.0001):
        model = LinearRegression(1, 1, self.start_point[0], self.start_point[1])
        optimizer = torch.optim.RMSprop(model.parameters(), lr=alpha, alpha=W)
        return self._grad_down_points(model, optimizer, runs, eps)

    def adam_stochastic_grad_down(self, b1=0.9, b2=0.9, alpha=0.01, runs=1000, eps=0.0001):
        model = LinearRegression(1, 1, self.start_point[0], self.start_point[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=alpha, betas=(b1, b2))
        return self._grad_down(model, optimizer, runs, eps)

    def adam_stochastic_grad_down_points(self, b1=0.9, b2=0.9, alpha=0.01, runs=1000, eps=0.0001):
        model = LinearRegression(1, 1, self.start_point[0], self.start_point[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=alpha, betas=(b1, b2))
        return self._grad_down_points(model, optimizer, runs, eps)
