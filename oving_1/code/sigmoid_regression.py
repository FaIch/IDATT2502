import torch
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('../data/day_head_circumference.csv', dtype='float')
train_y = train.pop('head circumference')

train_x = torch.tensor(train.to_numpy(), dtype=torch.float)
train_y = torch.tensor(train_y.to_numpy(), dtype=torch.float).reshape(-1, 1)


class SigmoidRegressionModel:
    def __init__(self, max):
        self.max = max
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return 20 * torch.sigmoid((x @ self.W + self.b)) + 31

    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)


model = SigmoidRegressionModel(train.shape[0])

optimizer = torch.optim.SGD([model.W, model.b], 0.000001)
for epoch in range(100000):
    model.loss(train_x, train_y).backward()
    optimizer.step()

    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(train_x, train_y)))

# %%
plt.figure('Sigmoid regression')
plt.title('Sigmoid regression')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(train_x, train_y)
x = torch.arange(torch.min(train_x), torch.max(train_x), 1.0).reshape(-1, 1)
y = model.f(x).detach()
plt.plot(x, y, color='orange',)

plt.legend()
plt.show()
