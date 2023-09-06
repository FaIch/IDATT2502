import torch
import matplotlib.pyplot as plt
import numpy as np

train_x = torch.tensor([[0, 1], [0, 0], [1, 0], [1, 1]], dtype=torch.float).reshape(-1, 2)
train_y = torch.tensor([[1], [1], [1], [0]], dtype=torch.float)


class NANDModel:
    def __init__(self):
        self.W = torch.rand((2,1), requires_grad=True)
        self.b = torch.rand((1,1), requires_grad=True)

    def f(self, x):
        return torch.sigmoid(self.logits(x))

    def logits(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x),y)


model = NANDModel()

optimizer = torch.optim.SGD([model.b, model.W], lr=0.1)
for epoch in range(10000):
    model.loss(train_x, train_y).backward()
    optimizer.step()
    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(train_x, train_y)))
xt = train_x.t()[0]
yt = train_x.t()[1]

fig = plt.figure("Logistic regression: the logical OR operator")

plot1 = fig.add_subplot(111, projection='3d')

plot1_f = plot1.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color="green", label="$\\hat y=f("
                                                                                                    "\\mathbf{"
                                                                                                    "x})=\\sigma("
                                                                                                    "\\mathbf{xW}+b)$")

plot1.plot(xt.squeeze(), yt.squeeze(), train_y[:, 0].squeeze(), 'o', label="$(x_1^{(i)}, x_2^{(i)},y^{(i)})$",
                                                                                                    color="blue")

plot1_info = fig.text(0.01, 0.02, "")

plot1.set_xlim(-0.25, 1.25)
plot1.set_ylim(-0.25, 1.25)
plot1.set_zlim(-0.25, 1.25)


plot1_f.remove()
x1_grid, x2_grid = np.meshgrid(np.linspace(-0.25, 1.25, 10), np.linspace(-0.25, 1.25, 10))
y_grid = np.empty([10, 10])
for i in range(0, x1_grid.shape[0]):
    for j in range(0, x1_grid.shape[1]):
        y_grid[i, j] = model.f(torch.tensor([[(x1_grid[i, j]),  (x2_grid[i, j])]], dtype=torch.float))
plot1_f = plot1.plot_wireframe(x1_grid, x2_grid, y_grid, color="green")


fig.canvas.draw()

plt.show()
