import torch
import torch.nn as nn
import torchvision
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 1, 28, 28).float().to(device)
y_train = torch.zeros((mnist_train.targets.shape[0], 10)).to(device)
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 1, 28, 28).float().to(device)
y_test = torch.zeros((mnist_test.targets.shape[0], 10)).to(device)
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1

mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

batches = 600
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)


class ConvolutionalNeuralNetworkModel(nn.Module):

    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        self.logits = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 10))

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


model = ConvolutionalNeuralNetworkModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), 0.001)
for epoch in range(20):
    for batch in range(len(x_train_batches)):
        model.loss(x_train_batches[batch].to(device), y_train_batches[batch].to(device)).backward()
        optimizer.step()
        optimizer.zero_grad()

    print("accuracy = %s" % model.accuracy(x_test.to(device), y_test.to(device)))
