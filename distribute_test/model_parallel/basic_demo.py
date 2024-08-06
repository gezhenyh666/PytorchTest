import torch
import torch.nn as nn
import torch.optim as optim

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10,5)

    def forward(self, x):
        x = self.relu(self.net1(x))
        return self.net2(x)


if __name__=="__main__":
    model = ToyModel()
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    outputs = model(torch.randn(20, 10))
    labels = torch.randn(20, 5)
    loss_fn(outputs, labels).backward()
    optimizer.step()







