from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(256*4*4, 1000)
        self.linear2 = nn.Linear(1000, 1000)
        self.linear3 = nn.Linear(1000, 10)
        self.dropout = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        x = self.pool(self.dropout2(F.relu(self.conv1(x))))
        x = self.pool(self.dropout2(F.relu(self.conv2(x))))
        x = self.pool(self.dropout2(F.relu(self.conv3(x))))
        x = x.view(-1, 256 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        x = F.log_softmax(self.linear3(x), dim=1)
        return x