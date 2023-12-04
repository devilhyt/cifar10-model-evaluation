# CIFAR10 Model Evaluation

In the Introduction to Artificial Intelligence course, we developed a baseline model based on the Lenet architecture to address the CIFAR10 classification task. Various techniques, including data augmentation, Batch Normalization, and the incorporation of Dropout layers, were employed to enhance the model's generalization capability and accuracy. After training for over 50 epochs, the predictive accuracy reached an impressive 86.6%.

Towards the end of the semester, we conducted experiments using the more powerful Resnet18 model. Remarkably, under the same training conditions, the Resnet18 model achieved a comparable level of predictive accuracy while utilizing fewer parameters.


## Baseline model

![baseline model](/img/baseline-model.png)

```python
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

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        x = F.log_softmax(self.linear3(x), dim=1)
        return x
```

## Result

| model                                           | accuracy |
| ----------------------------------------------- | -------- |
| Baseline model                                  | 84.6%    |
| Add Dropout layers                              | 84.7%    |
| Add Batch Normalization layers                  | 86.3%    |
| Add both Dropout and Batch Normalization layers | 86.6%    |
| ResNet18                                        | 86.6%    |