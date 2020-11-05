import torch.nn as nn
import torch.nn.functional as F


class PolicyModel(nn.Module):
    def __init__(self):
        super(PolicyModel, self).__init__()
        self.linear1 = nn.Linear(4, 256)
        self.dropout1 = nn.Dropout(p=0.65)
        self.linear2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(p=0.6)
        self.linear3 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout1(x)
        x = F.relu(x)
        # x = self.linear2(x)
        # x = self.dropout1(x)
        # x = F.relu(x)
        x = self.linear3(x)
        return F.softmax(x, dim=1)
