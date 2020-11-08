import torch.nn as nn
import torch.nn.functional as F
from skimage import color, transform
import cv2
import numpy as np


class Model(nn.Module):
    def __init__(self, history_length):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(history_length, 8, kernel_size=7, stride=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(2)
        self.conv4 = nn.Conv2d(32, 2, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(2)
        # self.flat = nn.Flatten()
        # self.conv2_drop = nn.Dropout2d()
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(576, 216)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(128, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(216, 12)

    def forward(self, x):
        # print('Input:', x.size())
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # x= self.bn1(x)

        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = self.bn4(x)
        # x = F.relu(self.conv3(x))
        # x = self.bn4(F.relu(self.conv4(x)))
        # print(x.size())
        # x = F.relu(F.max_pool2d(self.conv4(x), 2))
        # x = self.bn1(x)
        x = x.view(-1, 576)
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        # x = self.bn2(x)
        # x = self.dropout(x)
        # x = F.dropout(x, training=self.training)
        # x = F.relu(self.fc2(x))
        # x = F.dropout(x, training=self.training)
        # x = F.relu(self.fc3(x))

        # x = self.bn3(x)
        # print(x.size())
        x = self.fc4(x)

        return x


def rgb2gray(rgb):
    """
    this method converts rgb images to grayscale.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(float)
    gray /= 255.0
    return gray
    # gray = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    # return gray
# return 2 * color.rgb2gray(rgb) - 1.0