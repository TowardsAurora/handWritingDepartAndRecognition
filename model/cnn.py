import os

import torch.nn as nn


def count_types():
    count = 0
    for _ in os.listdir(r'D:\python_pyc_1\handwritingRecognition\dev\data\train'):
        count += 1
    return count


# count_types_of_train = count_types()
count_types_of_train = 53


class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(2304, 1024),
            # nn.Dropout(0.5),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            # nn.Linear(1024, 500)
            # nn.Linear(1024, count_types_of_train)
            nn.Linear(1024, 53)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
