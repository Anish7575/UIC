# Name: Sai Anish Garapati
# UIN: 650208577

import torch
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import shutil
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(2304, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 43)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_model(device, model, test_loader, batch_size):
    model.eval()
    tot_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            tot_loss += torch.nn.CrossEntropyLoss()(output, label).item()
            predictions = output.argmax(dim=1, keepdim=True)
            # print(predictions, label.view_as(predictions))
            correct += predictions.eq(label.view_as(predictions)).sum().item()

    print(correct, len(test_loader) * batch_size, (batch_idx + 1) * batch_size)

    return tot_loss/len(test_loader), 100.0 * correct / (len(test_loader) * batch_size)


if __name__ == '__main__':
    loaded_model = Net()
    loaded_model.load_state_dict(torch.load('./ML_Model.pt'))

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32),
        # transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    torch.manual_seed(2021)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    test_path = '../../../../ML_Project_data/Test/'

    test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    loss, accuracy = test_model(device, loaded_model, test_loader, 1)
    print('Test Loss: {}, Test Accuracy: {}'.format(loss, accuracy))
