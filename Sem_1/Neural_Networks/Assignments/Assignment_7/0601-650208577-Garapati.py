# Name: Sai Anish Garapati
# UIN: 650208577

from math import gamma
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import optimizer
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from random import shuffle


class RNN(nn.Module):
    def __init__(self, input_size=27, hidden_size=256, num_layers=2):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, input_size)
    
    def forward(self, x, hidden, cell):
        x, (hidden, cell) = self.lstm(x, (hidden, cell))
        # print(x)
        x = self.fc1(x)
        return x, (hidden, cell)

    def init_hidden_and_cell(self):
        hidden = torch.zeros(self.num_layers, 1, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, 1, self.hidden_size).to(device)
        return hidden, cell


def one_hot_encoder(name, seq_length):
    name_tensor = torch.zeros(seq_length, 1, 27)
    for i in range(len(name)):
        name_tensor[i][0][ord(name[i]) - ord('a') + 1] = 1
    for i in range(seq_length - len(name)):
        name_tensor[len(name) + i][0][0] = 1
    
    target_tensor = torch.zeros(27)

    for i in range(1, len(name)):
        target_tensor[i - 1] = ord(name[i]) - ord('a') + 1

    return name_tensor, target_tensor


def train(model, device, names, seq_length, optimizer, epoch):
    print(optimizer.param_groups[-1]['lr'])
    model.train()
    tot_loss = 0

    for name in names:
        name_tensor, target_tensor = one_hot_encoder(name, seq_length)
        hidden, cell = model.init_hidden_and_cell()
        loss = 0

        optimizer.zero_grad()

        for i in range(seq_length):
            output, (hidden, cell) = model(name_tensor[i].unsqueeze(0), hidden, cell)
            # print(output)
            # print(output[0].shape, target_tensor[i].unsqueeze(0).type(torch.LongTensor).shape)
            loss += torch.nn.CrossEntropyLoss()(output[0], target_tensor[i].unsqueeze(0).type(torch.LongTensor))
            # print(loss.item())
        
        loss.backward()
        optimizer.step()
        # print(loss.item(), loss.item() / seq_length)
        tot_loss += (loss.item() / seq_length)

    return tot_loss / len(names)


if __name__ == '__main__':
    max_seq_length = 11

# Hyperparameters
    learning_rate = 0.001
    num_epochs = 30

    names_list = open('./names.txt', 'r').read().split('\n')
    names_dataset = [name.lower() for name in names_list]

    random.Random(2021).shuffle(names_dataset)

    torch.manual_seed(2021)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = RNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)

    prev_loss = -1
    cur_loss = 0
    for epoch in range(num_epochs):
        cur_loss = train(model, device, names_dataset, max_seq_length, optimizer, epoch + 1)
        if prev_loss != -1:
            if cur_loss > prev_loss:
                scheduler.step()
        
        prev_loss = cur_loss
        print('End of epoch {}: loss: {}'.format(epoch, cur_loss))
        # scheduler.step()
    
    torch.save(model.state_dict(), 'test.pt')
