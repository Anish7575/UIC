# Name: Sai Anish Garapati
# UIN: 650208577

from math import gamma
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import LSTM
import torch.optim as optim
from torch.optim import optimizer
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from random import shuffle


class Encoder(nn.Module):
    def __init__(self, input_size=27, hidden_size=256, num_layers=2):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)

        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size=27, hidden_size=256, output_size=27, num_layers=2):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        x, (hidden, cell) = self.lstm(x, (hidden, cell))
        x = self.fc(x)
        return x, hidden, cell


class seq2seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, input, target):
        outputs = torch.zeros(11, 1, 27).to(device)
        hidden, cell = self.encoder(input)
        x = target[0, :]
        for i in range(0, 11):
            output, hidden, cell = self.decoder(x.reshape(1, 1, 27).type(torch.FloatTensor), hidden, cell)
            outputs[i] = output
            x = torch.zeros_like(x)
            x[0][output.argmax(2)] = 1
        return outputs


def one_hot_encoder(name, seq_length):
    name_tensor = torch.zeros(seq_length, 1, 27)
    for i in range(len(name)):
        name_tensor[i][0][ord(name[i]) - ord('a') + 1] = 1
    for i in range(seq_length - len(name)):
        name_tensor[len(name) + i][0][0] = 1

    target_tensor = torch.zeros(seq_length)
    eos_tensor = torch.zeros(1, 1, 27)
    eos_tensor[0][0][0] = 1

    for i in range(1, len(name)):
        target_tensor[i - 1] = ord(name[i]) - ord('a') + 1
    for i in range(seq_length - len(name) + 1):
        target_tensor[len(name) - 1 + i] = 0
    return name_tensor, torch.cat((name_tensor[1:], eos_tensor), 0), target_tensor


def train(model, device, names, seq_length, optimizer, epoch):
    model.train()
    print(optimizer.param_groups[-1]['lr'])
    tot_loss = 0

    for name in names:
        name_tensor, target, target_tensor = one_hot_encoder(name, seq_length)
        name_tensor.to(device)
        target.to(device)
        target_tensor.to(device)
        loss = 0

        optimizer.zero_grad()
        output = model(name_tensor.reshape(1, seq_length, 27), target.type(torch.LongTensor))
        loss += torch.nn.CrossEntropyLoss()(output.squeeze(1), target_tensor.type(torch.LongTensor))
        loss.backward()
        optimizer.step()

        tot_loss += loss.item()

    return tot_loss / len(names)


if __name__ == '__main__':
    max_seq_length = 11

    learning_rate = 0.002
    num_epochs = 100

    names_list = open('./names.txt', 'r').read().split('\n')
    names_dataset = [name.lower() for name in names_list]

    random.Random(2021).shuffle(names_dataset)

    torch.manual_seed(2021)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    encoder = Encoder()
    decoder = Decoder()
    model = seq2seq(encoder, decoder, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    prev_loss = -1
    cur_loss = 0
    epoch_loss = []
    for epoch in range(num_epochs):
        cur_loss = train(model, device, names_dataset, max_seq_length, optimizer, epoch + 1)
        epoch_loss.append(cur_loss)
        if prev_loss != -1:
            if cur_loss > prev_loss:
                scheduler.step()

        if abs(cur_loss - prev_loss) < 1e-7:
            break;

        prev_loss = cur_loss
        print('End of epoch {}: loss: {}'.format(epoch + 1, cur_loss))

    plt.title('Plot of Epochs vs loss for training data')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(list(range(1, len(epoch_loss) + 1)), epoch_loss, 'r', label='Epoch vs Loss')
    plt.legend(title='Legend')
    plt.show()

    torch.save(model.state_dict(), './0702-650208577-Garapati.pt')
