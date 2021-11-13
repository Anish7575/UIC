# Name: Sai Anish Garapati
# UIN: 650208577

from math import gamma
import torch
import os
from torch._C import _get_warnAlways
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

from torch.serialization import load


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
        outputs = torch.zeros(len(input[0]), 1, 27).to(device)

        hidden, cell = self.encoder(input)
        x = target[0, :]
        for i in range(0, len(input[0])):
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

def test_model(device, model, char, seq_length):
    model.eval()
    generated_name = char

    with torch.no_grad():
        while True:
            name_tensor, target, target_tensor = one_hot_encoder(generated_name, len(generated_name))
            target = torch.ones_like(target)
            target[0][0][0] = 0
            output = model(name_tensor.reshape(1, len(generated_name), 27), target.type(torch.LongTensor))
            output = F.softmax(output[0][-1], dim=0).reshape(1, 27)
            topk_chars = torch.topk(output, 3)

            char_pick = np.random.randint(0, len(topk_chars.indices[0]), 1)

            idx, val = topk_chars.indices[0][char_pick], topk_chars.values[0][char_pick]
            if idx == 0 or len(generated_name) >= 11:
                if len(generated_name) < 3:
                    continue
                return generated_name
            generated_name += chr(96 + idx)


if __name__ == '__main__':
    torch.manual_seed(2021)
    np.random.seed(2021)

    max_seq_length = 11

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    encoder = Encoder()
    decoder = Decoder()
    loaded_model = seq2seq(encoder, decoder, device).to(device)
    loaded_model.load_state_dict(torch.load('./0702-650208577-Garapati.pt'))
    
    generated_names = []

    char_input = input('Enter character between a and z:\n')

    while True:
        name = test_model(device, loaded_model, char_input, 11)
        if name not in generated_names:
            generated_names.append(name)
        if len(generated_names) == 20:
            break
    print(generated_names)
