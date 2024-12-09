# model.py
import torch
import torch.nn as nn
from config import *

class ConvLSTM(nn.Module):
    def __init__(self, input_channels=INPUT_CHANNELS, 
                 hidden_channels=HIDDEN_CHANNELS, 
                 kernel_size=KERNEL_SIZE, 
                 output_timesteps=OUTPUT_TIMESTEPS):
        super(ConvLSTM, self).__init__()
        self.conv_lstm1 = nn.Conv3d(input_channels, hidden_channels, 
                                   kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.bn1 = nn.BatchNorm3d(hidden_channels)
        self.conv_lstm2 = nn.Conv3d(hidden_channels, hidden_channels, 
                                   kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.bn2 = nn.BatchNorm3d(hidden_channels)
        self.conv3d = nn.Conv3d(hidden_channels, 1, 
                               kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def forward(self, x):
        x = self.conv_lstm1(x)
        x = self.bn1(x)
        x = self.conv_lstm2(x)
        x = self.bn2(x)
        x = self.conv3d(x)
        return torch.sigmoid(x)
