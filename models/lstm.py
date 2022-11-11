import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from .early_stopping import *
from tqdm import tqdm_notebook as tqdm

torch.manual_seed(0)

# (1) Encoder
class Encoder(nn.Module):
    def __init__(self, 
                 total_time_steps: int,
                 num_features: int,
                 hidden_size: int,):
        super().__init__()
        
        self.LSTM = nn.LSTM(
            input_size = num_features,
            hidden_size = hidden_size,
            num_layers = 1,
            batch_first=True
        )
        
    def forward(self, x):
        x, (hidden_state, cell_state) = self.LSTM(x)  
        last_lstm_layer_hidden_state = hidden_state[-1,:,:]
        return last_lstm_layer_hidden_state
    
# (2) Decoder
class Decoder(nn.Module):
    def __init__(self, 
                 total_time_steps: int, 
                 num_features: int, 
                 num_decoder_steps: int,):
        super().__init__()

        self.total_time_steps = total_time_steps
        self.num_features = num_features
        self.hidden_size = (2 * num_features)
        self.num_decoder_steps = num_decoder_steps
        self.LSTM = nn.LSTM(
            input_size = num_features,
            hidden_size = self.hidden_size,
            num_layers = 1,
            batch_first = True
        )

        self.fc = nn.Linear(self.hidden_size, 1)
        
    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.num_decoder_steps, 1)
        x, (hidden_state, cell_state) = self.LSTM(x)
        x = x.reshape((-1, self.seq_len, self.hidden_size))
        out = self.fc(x)
        return out
    
# (3) Autoencoder : putting the encoder and decoder together
class LSTM_SeqtoSeq(nn.Module):
    def __init__(self, 
                 total_time_steps: int, 
                 num_features: int, 
                 num_decoder_steps: int,
                 hidden_size: int,):
        super().__init__()
        
        self.total_time_steps = total_time_steps
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_decoder_steps = num_decoder_steps

        self.encoder = Encoder(self.total_time_steps, self.num_features, self.hidden_size)
        self.decoder = Decoder(self.total_time_steps, self.num_features, self.num_decoder_steps)
    
    def forward(self, x):
        torch.manual_seed(0)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded