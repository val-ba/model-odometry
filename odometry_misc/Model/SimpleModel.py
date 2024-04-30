
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RNN, LSTM

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout=0.1, func=torch.relu):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.func = func
        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) 
                                           for i in range(len(hidden_sizes)-1)])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], 3)

    def forward(self, x):
        # Forward pass through the layers
        x = self.func(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.func(layer(x))
            x = F.dropout(x, p=self.dropout)
        x = self.output_layer(x)
        return x
    
class RecurrentNet(nn.Module):

    def __init__(self, input_size, hidden_sizes, recurrent_size=8, recurrent_depth=3, dropout=0.1, func=torch.relu): 
        super(RecurrentNet, self).__init__()
        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.recurrent_size = recurrent_size
        self.recurrent_depth = recurrent_depth
        self.dropout = dropout
        self.func = func
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) 
                                           for i in range(len(hidden_sizes)-1)])
        
        self.rnn1 = RNN(hidden_sizes[-1], recurrent_size, recurrent_depth, batch_first=True)
        # Output layer
        self.output_layer = nn.Linear(recurrent_size, 3)


    def forward(self, x, hidden_states):
        # Add batch dimension if the input is not batched
        x = x.unsqueeze(0) if len(x.shape) == 1 else x
        x = self.func(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.func(layer(x))
            x = F.dropout(x, p=self.dropout)
        x, final_hidden = self.rnn1(x, hidden_states)
        x = self.output_layer(x)
        x = x.squeeze(0) if len(x.shape) == 2 else x
        return x, final_hidden
    
    def init_hidden(self, batch_size):
        if batch_size == 1:
            return torch.zeros(self.recurrent_depth, self.recurrent_size) 
        return torch.zeros(self.recurrent_depth, batch_size, self.recurrent_size)

class LSTMNet(nn.Module):

    def __init__(self, input_size, hidden_sizes, recurrent_size=8, recurrent_depth=3, dropout=0.01, func=torch.relu):
        super(LSTMNet, self).__init__()
        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.recurrent_size = recurrent_size
        self.recurrent_depth = recurrent_depth
        self.dropout = dropout
        self.func = func
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) 
                                           for i in range(len(hidden_sizes)-1)])
        
        self.rnn1 = LSTM(hidden_sizes[-1], recurrent_size, recurrent_depth, batch_first=True)
        # Output layer
        self.output_layer = nn.Linear(recurrent_size, 3)


    def forward(self, x, hidden_states):
        # Add batch dimension if the input is not batched

        x = x.unsqueeze(0) if len(x.shape) == 1 else x

        x = self.func(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.func(layer(x))
            x = F.dropout(x, p=self.dropout)

        x, final_hidden = self.rnn1(x, hidden_states)
        x = self.output_layer(x)
        x = x.squeeze(0) if len(x.shape) == 2 else x
        return x, final_hidden
    
    def init_hidden(self, batch_size):
        if batch_size == 1:
            return (torch.zeros(self.recurrent_depth, self.recurrent_size),
                    torch.zeros(self.recurrent_depth, self.recurrent_size))

        return (torch.zeros(self.recurrent_depth, batch_size, self.recurrent_size),
                torch.zeros(self.recurrent_depth, batch_size, self.recurrent_size))






