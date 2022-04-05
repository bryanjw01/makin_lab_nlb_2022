import torch
from torch import nn


class RNN_F(nn.Module):
    def __init__(self, config):
        super(RNN_F, self).__init__()
        self.rnn = torch.nn.GRU(input_size=config['input_dim'],
                                    hidden_size=config['hidden_dim'],
                                    num_layers=config['n_layers'],
                                    batch_first=True,
                                    dropout=config['p'],
                                    bidirectional=True)

        self.dropout1 = torch.nn.Dropout(p=config['p1'])
        self.dropout2 = torch.nn.Dropout(p=config['p2'])
    
    def forward(self, X):
        output, _ = self.rnn(self.dropout1(X))
        output = self.transform(self.dropout2(output))
        return torch.exp(output)