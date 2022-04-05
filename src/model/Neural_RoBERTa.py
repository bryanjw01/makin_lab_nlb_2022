from transformers import RobertaConfig, RobertaModel
import torch
from torch import nn

class Neural_RoBERTa(nn.Module):
    def __init__(self, config):
        super(Neural_RoBERTa, self).__init__()
        configuration = RobertaConfig(num_hidden_layers=config['n_layers_trans'],
                                        num_attention_heads=config['n_heads'], 
                                        hidden_size=config['output_dim'], 
                                        intermediate_size=config['output_dim']*2)
        
        model = RobertaModel(configuration)
        self.model = model.encoder.layer[0]
        self.rnn = torch.nn.GRU(input_size=config['input_dim'],
                                    hidden_size=config['hidden_dim'],
                                    num_layers=config['n_layers'],
                                    batch_first=True,
                                    dropout=config['p'],
                                    bidirectional=True)
        self.transform = torch.nn.Linear(config['hidden_dim']*2, config['output_dim'])
        self.transform2 = torch.nn.Linear(config['output_dim'], config['output_dim'])
        self.dropout1 = torch.nn.Dropout(p=config['p1'])
        self.dropout2 = torch.nn.Dropout(p=config['p2'])
        self.dropout3 = torch.nn.Dropout(p=config['p3'])
    
    def forward(self, X):
        output, _ = self.rnn(self.dropout1(X))
        output = self.transform(self.dropout2(output))
        output = self.model(output)[0]
        output = self.transform2(self.dropout3(output))        
        return torch.exp(output)