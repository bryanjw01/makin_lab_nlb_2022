from load_model import Loader
import torch
from torch import nn

class Neural_RoBERTa(nn.Module):
    def __init__(self, config):
        super(Neural_RoBERTa, self).__init__()
        model_name = config.keys()
        self.Neural_RoBERTa = Loader.load_model(model_name[0])
        self.RNN = Loader.load_model(model_name[1])
    
    def forward(self, X):
        return self.Neural_RoBERTa(X) * 0.5 + self.RNN(X) * 0.5