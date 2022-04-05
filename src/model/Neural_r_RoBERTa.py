from config import MODEL
from load_model import Loader
import torch
from torch import nn

class Neural_r_RoBERTa(nn.Module):
    def __init__(self):
        super(Neural_r_RoBERTa, self).__init__()
        self.Neural_RoBERTa = Loader.load_model(MODEL.NEURAL_ROBERTA.name)
        self.RNN = Loader.load_model(MODEL.RNN_F.name)
    
    def forward(self, X):
        return self.Neural_RoBERTa(X) * 0.5 + self.RNN(X) * 0.5