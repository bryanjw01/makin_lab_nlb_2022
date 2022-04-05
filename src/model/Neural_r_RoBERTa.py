from config import MODEL, CONFIG, CHECKPOINT_PATH, PHASE
from load_model import Loader
from model.Neural_RoBERTa import Neural_RoBERTa
from model.RNN_F import RNN_F
from torch import nn

class Neural_r_RoBERTa(nn.Module):
    def __init__(self, config):
        super(Neural_r_RoBERTa, self).__init__()
        config_NR = CONFIG[MODEL.NEURAL_ROBERTA.value][config['dataset_name']] 
        config_RNN = CONFIG[MODEL.RNN_F.value][config['dataset_name']] 

        # Add input and output dimension to cfg
        config_NR['input_dim'] = config['input_dim']
        config_NR['output_dim'] = config['output_dim']

        config_RNN['input_dim'] = config['input_dim']
        config_RNN['output_dim'] = config['output_dim']

        self.Neural_RoBERTa = Neural_RoBERTa(config_NR)
        self.RNN_F = RNN_F(config_RNN)
        
        checkpoint_NR = Loader.load_model(CHECKPOINT_PATH, PHASE, MODEL.NEURAL_ROBERTA.value, config['dataset_name'])
        checkpoint_RNN = Loader.load_model(CHECKPOINT_PATH, PHASE, MODEL.RNN_F.value, config['dataset_name'])
        
        if checkpoint_NR:
            self.Neural_RoBERTa.load_state_dict(checkpoint_NR['state_dict'])
    
        if checkpoint_RNN:
            self.RNN_F.load_state_dict(checkpoint_RNN['state_dict'])
    
    def forward(self, X):
        return self.Neural_RoBERTa(X) * 0.5 + self.RNN_F(X) * 0.5