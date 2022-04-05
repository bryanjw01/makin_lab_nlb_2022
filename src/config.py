import torch 
from enum import Enum

class MODEL(Enum):
    RNN_F = 'RNN-F' # RNN-F
    NEURAL_ROBERTA = 'Neural_RoBERTa' # Neural RoBERTa
    NEURAL_R_ROBERTA = 'Neural_r_RoBERTa' # Neural r-RoBERTa

class DATASET(Enum):
    MC_MAZE =  'mc_maze'
    MC_RTT =  'mc_rtt'
    AREA2_BUMP = 'area2_bump'
    DMFC = 'dmfc_rsg'
    MC_MAZE_LARGE = 'mc_maze_large'
    MC_MAZE_MEDIUM = 'mc_maze_medium'
    MC_MAZE_SMALL = 'mc_maze_small'

MODEL_TYPE = MODEL.NEURAL_ROBERTA
DATASET_TYPE = DATASET.MC_MAZE_SMALL

USE_GPU = torch.cuda.is_available()

MAX_GPUS = 1

CHECKPOINT_PATH = './checkpoint/'

LOG_PATH = './log/'

RESULT_PATH = '../results/'

PHASE = 'test'

DEVICE = torch.device('cuda' if USE_GPU else 'cpu')

TEST_SIZE = 0.25

BIN_SIZE = 5

VERBOSE = True

PATIENCE = 5000

EPOCHS = 20000

DATAPATH_DICT = {
    'mc_maze': '../data/000128/sub-Jenkins/',
    'mc_rtt': '../data/000129/sub-Indy/',
    'area2_bump': '../data/000127/sub-Han/',
    'dmfc_rsg': '../data/000130/sub-Haydn/',
    'mc_maze_large': '../data/000138/sub-Jenkins/',
    'mc_maze_medium': '../data/000139/sub-Jenkins/',
    'mc_maze_small': '../data/000140/sub-Jenkins/',
}

CONFIG = {
    'Neural_RoBERTa':    
        {
            'mc_maze': {
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.001,
                'n_layers_trans': 1, 
                'n_layers': 2, 
                'output_dim': 182,
                'hidden_dim': 40,
                'hidden_dropout': 0.1, 
                'attention_dropout': 0.1, 
                'p': 0.47, 
                'p1': 0.47, 
                'p2': 0.47, 
                'p3': 0.47},

            'mc_rtt': {
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.001,
                'n_layers_trans': 1, 
                'n_layers': 2,
                'output_dim': 130, 
                'hidden_dim': 40, 
                'hidden_dropout': 0.1, 
                'attention_dropout': 0.1, 
                'p': 0.47, 
                'p1': 0.47, 
                'p2': 0.47, 
                'p3': 0.47},

            'area2_bump': {
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.001,
                'n_layers_trans': 1, 
                'n_layers': 2,
                'output_dim': 65,
                'hidden_dim': 40, 
                'hidden_dropout': 0.1, 
                'attention_dropout': 0.1, 
                'p': 0.47, 
                'p1': 0.47, 
                'p2': 0.47, 
                'p3': 0.47},

            'dmfc_rsg': {
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.001,
                'n_layers_trans': 1, 
                'n_layers': 2, 
                'output_dim': 54,
                'hidden_dim': 40, 
                'hidden_dropout': 0.1, 
                'attention_dropout': 0.1, 
                'p': 0.47, 
                'p1': 0.47, 
                'p2': 0.47, 
                'p3': 0.47},

            'mc_maze_large': {
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.001,
                'n_layers_trans': 1, 
                'n_layers': 2,
                'output_dim': 162, 
                'hidden_dim': 40, 
                'hidden_dropout': 0.1, 
                'attention_dropout': 0.1, 
                'p': 0.47, 
                'p1': 0.47, 
                'p2': 0.47, 
                'p3': 0.47},

            'mc_maze_medium': {
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.001,
                'n_layers_trans': 1, 
                'n_layers': 2,
                'output_dim': 162,
                'hidden_dim': 40,
                'hidden_dropout': 0.1, 
                'attention_dropout': 0.1, 
                'p': 0.47, 
                'p1': 0.47, 
                'p2': 0.47, 
                'p3': 0.47},

            'mc_maze_small': {
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.001,
                'n_layers_trans': 1, 
                'n_layers': 2,
                'output_dim': 162, 
                'hidden_dim': 40, 
                'hidden_dropout': 0.1, 
                'attention_dropout': 0.1, 
                'p': 0.47, 
                'p1': 0.47, 
                'p2': 0.47, 
                'p3': 0.47}
        },
    'RNN_F':    
        {
            'mc_maze': {
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.001,
                'n_layers_trans': 1, 
                'n_layers': 2, 
                'output_dim': 182,
                'hidden_dim': 40,
                'hidden_dropout': 0.1, 
                'attention_dropout': 0.1, 
                'p': 0.47, 
                'p1': 0.47, 
                'p2': 0.47, 
                'p3': 0.47},

            'mc_rtt': {
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.001,
                'n_layers_trans': 1, 
                'n_layers': 2,
                'output_dim': 130, 
                'hidden_dim': 40, 
                'hidden_dropout': 0.1, 
                'attention_dropout': 0.1, 
                'p': 0.47, 
                'p1': 0.47, 
                'p2': 0.47, 
                'p3': 0.47},

            'area2_bump': {
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.001,
                'n_layers_trans': 1, 
                'n_layers': 2,
                'output_dim': 65,
                'hidden_dim': 40, 
                'hidden_dropout': 0.1, 
                'attention_dropout': 0.1, 
                'p': 0.47, 
                'p1': 0.47, 
                'p2': 0.47, 
                'p3': 0.47},

            'dmfc_rsg': {
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.001,
                'n_layers_trans': 1, 
                'n_layers': 2, 
                'output_dim': 54,
                'hidden_dim': 40, 
                'hidden_dropout': 0.1, 
                'attention_dropout': 0.1, 
                'p': 0.47, 
                'p1': 0.47, 
                'p2': 0.47, 
                'p3': 0.47},

            'mc_maze_large': {
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.001,
                'n_layers_trans': 1, 
                'n_layers': 2,
                'output_dim': 162, 
                'hidden_dim': 40, 
                'hidden_dropout': 0.1, 
                'attention_dropout': 0.1, 
                'p': 0.47, 
                'p1': 0.47, 
                'p2': 0.47, 
                'p3': 0.47},

            'mc_maze_medium': {
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.001,
                'n_layers_trans': 1, 
                'n_layers': 2,
                'output_dim': 162,
                'hidden_dim': 40,
                'hidden_dropout': 0.1, 
                'attention_dropout': 0.1, 
                'p': 0.47, 
                'p1': 0.47, 
                'p2': 0.47, 
                'p3': 0.47},

            'mc_maze_small': {
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.001,
                'n_layers_trans': 1, 
                'n_layers': 2,
                'output_dim': 162, 
                'hidden_dim': 40, 
                'hidden_dropout': 0.1, 
                'attention_dropout': 0.1, 
                'p': 0.47, 
                'p1': 0.47, 
                'p2': 0.47, 
                'p3': 0.47}
        }
}