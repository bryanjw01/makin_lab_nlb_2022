from enum import Enum
import torch 

CHECKPOINT_PATH = '/Users/bryanjw01/code/makin_lab_nlb_2022/src/checkpoint'

LOG_PATH = '/Users/bryanjw01/code/makin_lab_nlb_2022/src/log'

RESULT_PATH = '/Users/bryanjw01/code/makin_lab_nlb_2022/results'

DATAPATH_DICT = {
    'mc_maze': '/Users/bryanjw01/code/makin_lab_nlb_2022/data/000128/sub-Jenkins/',
    'mc_rtt': '/Users/bryanjw01/code/makin_lab_nlb_2022/data/000129/sub-Indy/',
    'area2_bump': '/Users/bryanjw01/code/makin_lab_nlb_2022/data/000127/sub-Han/',
    'dmfc_rsg': '/Users/bryanjw01/code/makin_lab_nlb_2022/data/000130/sub-Haydn/',
    'mc_maze_large': '/Users/bryanjw01/code/makin_lab_nlb_2022/data/000138/sub-Jenkins/',
    'mc_maze_medium': '/Users/bryanjw01/code/makin_lab_nlb_2022/data/000139/sub-Jenkins/',
    'mc_maze_small': '/Users/bryanjw01/code/makin_lab_nlb_2022/data/000140/sub-Jenkins/',
}

class MODEL(Enum):
    RNN_F = 'RNN_F' # RNN-F
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

MODEL_TYPE = MODEL.NEURAL_R_ROBERTA

DATASET_TYPE = DATASET.MC_MAZE_SMALL

TRAIN = True

PHASE = 'train' # 'test'

TEST_SIZE = 0.25

BIN_SIZE = 5

VERBOSE = True

PATIENCE = 5000

EPOCHS = 50

USE_GPU = torch.cuda.is_available()

DEVICE = torch.device('cuda' if USE_GPU else 'cpu')

MAX_GPUS = 1

CONFIG = {

    'Neural_RoBERTa':    
        {
            'mc_maze': {
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.001,
                'n_layers_trans': 1, 
                'n_layers': 2, 
                'n_heads': 1,
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
                'n_heads': 5,
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
                'n_heads': 13,
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
                'n_heads': 3, 
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
                'n_heads': 2,
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
                'n_heads': 2,
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
                'n_heads': 2,
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
                'LR': 0.01,
                'n_layers': 2,
                'hidden_dim': 64, 
                'p': 0.5, 
                'p1': 0.5,
                'p2': 0.5},

            'mc_rtt': {
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.01,
                'n_layers': 2,
                'hidden_dim': 64, 
                'p': 0.5, 
                'p1': 0.5,
                'p2': 0.5},

            'area2_bump': {
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.01,
                'n_layers': 2,
                'hidden_dim': 64, 
                'p': 0.6, 
                'p1': 0.6,
                'p2': 0.6},

            'dmfc_rsg': {
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.01,
                'n_layers': 3, 
                'hidden_dim': 64, 
                'p': 0.5, 
                'p1': 0.5,
                'p2': 0.5},

            'mc_maze_large': {
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.01,
                'n_layers': 2,
                'hidden_dim': 64, 
                'p': 0.5, 
                'p1': 0.5,
                'p2': 0.5},

            'mc_maze_medium': {
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.01,
                'n_layers': 2,
                'hidden_dim': 64, 
                'p': 0.5, 
                'p1': 0.5,
                'p2': 0.5},

            'mc_maze_small': {
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.01,
                'n_layers': 2,
                'hidden_dim': 64, 
                'p': 0.6, 
                'p1': 0.6,
                'p2': 0.6}
        },

    'Neural_r_RoBERTa':    
        {
            'mc_maze': {
                'dataset_name': 'mc_maze',
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.001},

            'mc_rtt': {
                'dataset_name': 'mc_rtt',
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.001},

            'area2_bump': {
                'dataset_name': 'area2_bump',
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.001},

            'dmfc_rsg': {
                'dataset_name': 'dmfc_rsg',
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.001},

            'mc_maze_large': {
                'dataset_name': 'mc_maze_large',
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.001},

            'mc_maze_medium': {
                'dataset_name': 'mc_maze_medium',
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.001},

            'mc_maze_small': {
                'dataset_name': 'mc_maze_small',
                'WEIGHT': 5e-7, 
                'CD_RATIO': 0.27, 
                'LR': 0.001}
        }
}