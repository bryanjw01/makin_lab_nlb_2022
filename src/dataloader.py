from config import DATAPATH_DICT
import numpy as np

from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors

import torch 

class DataLoader:
    def __init__(self, dataset_name, phase='test', bin_size='5', test_size='0.25'):
        self.train_input, self.train_output, self.eval_input, self.train_dict, self.eval_dict, self.dataset = self.get_data(dataset_name, phase, bin_size)
        self.train_size = int(round(self.train_input.shape[0] * (1 - test_size)))
        

    def get_data(self, dataset_name, phase='test', bin_size=5):
        """Function that extracts and formats data for training model"""
        dataset = NWBDataset(DATAPATH_DICT[dataset_name], 
            skip_fields=['cursor_pos', 'eye_pos', 'cursor_vel', 'eye_vel', 'hand_pos'])
        dataset.resample(bin_size)
        train_split = ['train', 'val'] if phase == 'test' else 'train'
        eval_split = 'test' if phase == 'test' else 'val'
        train_dict = make_train_input_tensors(dataset, dataset_name, train_split, save_file=False, include_forward_pred=True)
        eval_dict = make_eval_input_tensors(dataset, dataset_name, eval_split, save_file=False)
        training_input = np.concatenate([
            train_dict['train_spikes_heldin'],
            np.zeros(train_dict['train_spikes_heldin_forward'].shape),
        ], axis=1)
        training_output = np.concatenate([
            np.concatenate([
                train_dict['train_spikes_heldin'],
                train_dict['train_spikes_heldin_forward'],
            ], axis=1),
            np.concatenate([
                train_dict['train_spikes_heldout'],
                train_dict['train_spikes_heldout_forward'],
            ], axis=1),
        ], axis=2)
        eval_input = np.concatenate([
            eval_dict['eval_spikes_heldin'],
            np.zeros((
                eval_dict['eval_spikes_heldin'].shape[0],
                train_dict['train_spikes_heldin_forward'].shape[1],
                eval_dict['eval_spikes_heldin'].shape[2]
            )),
        ], axis=1)
        return training_input, training_output, eval_input, train_dict, eval_dict, dataset
    
    def get_dataset(self):
        return self.dataset
    
    def get_train_set(self):
        return torch.Tensor(self.train_input[:self.train_size]), torch.Tensor(self.train_output[:self.train_size])

    def get_test_set(self):
        return torch.Tensor(self.train_input[self.train_size:]), torch.Tensor(self.train_output[self.train_size:])

    def get_val_set(self):
        return torch.Tensor(self.eval_input)

    def get_train_input_set(self):
        return torch.Tensor(self.train_input)

if __name__ == "__main__":
    pass