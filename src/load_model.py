import os
import torch

class Loader:
    @staticmethod
    def load_model(checkpoint_path:str, phase:str, model_name:str, dataset_name:str):
        if os.path.exists(os.path.join(checkpoint_path, f'{phase}_{model_name}_{dataset_name}.ckpt')):
            return torch.load(os.path.join(checkpoint_path, f'{phase}_{model_name}_{dataset_name}.ckpt'))
        return None

if __name__ == "__main__":
    print(Loader.load_model())