import os
import torch

class Loader:
    @staticmethod
    def load_model(checkpoint_path:str, phase:str, model_name:str, dataset_name:str):
        if model_name:
            checkpoint = torch.load(os.path.join(checkpoint_path, f'{phase}_{model_name}_{dataset_name}.ckpt'))
        return checkpoint

if __name__ == "__main__":
    print(Loader.load_model())