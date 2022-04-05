from config import CHECKPOINT_PATH, MODEL_TYPE, DATASET_TYPE, PHASE
import os
import torch

class Loader:
    @staticmethod
    def load_model(model_name=None):
        if model_name:
            checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, f'{PHASE}_{model_name}_{DATASET_TYPE.name}.ckpt'))
        else:
            checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, f'{PHASE}_{MODEL_TYPE.name}_{DATASET_TYPE.name}.ckpt'))
        return checkpoint

if __name__ == "__main__":
    print(Loader.load_model())