from config import CHECKPOINT_PATH, MODEL_NAME, DATASET_NAME
import os
import torch

class Loader:
    @staticmethod
    def load_model(self, model_name=None):
        if model_name:
            checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, f'{model_name}_{DATASET_NAME}.ckpt'))
        else:
            checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, f'{MODEL_NAME}_{DATASET_NAME}.ckpt'))
        return checkpoint

if __name__ == "__main__":
    print(Loader.load_model())