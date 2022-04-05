import argparse

from dataloader import DataLoader as DL
from config import CONFIG, LOG_PATH, RESULT_PATH, CHECKPOINT_PATH, PHASE, PATIENCE, TEST_SIZE, USE_GPU, VERBOSE, \
    EPOCHS, MODEL_TYPE, DATASET_TYPE, BIN_SIZE, DEVICE, MAX_GPUS, MODEL
from load_model import Loader
import gc

from nlb_tools.evaluation import evaluate
from nlb_tools.make_tensors import make_eval_target_tensors
from nlb_tools.make_tensors import save_to_h5


import numpy as np
import pandas as pd

import os

from train import Trainer

if MODEL_TYPE == MODEL.RNN_F:
    from model.RNN_F import RNN_F as Model
elif MODEL_TYPE == MODEL.NEURAL_ROBERTA:
    from model.Neural_RoBERTa import Neural_RoBERTa as Model
elif MODEL_TYPE == MODEL.NEURAL_R_ROBERTA:
    from model.Neural_r_RoBERTa import Neural_r_RoBERTa as Model

def main():
    # Extract data
    dl = DL(MODEL_TYPE.name, PHASE, BIN_SIZE, TEST_SIZE)
    if True and MODEL_TYPE != MODEL.NEURAL_R_ROBERTA:

        # Load model config
        cfg = CONFIG[MODEL_TYPE.name]

        # Add input and output dimension to cfg
        cfg['input_dim'] = dl.train_input.shape[2]
        cfg['output_dim'] = dl.train_output.shape[2]

        # Create Model
        model = Model(cfg).to(DEVICE)
        train_input, train_output, val_input, val_output = dl.get_train_set(), dl.get_test_set() 

        # Init the Trainer class
        runner = Trainer(
            model=model,
            data=(train_input, train_output, val_input, val_output, dl.get_val_set()),
            train_cfg={'lr': cfg['LR'], 'alpha': cfg['WEIGHT'], 'cd_ratio': cfg['RATIO']},
            num_gpus=MAX_GPUS,
            model_name=MODEL_TYPE.name,
            dataset_name=DATASET_TYPE.name,
            checkpoint_path=CHECKPOINT_PATH,
            phase=PHASE,
            use_gpu=USE_GPU,
            device=DEVICE
        )

        # Start Training Process
        train_log = runner.train(n_iter=EPOCHS, patience=PATIENCE, verbose=VERBOSE)

        # Logging Training Losses
        train_log = pd.DataFrame(train_log)
        train_log.to_csv(os.path.join(LOG_PATH, f'{PHASE}_{MODEL_TYPE.name}_{DATASET_TYPE.name}_train_log.csv'))

        # Delete models and use garbage collection to clear memory
        del runner.model
        del model
        gc.collect()

    # Load Best Model
    model = Loader.load_model(CHECKPOINT_PATH, PHASE, MODEL_TYPE.name, DATASET_TYPE.name).to(DEVICE)

    # Evaluate train set + eval set
    model.eval()
    training_predictions = model(dl.get_train_set[0].to(DEVICE)).cpu().detach().numpy()
    eval_predictions = model(dl.get_val_set.to(DEVICE)).cpu().detach().numpy()

    tlen = dl.train_dict['train_spikes_heldin'].shape[1]
    num_heldin = dl.train_dict['train_spikes_heldin'].shape[2]

    submission = {
        DATASET_TYPE.name: {
            'train_rates_heldin': training_predictions[:, :tlen, :num_heldin],
            'train_rates_heldout': training_predictions[:, :tlen, num_heldin:],
            'eval_rates_heldin': eval_predictions[:, :tlen, :num_heldin],
            'eval_rates_heldout': eval_predictions[:, :tlen, num_heldin:],
            'eval_rates_heldin_forward': eval_predictions[:, tlen:, :num_heldin],
            'eval_rates_heldout_forward': eval_predictions[:, tlen:, num_heldin:]
        }
    }

    save_to_h5(submission, os.path.join(RESULT_PATH, f'{PHASE}_{MODEL_TYPE.name}_{DATASET_TYPE.name}.h5'))

    if PHASE == 'train':
        target_dict = make_eval_target_tensors(dataset=dl.get_dataset(), 
                                       dataset_name=DATASET_TYPE.name,
                                       train_trial_split='train',
                                       eval_trial_split='val',
                                       include_psth=True,
                                       save_file=False)
        print(evaluate(target_dict, submission))

if __name__ == "__main__":
    '''
        TODO: 
            Add argparser to take care of 
            num epochs, train/eval, phase, dataset, model, bin_size, test_size
    '''
    main()