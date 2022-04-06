from config import CONFIG, LOG_PATH, RESULT_PATH, CHECKPOINT_PATH, PHASE, PATIENCE, TEST_SIZE, USE_GPU, \
    EPOCHS, MODEL_TYPE, DATASET_TYPE, BIN_SIZE, DEVICE, MAX_GPUS, TRAIN, LEVEL, MODEL
from dataloader import DataLoader as DL
from load_model import Loader
from logger import setup_logger
import logging
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
    logger.debug("Loading Data")
    dl = DL(DATASET_TYPE.value, PHASE, BIN_SIZE, TEST_SIZE)
    logger.debug("Successfully Loaded Data")
    if TRAIN:
        logger.debug("Loading Model")
        # Load model config
        cfg = CONFIG[MODEL_TYPE.value][DATASET_TYPE.value]

        # Add input and output dimension to cfg
        cfg['input_dim'] = dl.train_input.shape[2]
        cfg['output_dim'] = dl.train_output.shape[2]

        # Create Model
        model = Model(cfg).to(DEVICE)
        train_input, train_output = dl.get_train_set()
        val_input, val_output = dl.get_test_set()
        logger.debug("Successfully Loaded Model") 
        logger.debug("Creating Trainer")
        # Init the Trainer class
        runner = Trainer(
            model=model,
            data=(train_input, train_output, val_input, val_output, dl.get_val_set()),
            train_cfg={'lr': cfg['LR'], 'alpha': cfg['WEIGHT'], 'cd_ratio': cfg['CD_RATIO']},
            num_gpus=MAX_GPUS,
            model_name=MODEL_TYPE.value,
            dataset_name=DATASET_TYPE.value,
            checkpoint_path=CHECKPOINT_PATH,
            phase=PHASE,
            use_gpu=USE_GPU,
            device=DEVICE
        )
        logger.debug("Successfully Created Trainer")
        logger.debug("Starting Training")
        # Start Training Process
        train_log = runner.train(n_iter=EPOCHS, patience=PATIENCE)
        logger.debug("Finished Training")
        logger.debug("Saving Training Stats")
        # Logging Training Losses
        train_log = pd.DataFrame(train_log)
        train_log.to_csv(os.path.join(LOG_PATH, f'{PHASE}_{MODEL_TYPE.value}_{DATASET_TYPE.value}_train_log.csv'))

        # Delete models and use garbage collection to clear memory
        del runner.model
        del model

        gc.collect()
        # Load Best Model
        cfg = CONFIG[MODEL_TYPE.value][DATASET_TYPE.value]

        # Add input and output dimension to cfg
        cfg['input_dim'] = dl.train_input.shape[2]
        cfg['output_dim'] = dl.train_output.shape[2]

        # Create Model
        model = Model(cfg)
        model.to(DEVICE)

        if EPOCHS > 0:
            logger.debug('Loading Best Model')
            checkpoint = Loader.load_model(CHECKPOINT_PATH, PHASE, MODEL_TYPE.value, DATASET_TYPE.value)
            assert checkpoint, f"Checkpoint for model is {checkpoint}"
            model.load_state_dict(checkpoint['state_dict'])
            model.to(DEVICE)
        logger.debug('Finished Loading Model')

    else:
        # Load Best Model
        cfg = CONFIG[MODEL_TYPE.value][DATASET_TYPE.value]

        # Add input and output dimension to cfg
        cfg['input_dim'] = dl.train_input.shape[2]
        cfg['output_dim'] = dl.train_output.shape[2]

        # Create Model
        logger.debug('Loading Best Model')
        model = Model(cfg)
        model.to(DEVICE)
        checkpoint = Loader.load_model(CHECKPOINT_PATH, PHASE, MODEL_TYPE.value, DATASET_TYPE.value)
        assert checkpoint, f"Checkpoint for model is {checkpoint}"
        model.load_state_dict(checkpoint['state_dict'])
        model.to(DEVICE)
        logger.debug('Finished Loading Model')

    # Evaluate train set + eval set
    logger.debug('Evaluation Data')
    model.eval()
    training_predictions = model(dl.get_train_input_set().to(DEVICE)).cpu().detach().numpy()
    eval_predictions = model(dl.get_val_set().to(DEVICE)).cpu().detach().numpy()
    logger.debug('Finished Evaluation Data')

    tlen = dl.train_dict['train_spikes_heldin'].shape[1]
    num_heldin = dl.train_dict['train_spikes_heldin'].shape[2]
    logger.debug('Creating Submission')
    submission = {
        DATASET_TYPE.value: {
            'train_rates_heldin': training_predictions[:, :tlen, :num_heldin],
            'train_rates_heldout': training_predictions[:, :tlen, num_heldin:],
            'eval_rates_heldin': eval_predictions[:, :tlen, :num_heldin],
            'eval_rates_heldout': eval_predictions[:, :tlen, num_heldin:],
            'eval_rates_heldin_forward': eval_predictions[:, tlen:, :num_heldin],
            'eval_rates_heldout_forward': eval_predictions[:, tlen:, num_heldin:]
        }
    }
    logger.debug('Saving Submission')
    save_to_h5(submission, os.path.join(RESULT_PATH, f'{PHASE}_{MODEL_TYPE.value}_{DATASET_TYPE.value}.h5'), overwrite=True)

    if PHASE == 'train':
        logger.debug('Evaluating Submission')
        target_dict = make_eval_target_tensors(dataset=dl.get_dataset(), 
                                       dataset_name=DATASET_TYPE.value,
                                       train_trial_split='train',
                                       eval_trial_split='val',
                                       include_psth=True,
                                       save_file=False)
        logger.info(f'Results: {evaluate(target_dict, submission)}')

if __name__ == "__main__":
    # Setup logger
    logger = logging.getLogger('main')
    logger = setup_logger(logger, '', '', '%(levelname)s | %(name)s | %(message)s', LEVEL.value)
    # Call Main
    main()