import argparse
from config import CONFIG
from nlb_tools.make_tensors import save_to_h5

def main():
    # Run parameters
    # dataset_name = 'area2_bump'
    phase = 'test'
    bin_size = 5

    # Extract data
    training_input, training_output, eval_input = get_data(dataset_name, phase, bin_size)

    # Train/val split and convert to Torch tensors
    num_train = int(round(training_input.shape[0] * 0.75))
    train_input = torch.Tensor(training_input[:num_train])
    train_output = torch.Tensor(training_output[:num_train])
    val_input = torch.Tensor(training_input[num_train:])
    val_output = torch.Tensor(training_output[num_train:])
    eval_input = torch.Tensor(eval_input)

    # Model hyperparams
    L2_WEIGHT = 5e-7
    LR_INIT = 1.0e-3
    CD_RATIO = 0.27
    HIDDEN_DIM = 40
    DROPOUT = 0.47


    RUN_NAME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_model'
    RUN_DIR = './runs/'

    gc.collect()
    print(dataset_name)
    USE_GPU = True
    MAX_GPUS = 1

    init = {'input_dim': train_input.shape[2], 'hidden_dim': HIDDEN_DIM, 'output_dim': train_output.shape[2], 'dropout': DROPOUT, 'model': model}
    model_2 = Roberta(**init).to('cuda')

    if not os.path.isdir(RUN_DIR):
        os.mkdir(RUN_DIR)

    runner = NLBRunner(
        model_init=model_2,
        model_cfg={'input_dim': train_input.shape[2], 'hidden_dim': HIDDEN_DIM, 'output_dim': train_output.shape[2], 'dropout': DROPOUT},
        data=(train_input, train_output, val_input, val_output, eval_input),
        train_cfg={'lr': LR_INIT, 'alpha': L2_WEIGHT, 'cd_ratio': CD_RATIO},
        use_gpu=USE_GPU,
        num_gpus=MAX_GPUS,
    )

    model_dir = os.path.join(RUN_DIR, RUN_NAME)
    # if 
    # os.mkdir(os.path.join(RUN_DIR, RUN_NAME))
    train_log = runner.train(n_iter=20000, patience=5000, save_path=os.path.join(model_dir, 'model.ckpt'), verbose=True)

    # Save results
    import pandas as pd
    train_log = pd.DataFrame(train_log)
    train_log.to_csv(os.path.join(model_dir, 'train_log.csv'))

    checkpoint = torch.load(os.path.join(model_dir, "model.ckpt"))
    runner.model.load_state_dict(checkpoint['state_dict'])

    training_input = torch.Tensor(
    np.concatenate([
        train_dict['train_spikes_heldin'], 
        np.zeros(train_dict['train_spikes_heldin_forward'].shape), # zeroed inputs for forecasting
    ], axis=1))

    training_output = torch.Tensor(
        np.concatenate([
            np.concatenate([
                train_dict['train_spikes_heldin'],
                train_dict['train_spikes_heldin_forward'],
            ], axis=1),
            np.concatenate([
                train_dict['train_spikes_heldout'],
                train_dict['train_spikes_heldout_forward'],
            ], axis=1),
        ], axis=2))

    eval_input = torch.Tensor(
    np.concatenate([
        eval_dict['eval_spikes_heldin'],
        np.zeros((
            eval_dict['eval_spikes_heldin'].shape[0],
            train_dict['train_spikes_heldin_forward'].shape[1],
            eval_dict['eval_spikes_heldin'].shape[2]
        )),
    ], axis=1))

    torch.save(runner.model, 'model.pt')
    runner.model.eval()
    training_predictions = runner.model(training_input).cpu().detach().numpy()
    eval_predictions = runner.model(eval_input).cpu().detach().numpy()

    tlen = train_dict['train_spikes_heldin'].shape[1]
    num_heldin = train_dict['train_spikes_heldin'].shape[2]

    submission = {
        dataset_name: {
            'train_rates_heldin': training_predictions[:, :tlen, :num_heldin],
            'train_rates_heldout': training_predictions[:, :tlen, num_heldin:],
            'eval_rates_heldin': eval_predictions[:, :tlen, :num_heldin],
            'eval_rates_heldout': eval_predictions[:, :tlen, num_heldin:],
            'eval_rates_heldin_forward': eval_predictions[:, tlen:, :num_heldin],
            'eval_rates_heldout_forward': eval_predictions[:, tlen:, num_heldin:]
        }
    }

    save_to_h5(submission, 'submission.h5')

if __name__ == "__main__":
    pass