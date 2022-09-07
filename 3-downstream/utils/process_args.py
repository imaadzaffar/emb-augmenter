import argparse

def process_args():

    parser = argparse.ArgumentParser(description='Configurations for WSI Survival Training')

    #----> data/ splits args
    parser.add_argument('--data_root_dir', type=str, default=None, help='data directory')
    parser.add_argument('--csv_fpath', type=str, default=None, help='CSV with labels')
    parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
    parser.add_argument('--split_dir', type=str, default=None, help='manually specify the set of splits to use, ' 
                        +'instead of infering from the task and label_frac argument (default: None)')

    #----> training args
    parser.add_argument('--max_epochs', type=int, default=2, help='maximum number of epochs to train (default: 2)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
    parser.add_argument('--opt', type=str, default="adam", choices=["adam", "sgd",  "adamW", "radam"], help="Optimizer")
    parser.add_argument('--reg_type', type=str, default=None, choices=[None, "L1", "L2"], help="regularization type [None, L1, L2]")
    parser.add_argument('--drop_out', type=int, default=0., help='dropout value for model')
    parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')

    parser.add_argument('--num_runs', type=int, default=1, help='number of times to repeat experiment')
    parser.add_argument('--augmentation', type=str, default=None, choices=[None, "combined", "rotation", "hue", "sat", "value", "zoom"], help='augmentation type')

    # TODO: add dagan settings args
    parser.add_argument('--dagan', action='store_true', default=False, help='Enable using DA-GAN as source for generated augmentations in training')

    parser.add_argument('--mlflow_exp_name', type=str, default='ABMIL', help='ABMIL, whatever to be created')

    args = parser.parse_args()

    return args