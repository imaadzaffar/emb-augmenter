import argparse

def process_args():

    parser = argparse.ArgumentParser(description='Configurations for GAN Augmented Embeddings')

    #----> data/ splits args
    parser.add_argument('--data_root_dir', type=str, default=None, help='data directory')
    parser.add_argument('--csv_fpath', type=str, default=None, help='CSV with labels')
    parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
    parser.add_argument('--split_dir', type=str, default=None, help='manually specify the set of splits to use, ' 
                        +'instead of infering from the task and label_frac argument (default: None)')

    #----> training args
    parser.add_argument('--max_epochs', type=int, default=20, help='maximum number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
    parser.add_argument('--opt', type=str, default="adam", help="optimizer for models")
    parser.add_argument('--reg_type', type=str, default="None", help="regularization type [None, L1, L2]")
    parser.add_argument('--drop_out', type=float, default=0., help='dropout for models')
    parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--model_type', type=str, default="MLP", help="type of models used [MLP, Transformer]")

    parser.add_argument('--mlflow_exp_name', type=str, default='GAN', help='GAN, whatever to be created')

    args = parser.parse_args()

    return args