#----> internal imports
from datasets.dataset import PatchDatasetFactory
from utils.process_args import process_args
from utils.utils import create_results_dir, get_custom_exp_code, print_and_log_experiment, seed_torch
from utils.core_utils import train_val_test
from utils.file_utils import save_pkl

#----> pytorch imports
import torch

#----> general imports
import pandas as pd
import os
from timeit import default_timer as timer

#----> main
def main(args):
    results = []

    for fold_id in range(5):  # 5-fold cross validation 
        train_dataset, val_dataset, test_dataset = args.dataset_factory.return_splits(
            fold_id,
        )
        
        total_val_loss, total_test_loss  = train_val_test(train_dataset, val_dataset, test_dataset, args, fold_id)
        print(total_val_loss)
        print(total_test_loss)
        fold_results = results.append({
            "val": total_val_loss,
            "test": total_test_loss,
        })

        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_results_{}.pkl'.format(fold_id))
        save_pkl(filename, fold_results)
    
    # write summary of fold results to csv
    df = pd.DataFrame.from_records(results)
    filename = os.path.join(args.results_dir, 'summary.csv')
    df.to_csv(filename)


#----> call main
if __name__ == "__main__":
    start = timer()

    #----> args
    args = process_args()

    #----> Prep
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.device = device
    args = get_custom_exp_code(args)
    seed_torch(args.seed)

    settings = {
                'data_root_dir': args.data_root_dir, 
                'results_dir': args.results_dir, 
                'max_epochs': args.max_epochs, 
                'lr': args.lr,
                'seed': args.seed,
                'drop_out': args.drop_out,
                'weighted_sample': args.weighted_sample,
                'opt': args.opt,
                'batch_size': args.batch_size,
                'model_type': args.model_type,
                'early_stopping': args.early_stopping,
                }

    # #----> Outputs
    create_results_dir(args)

    #---> split dir
    assert os.path.isdir(args.split_dir)
    print('split_dir: ', args.split_dir)
    settings.update({'split_dir': args.split_dir})

    assert os.path.isfile(args.csv_fpath)
    print('csv_path: ', args.csv_fpath)
    settings.update({'csv_path': args.csv_fpath})

    #----> create dataset factory (process omics and WSI to create graph)
    args.dataset_factory = PatchDatasetFactory(
        data_dir=args.data_root_dir,
        csv_path=args.csv_fpath,
        split_dir=args.split_dir,
        seed = args.seed, 
        print_info = True
        )
    
    print_and_log_experiment(args, settings)

    results = main(args)
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))
