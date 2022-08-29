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
        
        fold_results  = train_val_test(train_dataset, val_dataset, test_dataset, args, fold_id)
        print(fold_results)
        results.append(fold_results)

        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_results.pkl')
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
    args.split_dir = "splits"
    args = get_custom_exp_code(args)
    seed_torch(args.seed)

    settings = {'num_splits': args.k, 
                'k_start': args.k_start,
                'k_end': args.k_end,
                'task': args.task,
                'max_epochs': args.max_epochs, 
                'results_dir': args.results_dir, 
                'lr': args.lr,
                'experiment': args.study,
                'reg': args.reg,
                'label_frac': args.label_frac,
                'bag_loss': args.bag_loss,
                'seed': args.seed,
                'model_type': args.model_type,
                "use_drop_out": args.drop_out,
                'weighted_sample': args.weighted_sample,
                'opt': args.opt}

    # #----> Outputs
    create_results_dir(args)

    #---> split dir
    assert os.path.isdir(args.split_dir)
    print('split_dir: ', args.split_dir)
    settings.update({'split_dir': args.split_dir})

    #----> create dataset factory (process omics and WSI to create graph)
    args.dataset_factory = PatchDatasetFactory(
        data_dir=args.data_dir,
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
