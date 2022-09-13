# ----> internal imports
from datasets.dataset import PatchDatasetFactory
from utils.process_args import process_args
from utils.utils import (
    create_results_dir,
    get_custom_exp_code,
    print_and_log_experiment,
    seed_torch,
)
from utils.core_utils import train_val_test
from utils.file_utils import save_pkl

# ----> pytorch imports
import torch

# ----> general imports
import pandas as pd
import os
from timeit import default_timer as timer

import logging

logging.basicConfig(
    # filename="test_logfile.log",
    format="%(asctime)s | %(message)s",
    level=logging.DEBUG,
)

# ----> main
def main(args):
    results = []

    for fold_id in range(5):  # 5-fold cross validation
        train_dataset, val_dataset, test_dataset = args.dataset_factory.return_splits(
            fold_id,
        )

        total_val_loss, total_test_loss = train_val_test(
            train_dataset, val_dataset, test_dataset, args, fold_id
        )
        logging.debug(f"Fold {fold_id}, final val loss: {total_val_loss}")
        logging.debug(f"Fold {fold_id}, final test loss: {total_test_loss}")
        fold_results = results.append(
            {
                "val": total_val_loss,
                "test": total_test_loss,
            }
        )

        # write results to pkl
        filename = os.path.join(
            args.results_dir, "split_results_{}.pkl".format(fold_id)
        )
        save_pkl(filename, fold_results)

    # write summary of fold results to csv
    df = pd.DataFrame.from_records(results)
    filename = os.path.join(args.results_dir, "summary.csv")
    df.to_csv(filename)


# ----> call main
if __name__ == "__main__":

    start = timer()

    # ----> args
    args = process_args()

    # ----> Prep
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.device = device
    args = get_custom_exp_code(args)
    seed_torch(args.seed)

    settings = {
        "data_root_dir": args.data_root_dir,
        "split_dir": args.split_dir,
        "csv_path": args.csv_fpath,
        "results_dir": args.results_dir,
        "model_type": args.model_type,
        "n_heads": args.n_heads,
        "emb_dim": args.emb_dim,
        "reg_type": args.reg_type,
        "max_epochs": args.max_epochs,
        "lr": args.lr,
        "seed": args.seed,
        "drop_out": args.drop_out,
        "weighted_sample": args.weighted_sample,
        "opt": args.opt,
        "batch_size": args.batch_size,
        "early_stopping": args.early_stopping,
    }

    # ---> Make sure directories/files exist
    assert os.path.isdir(args.data_root_dir)
    assert os.path.isdir(args.split_dir)
    assert os.path.isfile(args.csv_fpath)

    # ----> Outputs
    create_results_dir(args)

    # ----> create dataset factory (process omics and WSI to create graph)
    args.dataset_factory = PatchDatasetFactory(
        data_dir=args.data_root_dir,
        csv_path=args.csv_fpath,
        split_dir=args.split_dir,
        seed=args.seed,
        print_info=True,
    )

    print_and_log_experiment(args, settings)

    results = main(args)
    end = timer()
    # logging.info("Finished!")
    logging.info("Script Time: %f seconds" % (end - start))
    print(args.param_code)
