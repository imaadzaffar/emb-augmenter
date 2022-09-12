# ----> internal imports
from utils.radam import RAdam

# ----> general imports
import torch
import numpy as np
import os

import torch
import numpy as np
from torch.utils.data import (
    DataLoader,
    Sampler,
    WeightedRandomSampler,
    RandomSampler,
    SequentialSampler,
)
import torch.optim as optim
import math
from itertools import islice
import collections
import mlflow
from datetime import datetime

import logging

log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name="checkpoint.pt"):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            log.debug(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        """Saves model when validation loss decrease."""
        if self.verbose:
            log.debug(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
            indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator, n, None), default)


def get_custom_exp_code(args):
    r"""
    Updates the argparse.NameSpace with a custom experiment code.

    Args:
        - args (NameSpace)

    Returns:
        - args (NameSpace)
    """
    dataset_path = "datasets_csv"
    param_code = ""

    # ----> Augmentation
    param_code += f"_{args.augmentation}" if args.augmentation else "_original"

    if args.dagan_run_code is not None:
        param_code += "_dagan"

    # ----> Seed
    param_code += f"_s{args.seed}"

    # ----> Learning Rate
    param_code += f"_lr{format(args.lr, '.0e')}"

    # ----> Batch Size
    param_code += f"_b{args.batch_size}"

    # ----> Time Stamp to make it unique
    param_code += "_" + datetime.now().strftime("%Y%d%m_%H%M%S")

    # ----> Updating
    args.param_code = param_code
    args.dataset_path = dataset_path

    log.debug("PARAM_CODE: " + args.param_code)

    return args


def seed_torch(seed=7):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def print_network(results_dir, net):
    num_params = 0
    num_params_train = 0

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    log.debug("Total number of parameters: %d" % num_params)
    log.debug("Total number of trainable parameters: %d" % num_params_train)

    fname = "model_" + results_dir.split("/")[-1] + ".txt"
    path = os.path.join(results_dir, fname)
    f = open(path, "w")
    f.write(str(net))
    f.write("\n")
    f.write("Total number of parameters: %d \n" % num_params)
    f.write("Total number of trainable parameters: %d \n" % num_params_train)
    f.close()


def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00001)
    elif args.opt == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.00001
        )
    elif args.opt == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.00001)
    elif args.opt == "radam":
        optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=0.00001)
    else:
        raise NotImplementedError

    return optimizer


def get_split_loader(args, split_dataset, training=False, batch_size=1):

    kwargs = {"num_workers": 4} if device.type == "cuda" else {}

    if training:
        loader = DataLoader(
            split_dataset,
            batch_size=batch_size,
            sampler=RandomSampler(split_dataset),
            drop_last=True,
            **kwargs,
        )
    else:
        loader = DataLoader(
            split_dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(split_dataset),
            drop_last=True,
            **kwargs,
        )

    return loader


def calculate_error(Y_hat, Y):
    error = 1.0 - Y_hat.float().eq(Y.float()).float().mean().item()
    return error


def create_results_dir(args):
    # aug_results_dir = args.augmentation if args.augmentation else 'original'
    args.results_dir = os.path.join("./results", args.results_dir)
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
        # ---> add gitignore to results dir
        f = open(os.path.join(args.results_dir, ".gitignore"), "w")
        f.write("*\n")
        f.write("*/\n")
        f.write("!.gitignore")
        f.close()

    # ---> results for this specific experiment
    args.results_dir = os.path.join(args.results_dir, args.param_code)
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)


def print_and_log_experiment(args, settings):
    with open(
        args.results_dir + "/experiment_{}.txt".format(args.param_code), "w"
    ) as f:
        print(settings, file=f)

    f.close()

    log.info("################# Settings ###################")
    for key, val in settings.items():
        log.info("{}:  {}".format(key, val))

    mlflow.set_experiment(args.mlflow_exp_name)
    mlflow.start_run(run_name=args.param_code)

    # ----> keep track of expriment params in mlflow
    argsDict = vars(args)
    for key, value in argsDict.items():
        mlflow.log_param(key, value)
