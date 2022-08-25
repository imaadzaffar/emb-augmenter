#----> internal imports
from inspect import trace
# from datasets.datasets import save_splits
from utils.utils import EarlyStopping, get_optim, get_split_loader, print_network

#----> pytorch imports
import torch
import torch.nn as nn 

#----> general imports
import numpy as np
import mlflow 
import os
from models.dagan import Pix2PixModel
from sksurv.metrics import concordance_index_censored

def step(cur, args, loss_fn, model, optimizer, train_loader, val_loader, test_loader, early_stopping):
    
    for epoch in range(args.max_epochs):
        train_loop(epoch, cur, model, train_loader, optimizer, loss_fn)
        stop = validate(cur, epoch, model, val_loader, early_stopping, loss_fn, args.results_dir)
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_metric = summary(model, args.model_type, val_loader, loss_fn)
    print('Final Val metric: {:.4f}'.format(val_metric))

    results_dict, test_metric, = summary(model, args.model_type, test_loader, loss_fn)
    print('Final Test metric: {:.4f}'.format(test_metric))

    mlflow.log_metric("final_val_fold{}".format(cur), val_metric)
    mlflow.log_metric("final_test_fold{}".format(cur), test_metric)
    return val_metric, results_dict, test_metric

def init_early_stopping(args):
    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)

    else:
        early_stopping = None
    print('Done!')
    return early_stopping

def init_loaders(args, train_split, val_split, test_split):
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(args, train_split, training=True, testing = args.testing, weighted = args.weighted_sample, batch_size = args.batch_size)
    val_loader = get_split_loader(args, val_split,  testing = args.testing, batch_size = args.batch_size)
    test_loader = get_split_loader(args, test_split, testing = args.testing, batch_size = args.batch_size)
    print('Done!')
    return train_loader,val_loader,test_loader

def init_optim(args, model):
    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    return optimizer

def init_model(args):
    print('\nInit Model...', end=' ')
    model = Pix2PixModel(args)
    model = model.to(torch.device('cuda'))
    print_network(args.results_dir, model)
    return model

def init_loss_function(args):
    print('\nInit loss function...', end=' ')
    # @TODO: define loss function. 
    loss_fn = nn.BCELoss()
    return loss_fn

def get_splits(datasets, cur, args):
    print('\nTraining Fold {}!'.format(cur))
    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    # save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))
    return train_split,val_split,test_split 

def train_loop(epoch, cur, model, loader, optimizer, loss_fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    # @TODO: Sum losses
    total_loss = 0.
    
    for batch_idx, data in enumerate(loader):

        # @TODO: GAN training goes here. 
        original, augmentation = data
        patch_embs = patch_embs.squeeze().to(device)
        label = label.to(device)

        model.set_input()
        model.optimize_parameters()

        if (batch_idx % 20) == 0:
            losses = model.get_current_losses()
            print("batch: {}, loss: {:.3f}".format(batch_idx, losses))

    total_loss /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_metric: {:.4f}'.format(epoch, total_loss, 0.))

    mlflow.log_metric("train_loss_fold{}".format(cur), total_loss)
    mlflow.log_metric("train_cindex_fold{}".format(cur), 0.)

    return 0., total_loss

def validate(cur, epoch, model, loader, early_stopping, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # @TODO: Sum losses
    total_loss = 0.

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            # @TODO: GAN forward pass 
            model.set_input(data)
            model.forward()

            # @TODO: Retrieve loss from model
            loss = loss_fn()
            loss_value = loss.item()
            total_loss += loss_value 

    total_loss /= len(loader)

    print('Epoch: {}, val_loss: {:.4f}, val_c_index: {:.4f}'.format(epoch, total_loss, 0.))

    mlflow.log_metric("val_loss_fold{}".format(cur), total_loss)
    mlflow.log_metric("val_metric_fold{}".format(cur), 0.)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, total_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, loss_fn):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # @TODO: Sum losses
    total_loss = 0.

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            # @TODO: GAN testing 
            model.set_input(data)
            model.test()

            # @TODO: Retrieve loss from model
    return None

def train_val_test(datasets, args, cur):
    """   
    Performs train val test for the fold over number of epochs
    """

    #----> gets splits and summarize
    train_split, val_split, test_split = get_splits(datasets, args)
    
    #----> init loss function
    loss_fn = init_loss_function(args)

    #----> init model
    model = init_model(args)
    
    #---> init optimizer
    optimizer = init_optim(args, model)
    
    #---> init loaders
    train_loader, val_loader, test_loader = init_loaders(args, train_split, val_split, test_split)

    #---> init early stopping
    early_stopping = init_early_stopping(args)

    #---> do train val test
    val_cindex, results_dict, test_cindex = step(cur, args, loss_fn, model, optimizer, train_loader, val_loader, test_loader, early_stopping)

    return results_dict, test_cindex, val_cindex