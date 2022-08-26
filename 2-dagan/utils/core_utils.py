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
from models.discriminator import DiscriminatorMLP
from models.generator import GeneratorMLP
from sksurv.metrics import concordance_index_censored

def step(cur, args, loss_fns, models, optimizers, train_loader, val_loader, test_loader, early_stopping):
    
    for epoch in range(args.max_epochs):
        train_loop(epoch, cur, models, train_loader, optimizers, loss_fns)
        stop = validate(cur, epoch, models, val_loader, early_stopping, loss_fns, args.results_dir)
        if stop: 
            break

    if args.early_stopping:
        models.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(models.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    total_val_loss = summary(models, args.model_type, val_loader, loss_fns)
    print("Final val losses: {}, loss_D_real: {:.3f}, loss_D_fake: {:.3f}, loss_G_GAN: {:.3f}, loss_G_L1: {:.3f}".format(epoch, total_val_loss["loss_D_real"], total_val_loss["loss_D_fake"], total_val_loss["loss_G_GAN"], total_val_loss["loss_G_L1"]))

    total_test_loss = summary(models, args.model_type, test_loader, loss_fns)
    print("Final test losses: {}, loss_D_real: {:.3f}, train_loss_D_fake: {:.3f}, train_loss_G_GAN: {:.3f}, train_loss_G_L1: {:.3f}".format(epoch, total_test_loss["loss_D_real"], total_test_loss["loss_D_fake"], total_test_loss["loss_G_GAN"], total_test_loss["loss_G_L1"]))

    mlflow.log_metric("final_val_fold{}".format(cur), total_val_loss)
    mlflow.log_metric("final_test_fold{}".format(cur), total_test_loss)
    return None, None, None


# helper functions
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
    train_loader = get_split_loader(args, train_split, training=True, testing = args.testing, batch_size = args.batch_size)
    val_loader = get_split_loader(args, val_split,  testing = args.testing, batch_size = args.batch_size)
    test_loader = get_split_loader(args, test_split, testing = args.testing, batch_size = args.batch_size)
    print('Done!')
    return train_loader,val_loader,test_loader

def init_optims(args, models):
    print('\nInit optimizers...', end=' ')
    optimizers = {
        "optim_G": torch.optim.Adam(models.netG.parameters(), lr=args.learning_rate, betas=(0.9, 0.999)),
        "optim_D": torch.optim.Adam(models.netD.parameters(), lr=args.learning_rate, betas=(0.9, 0.999)),
    }
    print('Done!')

    return optimizers

def init_models(args):
    print('\nInit models...', end=' ')
    models = {
        "net_G": GeneratorMLP(n_tokens=1024, dropout=0.),
        "net_D": DiscriminatorMLP(n_tokens=1024, dropout=0.),
    }
    print_network(args.results_dir, models.net_G)
    print_network(args.results_dir, models.net_D)

    return models

def init_loss_functions(args):
    print('\nInit loss functions...', end=' ')
    loss_fns = {
        "loss_GAN": nn.BCEWithLogitsLoss(),
        "loss_L1": torch.nn.L1Loss()
    }

    return loss_fns

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


# GAN stuff
def forward(net_G, real_A):
    """Run forward pass; generate fake data from real input data"""
    return net_G(real_A)  # G(A)

def calculate_losses_G(net_D, loss_fns, real_A, real_B, fake_B, lambda_L1 = 100):
    """Calculate GAN and L1 loss for the generator"""
    # First, G(A) should fake the discriminator
    fake_AB = torch.cat((real_A, fake_B), 1)
    pred_fake = net_D(fake_AB)
    loss_G_GAN = loss_fns.loss_GAN(pred_fake, True)
    # Second, G(A) = B
    loss_G_L1 = loss_fns.loss_L1(fake_B, real_B) * lambda_L1
    
    return loss_G_GAN, loss_G_L1

def calculate_losses_D(net_D, loss_fns, real_A, real_B, fake_B):
    """Calculate GAN loss for the discriminator"""
    # Fake; stop backprop to the generator by detaching fake_B
    fake_AB = torch.cat((real_A, fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
    pred_fake = net_D(fake_AB.detach())
    loss_D_fake = loss_fns.loss_GAN(pred_fake, False)
    # Real
    real_AB = torch.cat((real_A, real_B), 1)
    pred_real = net_D(real_AB)
    loss_D_real = loss_fns.loss_GAN(pred_real, True)

    return loss_D_real, loss_D_fake


# train, val, test
def train_loop(epoch, cur, models, loader, optimizers, loss_fns):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models.net_G.train().to(device)
    models.net_D.train().to(device)

    total_loss = {
        "loss_D_real": 0.,
        "loss_D_fake": 0.,
        "loss_G_GAN": 0.,
        "loss_G_L1": 0.,
    }
    
    for batch_idx, data in enumerate(loader):
        original, augmentation = data     # split data

        fake_augmentation = forward(models.net_G, original)                   # compute fake images: G(A)

        # update D
        # model.set_requires_grad(models.net_D, True)  # enable backprop for D
        optimizers.optim_D.zero_grad()     # set D's gradients to zero
        loss_D_real, loss_D_fake = calculate_losses_D(models.net_D, loss_fns, original, augmentation, fake_augmentation)                # calculate gradients for D
        # combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward()
        optimizers.optim_D.step()          # update D's weights

        # update G
        # model.set_requires_grad(models.net_D, False)  # D requires no gradients when optimizing G
        optimizers.optim_G.zero_grad()        # set G's gradients to zero
        loss_G_GAN, loss_G_L1 = calculate_losses_G(models.net_D, loss_fns, original, augmentation, fake_augmentation)                   # calculate graidents for G
        # combine loss and calculate gradients
        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()

        optimizers.optim_G.step()             # udpate G's weights

        total_loss["loss_D_real"] += loss_D_real
        total_loss["loss_D_fake"] += loss_D_fake
        total_loss["loss_G_GAN"] += loss_G_GAN
        total_loss["loss_G_L1"] += loss_G_L1

        if (batch_idx % 20) == 0:
            print("batch: {}, loss_D_real: {:.3f}, loss_D_fake: {:.3f}, loss_G_GAN: {:.3f}, loss_G_L1: {:.3f}".format(batch_idx, total_loss["loss_D_real"], total_loss["loss_D_fake"], total_loss["loss_G_GAN"], total_loss["loss_G_L1"]))

    total_loss["loss_D_real"] /= len(loader)
    total_loss["loss_D_fake"] /= len(loader)
    total_loss["loss_G_GAN"] /= len(loader)
    total_loss["loss_G_L1"] /= len(loader)

    print("Epoch: {}, train_loss_D_real: {:.3f}, train_loss_D_fake: {:.3f}, train_loss_G_GAN: {:.3f}, train_loss_G_L1: {:.3f}".format(epoch, total_loss["loss_D_real"], total_loss["loss_D_fake"], total_loss["loss_G_GAN"], total_loss["loss_G_L1"]))

    mlflow.log_metric("train_loss_D_real_fold{}".format(cur), total_loss["loss_D_real"])
    mlflow.log_metric("train_loss_D_fake_fold{}".format(cur), total_loss["loss_D_fake"])
    mlflow.log_metric("train_loss_G_GAN_fold{}".format(cur), total_loss["loss_G_GAN"])
    mlflow.log_metric("train_loss_G_L1_fold{}".format(cur), total_loss["loss_G_L1"])

    return total_loss

def validate(cur, epoch, models, loader, early_stopping, loss_fns = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models.net_G.eval().to(device)
    models.net_D.eval().to(device)

    total_loss = {
        "loss_D_real": 0.,
        "loss_D_fake": 0.,
        "loss_G_GAN": 0.,
        "loss_G_L1": 0.,
    }
    
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            original, augmentation = data     # split data

            fake_augmentation = forward(models.net_G, original)                   # compute fake images: G(A)

            # calculate losses
            loss_D_real, loss_D_fake = calculate_losses_D(models.net_D, loss_fns, original, augmentation, fake_augmentation)
            loss_G_GAN, loss_G_L1 = calculate_losses_G(models.net_D, loss_fns, original, augmentation, fake_augmentation)

            total_loss["loss_D_real"] += loss_D_real
            total_loss["loss_D_fake"] += loss_D_fake
            total_loss["loss_G_GAN"] += loss_G_GAN
            total_loss["loss_G_L1"] += loss_G_L1

    total_loss["loss_D_real"] /= len(loader)
    total_loss["loss_D_fake"] /= len(loader)
    total_loss["loss_G_GAN"] /= len(loader)
    total_loss["loss_G_L1"] /= len(loader)

    print("Epoch: {}, val_loss_D_real: {:.3f}, val_loss_D_fake: {:.3f}, val_loss_G_GAN: {:.3f}, val_loss_G_L1: {:.3f}".format(epoch, total_loss["loss_D_real"], total_loss["loss_D_fake"], total_loss["loss_G_GAN"], total_loss["loss_G_L1"]))

    mlflow.log_metric("val_loss_D_real_fold{}".format(cur), total_loss["loss_D_real"])
    mlflow.log_metric("val_loss_D_fake_fold{}".format(cur), total_loss["loss_D_fake"])
    mlflow.log_metric("val_loss_G_GAN_fold{}".format(cur), total_loss["loss_G_GAN"])
    mlflow.log_metric("val_loss_G_L1_fold{}".format(cur), total_loss["loss_G_L1"])

    if early_stopping:
        assert results_dir
        early_stopping(epoch, total_loss, models, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(models, loader, loss_fns):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models.net_G.eval().to(device)
    models.net_D.eval().to(device)

    total_loss = {
        "loss_D_real": 0.,
        "loss_D_fake": 0.,
        "loss_G_GAN": 0.,
        "loss_G_L1": 0.,
    }
    
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            original, augmentation = data     # split data

            fake_augmentation = forward(models.net_G, original)                   # compute fake images: G(A)

            # calculate losses
            loss_D_real, loss_D_fake = calculate_losses_D(models.net_D, loss_fns, original, augmentation, fake_augmentation)
            loss_G_GAN, loss_G_L1 = calculate_losses_G(models.net_D, loss_fns, original, augmentation, fake_augmentation)

            total_loss["loss_D_real"] += loss_D_real
            total_loss["loss_D_fake"] += loss_D_fake
            total_loss["loss_G_GAN"] += loss_G_GAN
            total_loss["loss_G_L1"] += loss_G_L1

    return total_loss

def train_val_test(datasets, args, cur):
    """   
    Performs train val test for the fold over number of epochs
    """

    #----> gets splits and summarize
    train_split, val_split, test_split = get_splits(datasets, args)
    
    #----> init loss function
    loss_fns = init_loss_functions(args)

    #----> init model
    models = init_models(args)
    
    #---> init optimizer
    optimizers = init_optims(args, models)
    
    #---> init loaders
    train_loader, val_loader, test_loader = init_loaders(args, train_split, val_split, test_split)

    #---> init early stopping
    early_stopping = init_early_stopping(args)

    #---> do train val test
    val_cindex, results_dict, test_cindex = step(cur, args, loss_fns, models, optimizers, train_loader, val_loader, test_loader, early_stopping)

    return None, None, None