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
from sksurv.metrics import concordance_index_censored

def step(cur, args, losses, models, optimizers, train_loader, val_loader, test_loader, early_stopping):
    
    for epoch in range(args.max_epochs):
        train_loop(epoch, cur, models, train_loader, optimizers, losses)
        stop = validate(cur, epoch, models, val_loader, early_stopping, losses, args.results_dir)
        if stop: 
            break

    if args.early_stopping:
        models.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(models.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_metric = summary(models, args.model_type, val_loader, losses)
    print('Final Val metric: {:.4f}'.format(val_metric))

    results_dict, test_metric, = summary(models, args.model_type, test_loader, losses)
    print('Final Test metric: {:.4f}'.format(test_metric))

    mlflow.log_metric("final_val_fold{}".format(cur), val_metric)
    mlflow.log_metric("final_test_fold{}".format(cur), test_metric)
    return val_metric, results_dict, test_metric


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
    train_loader = get_split_loader(args, train_split, training=True, testing = args.testing, weighted = args.weighted_sample, batch_size = args.batch_size)
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
    # @TODO: add models for generator and discriminator
    models = {
        "net_G": None,
        "net_D": None,
    }
    print_network(args.results_dir, models.net_G)
    print_network(args.results_dir, models.net_D)

    return models

def init_loss_functions(args):
    print('\nInit loss functions...', end=' ')
    losses = {
        "loss_GAN": nn.BCEWithLogitsLoss(),
        "loss_L1": torch.nn.L1Loss()
    }

    return losses

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
def get_target_tensor(self, prediction, target_is_real):
    """Create label tensors with the same size as the input.
    Parameters:
        prediction (tensor) - - tpyically the prediction from a discriminator
        target_is_real (bool) - - if the ground truth label is for real images or fake images
    Returns:
        A label tensor filled with ground truth label, and with the size of the input
    """

    if target_is_real:
        target_tensor = self.real_label
    else:
        target_tensor = self.fake_label
    return target_tensor.expand_as(prediction)

def forward(net_G, real_A):
    """Run forward pass; generate fake data from real input data"""
    return net_G(real_A)  # G(A)

def backward_G(net_D, losses, real_A, real_B, fake_B, lambda_L1 = 100):
    """Calculate GAN and L1 loss for the generator"""
    # First, G(A) should fake the discriminator
    fake_AB = torch.cat((real_A, fake_B), 1)
    pred_fake = net_D(fake_AB)
    loss_G_GAN = losses.loss_GAN(pred_fake, True)
    # Second, G(A) = B
    loss_G_L1 = losses.loss_L1(fake_B, real_B) * lambda_L1
    # combine loss and calculate gradients
    loss_G = loss_G_GAN + loss_G_L1
    loss_G.backward()

def backward_D(net_D, losses, real_A, real_B, fake_B):
    """Calculate GAN loss for the discriminator"""
    # Fake; stop backprop to the generator by detaching fake_B
    fake_AB = torch.cat((real_A, fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
    pred_fake = net_D(fake_AB.detach())
    loss_D_fake = losses.loss_GAN(pred_fake, False)
    # Real
    real_AB = torch.cat((real_A, real_B), 1)
    pred_real = net_D(real_AB)
    loss_D_real = losses.loss_GAN(pred_real, True)
    # combine loss and calculate gradients
    loss_D = (loss_D_fake + loss_D_real) * 0.5
    loss_D.backward()


# train, val, test
def train_loop(epoch, cur, models, loader, optimizers, losses):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.train()

    # @TODO: Sum losses
    total_loss = 0.
    
    for batch_idx, data in enumerate(loader):
        # @TODO: GAN training goes here. 

        original, augmentation = data     # split data

        fake_augmentation = forward(models.net_G, original)                   # compute fake images: G(A)

        # update D
        # model.set_requires_grad(models.net_D, True)  # enable backprop for D
        optimizers.optim_D.zero_grad()     # set D's gradients to zero
        backward_D(models.net_D, losses, original, augmentation, fake_augmentation)                # calculate gradients for D
        optimizers.optim_D.step()          # update D's weights

        # update G
        # model.set_requires_grad(models.net_D, False)  # D requires no gradients when optimizing G
        optimizers.optim_G.zero_grad()        # set G's gradients to zero
        backward_G(models.net_D, losses, original, augmentation, fake_augmentation)                   # calculate graidents for G
        optimizers.optim_G.step()             # udpate G's weights

        if (batch_idx % 20) == 0:
            losses = 0.  # get current losses
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
    losses = init_loss_functions(args)

    #----> init model
    models = init_models(args)
    
    #---> init optimizer
    optimizers = init_optims(args, models)
    
    #---> init loaders
    train_loader, val_loader, test_loader = init_loaders(args, train_split, val_split, test_split)

    #---> init early stopping
    early_stopping = init_early_stopping(args)

    #---> do train val test
    val_cindex, results_dict, test_cindex = step(cur, args, losses, models, optimizers, train_loader, val_loader, test_loader, early_stopping)

    return results_dict, test_cindex, val_cindex