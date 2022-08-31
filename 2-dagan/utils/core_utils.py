#----> internal imports
from configparser import NoSectionError
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
from models.generator import GeneratorMLP, GeneratorTransformer
from models.discriminator import DiscriminatorMLP, DiscriminatorTransformer
from sksurv.metrics import concordance_index_censored

def step(cur, args, loss_fns, models, optimizers, train_loader, val_loader, test_loader, early_stopping):
    
    for epoch in range(args.max_epochs):
        train_loop(epoch, cur, models, train_loader, optimizers, loss_fns)
        stop = validate(cur, epoch, models, val_loader, early_stopping, loss_fns, args.results_dir)
        if stop: 
            break

    if args.early_stopping:
        models_state_dict = torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
        models["net_G"].load_state_dict(models_state_dict["G_state_dict"])
        models["net_D"].load_state_dict(models_state_dict["D_state_dict"])
    else:
        torch.save({"G_state_dict": models["net_G"].state_dict(), "D_state_dict": models["net_D"].state_dict()}, os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    # total_val_loss = summary(models, val_loader, loss_fns)
    # print("Final val losses: {}, loss_D_real: {:.3f}, loss_D_fake: {:.3f}, loss_G_GAN: {:.3f}, loss_G_L1: {:.3f}".format(epoch, total_val_loss["loss_D_real"], total_val_loss["loss_D_fake"], total_val_loss["loss_G_GAN"], total_val_loss["loss_G_L1"]))

    # total_test_loss = summary(models, test_loader, loss_fns)
    # print("Final test losses: {}, loss_D_real: {:.3f}, train_loss_D_fake: {:.3f}, train_loss_G_GAN: {:.3f}, train_loss_G_L1: {:.3f}".format(epoch, total_test_loss["loss_D_real"], total_test_loss["loss_D_fake"], total_test_loss["loss_G_GAN"], total_test_loss["loss_G_L1"]))

    # mlflow.log_metric("fold{}_final_val_loss_G_GAN".format(cur), total_val_loss["loss_G_GAN"])
    # mlflow.log_metric("fold{}_final_test_loss_G_GAN".format(cur), total_test_loss["loss_G_GAN"])

    return None, None


# helper functions
def init_early_stopping(args):
    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=20, stop_epoch=50, verbose = True)

    else:
        early_stopping = None
    print('Done!')
    return early_stopping

def init_loaders(args, train_split, val_split, test_split):
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(args, train_split, training=True, batch_size = args.batch_size)
    val_loader = get_split_loader(args, val_split,  testing=False, batch_size = args.batch_size)
    test_loader = get_split_loader(args, test_split, testing=False, batch_size = args.batch_size)
    print('Done!')
    return train_loader,val_loader,test_loader

def init_optims(args, models):
    print('\nInit optimizers...', end=' ')
    optimizers = {
        "optim_G": torch.optim.Adam(models["net_G"].parameters(), lr=args.lr, betas=(0.9, 0.999)),
        "optim_D": torch.optim.Adam(models["net_D"].parameters(), lr=args.lr, betas=(0.9, 0.999)),
    }
    print('Done!')

    return optimizers

def init_models(args):
    print('\nInit models...', end=' ')
    if args.model_type == 'MLP':
        models = {
            "net_G": GeneratorMLP(n_tokens=1024, dropout=args.drop_out),
            "net_D": DiscriminatorMLP(n_tokens=1024, dropout=args.drop_out),
        }
    elif args.model_type == 'Transformer':
        models = {
            "net_G": GeneratorTransformer(n_tokens=1024, dropout=args.drop_out),
            "net_D": DiscriminatorTransformer(n_tokens=1024, dropout=args.drop_out),
        }
    else:
        raise ValueError("Invalid model type.")
    
    print_network(args.results_dir, models["net_G"], "G")
    print_network(args.results_dir, models["net_D"], "D")

    return models

def init_loss_functions(args):
    print('\nInit loss functions...', end=' ')
    loss_fns = {
        "loss_GAN": GANLoss(),
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
class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.BCEWithLogitsLoss()

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

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target_tensor = self.get_target_tensor(prediction, target_is_real).to(device)
        # print("loss devices:", prediction.device, target_tensor.device)

        loss = self.loss(prediction, target_tensor)
        return loss

def calculate_losses_G(net_D, loss_fns, real_A, real_B, fake_B, lambda_L1 = 100):
    """Calculate GAN and L1 loss for the generator"""
    # First, G(A) should fake the discriminator
    # we use conditional GANs; we need to feed both input and output to the discriminator
    pred_fake = net_D.forward(real_A, fake_B)
    loss_G_GAN = loss_fns["loss_GAN"](pred_fake, True)

    # Second, G(A) = B
    loss_G_L1 = loss_fns["loss_L1"](fake_B, real_B) * lambda_L1
    
    return loss_G_GAN, loss_G_L1

def calculate_losses_D(net_D, loss_fns, real_A, real_B, fake_B):
    """Calculate GAN loss for the discriminator"""
    # Fake; stop backprop to the generator by detaching fake_B
    # we use conditional GANs; we need to feed both input and output to the discriminator
    pred_fake = net_D.forward(real_A, fake_B)
    loss_D_fake = loss_fns["loss_GAN"](pred_fake, False)

    # Real
    pred_real = net_D(real_A, real_B)
    loss_D_real = loss_fns["loss_GAN"](pred_real, True)

    return loss_D_real, loss_D_fake


def set_requires_grad(net, requires_grad=False):
    """Sets requires_grad for all the networks to avoid unnecessary computations"""
    if net is not None:
        for param in net.parameters():
            param.requires_grad = requires_grad


# train, val, test
def train_loop(epoch, cur, models, loader, optimizers, loss_fns):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models["net_G"].train().to(device)
    models["net_D"].train().to(device)

    total_loss = {
        "loss_D_real": 0.,
        "loss_D_fake": 0.,
        "loss_G_GAN": 0.,
        "loss_G_L1": 0.,
    }
    
    for batch_idx, data in enumerate(loader):
        original, augmentation, noise = data     # split data
        # print("devices before:", original.device, augmentation.device, noise.device)

        # move tensors to cuda
        original, augmentation, noise = original.to(device), augmentation.to(device), noise.to(device)
        # print("devices after:", original.device, augmentation.device, noise.device)

        # print("ORIGINAL:", original.shape)
        # print("AUGMENTATION:", augmentation.shape)
        # print("NOISE:", noise.shape)
        fake_augmentation = models["net_G"].forward(original, noise)                   # compute fake images: G(A)

        # update D
        set_requires_grad(models["net_D"], True)  # enable backprop for D
        optimizers["optim_D"].zero_grad()     # set D's gradients to zero
        loss_D_real, loss_D_fake = calculate_losses_D(models["net_D"], loss_fns, original, augmentation, fake_augmentation)                # calculate gradients for D
        # combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward(retain_graph=True)
        optimizers["optim_D"].step()          # update D's weights

        # update G
        set_requires_grad(models["net_D"], False)  # D requires no gradients when optimizing G
        optimizers["optim_G"].zero_grad()        # set G's gradients to zero
        loss_G_GAN, loss_G_L1 = calculate_losses_G(models["net_D"], loss_fns, original, augmentation, fake_augmentation)                   # calculate gradients for G
        # combine loss and calculate gradients
        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward(retain_graph=True)

        optimizers["optim_G"].step()             # update G's weights

        total_loss["loss_D_real"] += loss_D_real.item()
        total_loss["loss_D_fake"] += loss_D_fake.item()
        total_loss["loss_G_GAN"] += loss_G_GAN.item()
        total_loss["loss_G_L1"] += loss_G_L1.item()

        # if (batch_idx % 20) == 0:
        #     print("batch: {}, loss_D_real: {:.3f}, loss_D_fake: {:.3f}, loss_G_GAN: {:.3f}, loss_G_L1: {:.3f}".format(batch_idx, total_loss["loss_D_real"], total_loss["loss_D_fake"], total_loss["loss_G_GAN"], total_loss["loss_G_L1"]))

    total_loss["loss_D_real"] /= len(loader)
    total_loss["loss_D_fake"] /= len(loader)
    total_loss["loss_G_GAN"] /= len(loader)
    total_loss["loss_G_L1"] /= len(loader)

    print(f"Fold: {cur}, Epoch: {epoch}, train_loss_D_real: {total_loss['loss_D_real']:.3f}, train_loss_D_fake: {total_loss['loss_D_fake']:.3f}, train_loss_G_GAN: {total_loss['loss_G_GAN']:.3f}, train_loss_G_L1: {total_loss['loss_G_L1']:.3f}")

    for key, loss in total_loss.items():
        mlflow.log_metric(key=f"fold{cur}_train_{key}", value=loss, step=epoch)

    return total_loss

def validate(cur, epoch, models, loader, early_stopping, loss_fns = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models["net_G"].eval().to(device)
    models["net_D"].eval().to(device)

    total_loss = {
        "loss_D_real": 0.,
        "loss_D_fake": 0.,
        "loss_G_GAN": 0.,
        "loss_G_L1": 0.,
    }
    
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            original, augmentation, noise = data     # split data
            # print("devices before:", original.device, augmentation.device, noise.device)

            # move tensors to cuda
            original, augmentation, noise = original.to(device), augmentation.to(device), noise.to(device)
            # print("devices after:", original.device, augmentation.device, noise.device)

            fake_augmentation = models["net_G"].forward(original, noise)                   # compute fake images: G(A)

            # calculate losses
            loss_D_real, loss_D_fake = calculate_losses_D(models["net_D"], loss_fns, original, augmentation, fake_augmentation)
            loss_G_GAN, loss_G_L1 = calculate_losses_G(models["net_D"], loss_fns, original, augmentation, fake_augmentation)

            total_loss["loss_D_real"] += loss_D_real.item()
            total_loss["loss_D_fake"] += loss_D_fake.item()
            total_loss["loss_G_GAN"] += loss_G_GAN.item()
            total_loss["loss_G_L1"] += loss_G_L1.item()

    total_loss["loss_D_real"] /= len(loader)
    total_loss["loss_D_fake"] /= len(loader)
    total_loss["loss_G_GAN"] /= len(loader)
    total_loss["loss_G_L1"] /= len(loader)

    print(f"Fold: {cur}, Epoch: {epoch}, val_loss_D_real: {total_loss['loss_D_real']:.3f}, val_loss_D_fake: {total_loss['loss_D_fake']:.3f}, val_loss_G_GAN: {total_loss['loss_G_GAN']:.3f}, val_loss_G_L1: {total_loss['loss_G_L1']:.3f}")

    for key, loss in total_loss.items():
        mlflow.log_metric(key=f"fold{cur}_val_{key}", value=loss, step=epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, total_loss, models, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(models, loader, loss_fns):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models["net_G"].eval().to(device)
    models["net_D"].eval().to(device)

    total_loss = {
        "loss_D_real": 0.,
        "loss_D_fake": 0.,
        "loss_G_GAN": 0.,
        "loss_G_L1": 0.,
    }
    
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            original, augmentation, noise = data     # split data
            # print("devices before:", original.device, augmentation.device, noise.device)

            # move tensors to cuda
            original, augmentation, noise = original.to(device), augmentation.to(device), noise.to(device)
            # print("devices after:", original.device, augmentation.device, noise.device)   

            fake_augmentation = models["net_G"].forward(original, noise)                   # compute fake images: G(A)

            # calculate losses
            loss_D_real, loss_D_fake = calculate_losses_D(models["net_D"], loss_fns, original, augmentation, fake_augmentation)
            loss_G_GAN, loss_G_L1 = calculate_losses_G(models["net_D"], loss_fns, original, augmentation, fake_augmentation)

            total_loss["loss_D_real"] += loss_D_real.item()
            total_loss["loss_D_fake"] += loss_D_fake.item()
            total_loss["loss_G_GAN"] += loss_G_GAN.item()
            total_loss["loss_G_L1"] += loss_G_L1.item()

    return total_loss

def train_val_test(train_split, val_split, test_split, args, cur):
    """   
    Performs train val test for the fold over number of epochs
    """

    #----> gets splits and summarize
    # train_split, val_split, test_split = get_splits(datasets, args)
    
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
    total_val_loss, total_test_loss = step(cur, args, loss_fns, models, optimizers, train_loader, val_loader, test_loader, early_stopping)

    return total_val_loss, total_test_loss