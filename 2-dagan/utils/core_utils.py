# ----> internal imports
from configparser import NoSectionError
from inspect import trace

# from datasets.datasets import save_splits
from utils.utils import EarlyStopping, get_optim, get_split_loader, print_network

# ----> pytorch imports
import torch
import torch.nn as nn
from torch.autograd import Variable

# ----> general imports
import numpy as np
import mlflow
import os
from models.generator import (
    GeneratorMLP,
    GeneratorTransformer,
    GeneratorIndependent,
    GeneratorIndependentFast,
)
from models.discriminator import (
    DiscriminatorMLP,
    DiscriminatorTransformer,
    DiscriminatorIndependent,
    DiscriminatorIndependentFast,
)
from sksurv.metrics import concordance_index_censored

import logging

log = logging.getLogger(__name__)


def step(
    args,
    loss_fns,
    models,
    optimizers,
    train_loader,
    val_loader,
    test_loader,
    early_stopping,
):

    train_loop(models, train_loader, optimizers, loss_fns)
    stop = validate(models, val_loader, early_stopping, loss_fns, args.results_dir)

    if args.early_stopping:
        models_state_dict = torch.load(os.path.join(args.results_dir, "checkpoint.pt"))
        models["net_G"].load_state_dict(models_state_dict["G_state_dict"])
        models["net_D"].load_state_dict(models_state_dict["D_state_dict"])
    else:
        torch.save(
            {
                "G_state_dict": models["net_G"].state_dict(),
                "D_state_dict": models["net_D"].state_dict(),
            },
            os.path.join(args.results_dir, "checkpoint.pt"),
        )

    # final_val_loss, final_val_e = summary(models, val_loader, loss_fns)
    # log.debug(
    #     "{} | e: {:.3f}, D_real: {:.3f}, D_fake: {:.3f}, G_GAN: {:.3f}".format(
    #         "final val",
    #         final_val_e,
    #         final_val_loss["D_real"],
    #         final_val_loss["D_fake"],
    #         final_val_loss["G_GAN"],
    #         # final_val_loss["G_criterion"],
    #     )
    # )

    # final_test_loss, final_test_e = summary(models, test_loader, loss_fns)
    # log.debug(
    #     "{} | e: {:.3f}, D_real: {:.3f}, D_fake: {:.3f}, G_GAN: {:.3f}".format(
    #         "final test",
    #         final_test_e,
    #         final_test_loss["D_real"],
    #         final_test_loss["D_fake"],
    #         final_test_loss["G_GAN"],
    #         # final_test_loss["G_criterion"],
    #     )
    # )

    # mlflow.log_metric("fold{}_final_val_G_GAN".format(), total_val_loss["G_GAN"])
    # mlflow.log_metric("fold{}_final_test_G_GAN".format(), total_test_loss["G_GAN"])

    return 0.0, 0.0


# helper functions
def init_early_stopping(args):
    log.debug("Setup EarlyStopping...")
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=20, stop_epoch=50, verbose=True)
    else:
        early_stopping = None

    return early_stopping


def init_loaders(args, train_split, val_split, test_split):
    log.debug("Init Loaders...")
    train_loader = get_split_loader(
        args, train_split, training=True, batch_size=args.batch_size
    )
    val_loader = get_split_loader(
        args, val_split, training=False, batch_size=args.batch_size
    )
    test_loader = get_split_loader(
        args, test_split, training=False, batch_size=args.batch_size
    )

    return train_loader, val_loader, test_loader


def init_optims(args, models):
    log.debug("Init optimizers...")
    optimizers = {
        "optim_G": torch.optim.Adam(
            models["net_G"].parameters(), lr=args.lr, betas=(0.9, 0.999)
        ),
        "optim_D": torch.optim.Adam(
            models["net_D"].parameters(), lr=args.lr, betas=(0.9, 0.999)
        ),
    }

    return optimizers


def init_models(args):
    log.debug("Init models...")
    if args.model_type == "mlp":
        models = {
            "net_G": GeneratorMLP(n_tokens=1024, dropout=args.drop_out),
            "net_D": DiscriminatorMLP(n_tokens=1024, dropout=args.drop_out),
        }
    elif args.model_type == "transformer":
        models = {
            "net_G": GeneratorTransformer(
                n_tokens=1024,
                dropout=args.drop_out,
                n_heads=args.n_heads,
                emb_dim=args.emb_dim,
            ),
            "net_D": DiscriminatorTransformer(
                n_tokens=1024,
                dropout=args.drop_out,
                n_heads=args.n_heads,
                emb_dim=args.emb_dim,
            ),
        }
    elif args.model_type == "independent":
        models = {
            "net_G": GeneratorIndependent(),
            "net_D": DiscriminatorIndependent(),
        }
    elif args.model_type == "independent_fast":
        models = {
            "net_G": GeneratorIndependentFast(),
            "net_D": DiscriminatorIndependentFast(),
        }
    else:
        raise ValueError("Invalid model type.")

    print_network(args.results_dir, models["net_G"], "G")
    print_network(args.results_dir, models["net_D"], "D")

    return models


def init_loss_functions(args):
    log.debug("Init loss functions...")
    loss_fns = {
        "GAN": GANLoss(),
    }

    if args.reg_type == "L1":
        loss_fns["loss_criterion"] = torch.nn.L1Loss()
    elif args.reg_type == "L2":
        loss_fns["loss_criterion"] = torch.nn.MSELoss()
    elif args.reg_type == "cosine":
        loss_fns["loss_criterion"] = torch.nn.CosineEmbeddingLoss()
    else:
        loss_fns["loss_criterion"] = None
        # raise ValueError("Invalid norm type")

    return loss_fns


def get_splits(datasets, args):
    log.debug("Init train/val/test splits...")
    train_split, val_split, test_split = datasets
    # save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format()))
    log.debug("Done!")
    log.debug("Training on {} samples".format(len(train_split)))
    log.debug("Validating on {} samples".format(len(val_split)))
    log.debug("Testing on {} samples".format(len(test_split)))
    return train_split, val_split, test_split


# GAN stuff
class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        """Initialize the GANLoss class.
        Parameters:
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
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

        loss = self.loss(prediction, target_tensor)
        return loss


def calculate_losses_G(net_D, loss_fns, real_A, real_B, fake_B, lambda_criterion=100):
    """Calculate GAN and L1 loss for the generator"""
    # First, G(A) should fake the discriminator
    # we use conditional GANs; we need to feed both input and output to the discriminator
    pred_fake = net_D.forward(real_A, fake_B)
    G_GAN = loss_fns["GAN"](pred_fake, True)

    # Second, G(A) = B
    G_criterion = 0.0
    # if isinstance(loss_fns["loss_criterion"], torch.nn.CosineEmbeddingLoss):
    #     G_criterion = loss_fns["loss_criterion"](fake_B, real_B, Variable(torch.Tensor(fake_B.size(0)).cuda().fill_(1.0))) * lambda_criterion
    # else:
    #     G_criterion = loss_fns["loss_criterion"](fake_B, real_B) * lambda_criterion

    return G_GAN, G_criterion


def calculate_losses_D(net_D, loss_fns, real_A, real_B, fake_B):
    """Calculate GAN loss for the discriminator"""
    # Fake; stop backprop to the generator by detaching fake_B
    # we use conditional GANs; we need to feed both input and output to the discriminator
    pred_fake = net_D.forward(real_A, fake_B)
    D_fake = loss_fns["GAN"](pred_fake, False)

    # Real
    pred_real = net_D(real_A, real_B)
    D_real = loss_fns["GAN"](pred_real, True)

    return D_real, D_fake


def set_requires_grad(net, requires_grad=False):
    """Sets requires_grad for all the networks to avoid unnecessary computations"""
    if net is not None:
        for param in net.parameters():
            param.requires_grad = requires_grad


# train, val, test
def train_loop(models, loader, optimizers, loss_fns):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models["net_G"].train().to(device)
    models["net_D"].train().to(device)

    # sanity check metric
    avg_e = 0.0

    total_loss = {
        "D_real": 0.0,
        "D_fake": 0.0,
        "G_GAN": 0.0,
        "G_criterion": 0.0,
    }

    for batch_idx, data in enumerate(loader):
        original, augmentation, noise = data  # split data

        # move tensors to cuda
        original, augmentation, noise = (
            original.to(device),
            augmentation.to(device),
            noise.to(device),
        )

        fake_augmentation = models["net_G"].forward(
            original, noise
        )  # compute fake images: G(A)

        # update D
        set_requires_grad(models["net_D"], True)  # enable backprop for D
        optimizers["optim_D"].zero_grad()  # set D's gradients to zero
        D_real, D_fake = calculate_losses_D(
            models["net_D"], loss_fns, original, augmentation, fake_augmentation
        )  # calculate gradients for D
        # combine loss and calculate gradients
        D = (D_fake + D_real) * 0.5
        D.backward(retain_graph=True)
        optimizers["optim_D"].step()  # update D's weights

        # update G
        set_requires_grad(
            models["net_D"], False
        )  # D requires no gradients when optimizing G
        optimizers["optim_G"].zero_grad()  # set G's gradients to zero
        G_GAN, G_criterion = calculate_losses_G(
            models["net_D"], loss_fns, original, augmentation, fake_augmentation
        )  # calculate gradients for G
        # combine loss and calculate gradients

        # get rid of criterion loss
        G_criterion = 0.0

        G = G_GAN + G_criterion
        G.backward(retain_graph=True)

        optimizers["optim_G"].step()  # update G's weights

        total_loss["D_real"] = D_real.item()
        total_loss["D_fake"] = D_fake.item()
        total_loss["G_GAN"] = G_GAN.item()
        # total_loss["G_criterion"] += G_criterion.item()

        # l1 distance
        with torch.no_grad():
            e = torch.cdist(augmentation, fake_augmentation, p=1)
            # log.debug(f"aug: {augmentation.size()}")
            # log.debug(f"fake: {fake_augmentation.size()}")
            # log.debug(f"e: {e.size()}")

            e = e.mean()
            avg_e += e

            # log.debug(f"train | batch: {batch_idx} | e: {e:.3f}")

            for key, loss in total_loss.items():
                mlflow.log_metric(key=f"train_{key}", value=loss, step=batch_idx)

            mlflow.log_metric(key=f"train_e", value=e, step=batch_idx)

            if (batch_idx % len(loader) // 100) == 0:
                log.debug(
                    "train | batch: {} | e: {:.3f}, D_real: {:.3f}, D_fake: {:.3f}, G_GAN: {:.3f}".format(
                        batch_idx,
                        e,
                        total_loss["D_real"],
                        total_loss["D_fake"],
                        total_loss["G_GAN"],
                        # total_loss["G_criterion"],
                    )
                )

    # total_loss["D_real"] /= len(loader)
    # total_loss["D_fake"] /= len(loader)
    # total_loss["G_GAN"] /= len(loader)
    # total_loss["G_criterion"] /= len(loader)

    # avg_e /= len(loader)

    # log.debug(
    #     "{} | D_real: {:.3f}, D_fake: {:.3f}, G_GAN: {:.3f}, G_criterion: {:.3f}".format(
    #         "train",
    #         total_loss["D_real"],
    #         total_loss["D_fake"],
    #         total_loss["G_GAN"],
    #         total_loss["G_criterion"],
    #     )
    # )

    return total_loss


def validate(models, loader, early_stopping, loss_fns=None, results_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models["net_G"].eval().to(device)
    models["net_D"].eval().to(device)

    # # sanity check metric
    # avg_e = 0.0

    total_loss = {
        "D_real": 0.0,
        "D_fake": 0.0,
        "G_GAN": 0.0,
        "G_criterion": 0.0,
    }

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            # log.info(f"Batch: {batch_idx}")
            original, augmentation, noise = data  # split data

            # move tensors to cuda
            original, augmentation, noise = (
                original.to(device),
                augmentation.to(device),
                noise.to(device),
            )

            fake_augmentation = models["net_G"].forward(
                original, noise
            )  # compute fake images: G(A)

            # calculate losses
            D_real, D_fake = calculate_losses_D(
                models["net_D"], loss_fns, original, augmentation, fake_augmentation
            )
            G_GAN, G_criterion = calculate_losses_G(
                models["net_D"], loss_fns, original, augmentation, fake_augmentation
            )

            total_loss["D_real"] = D_real.item()
            total_loss["D_fake"] = D_fake.item()
            total_loss["G_GAN"] = G_GAN.item()
            # loss["G_criterion"] = G_criterion.item()

            # l1 distance
            e = torch.cdist(augmentation, fake_augmentation, p=1)
            # log.debug(f"aug: {augmentation.size()}")
            # log.debug(f"fake: {fake_augmentation.size()}")
            # log.debug(f"e: {e.size()}")

            e = e.mean()
            # avg_e += e

            for key, loss in total_loss.items():
                mlflow.log_metric(key=f"val_{key}", value=loss, step=batch_idx)

            # log.debug(f"val | batch: {batch_idx} | e: {e}")
            mlflow.log_metric(key=f"val_e", value=e, step=batch_idx)

            if (batch_idx % len(loader) // 100) == 0:
                log.debug(
                    "val | batch: {} | e: {:.3f}, D_real: {:.3f}, D_fake: {:.3f}, G_GAN: {:.3f}".format(
                        batch_idx,
                        e,
                        total_loss["D_real"],
                        total_loss["D_fake"],
                        total_loss["G_GAN"],
                        # total_loss["G_criterion"],
                    )
                )

    # total_loss["D_real"] /= len(loader)
    # total_loss["D_fake"] /= len(loader)
    # total_loss["G_GAN"] /= len(loader)
    # total_loss["G_criterion"] /= len(loader)
    # avg_e /= len(loader)

    # for key, loss in total_loss.items():
    #     mlflow.log_metric(key=f"val_{key}", value=loss, step=epoch)
    # mlflow.log_metric(key=f"val_e", value=avg_e, step=epoch)

    # if early_stopping:
    #     assert results_dir
    #     early_stopping(
    #         epoch,
    #         total_loss,
    #         models,
    #         ckpt_name=os.path.join(results_dir, "checkpoint.pt"),
    #     )

    #     if early_stopping.early_stop:
    #         log.debug("Early stopping")
    #         return True

    # return False


def summary(models, loader, loss_fns):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models["net_G"].eval().to(device)
    models["net_D"].eval().to(device)

    total_loss = {
        "D_real": 0.0,
        "D_fake": 0.0,
        "G_GAN": 0.0,
        "G_criterion": 0.0,
    }

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            original, augmentation, noise = data  # split data

            # move tensors to cuda
            original, augmentation, noise = (
                original.to(device),
                augmentation.to(device),
                noise.to(device),
            )

            fake_augmentation = models["net_G"].forward(
                original, noise
            )  # compute fake images: G(A)

            # calculate losses
            D_real, D_fake = calculate_losses_D(
                models["net_D"], loss_fns, original, augmentation, fake_augmentation
            )
            G_GAN, G_criterion = calculate_losses_G(
                models["net_D"], loss_fns, original, augmentation, fake_augmentation
            )

            total_loss["D_real"] += D_real.item()
            total_loss["D_fake"] += D_fake.item()
            total_loss["G_GAN"] += G_GAN.item()
            # total_loss["G_criterion"] += G_criterion.item()

    return total_loss


def train_val_test(train_split, val_split, test_split, args):
    """
    Performs train val test for the fold over number of epochs
    """

    # ----> gets splits and summarize
    # train_split, val_split, test_split = get_splits(datasets, args)

    # ----> init loss function
    loss_fns = init_loss_functions(args)

    # ----> init model
    models = init_models(args)

    # ---> init optimizer
    optimizers = init_optims(args, models)

    # ---> init loaders
    train_loader, val_loader, test_loader = init_loaders(
        args, train_split, val_split, test_split
    )

    # ---> init early stopping
    early_stopping = init_early_stopping(args)

    # ---> do train val test
    total_val_loss, total_test_loss = step(
        args,
        loss_fns,
        models,
        optimizers,
        train_loader,
        val_loader,
        test_loader,
        early_stopping,
    )

    return total_val_loss, total_test_loss
