# ----> internal imports
from inspect import trace

# from datasets.dataset import save_splits
from utils.utils import EarlyStopping, get_optim, get_split_loader, print_network

# ----> pytorch imports
import torch
import torch.nn as nn
from torchmetrics.functional import auc

# ----> general imports
import numpy as np
import mlflow
import os
from models.model_attention_mil import SingleTaskAttentionMILClassifier
from sklearn import metrics

import logging

log = logging.getLogger(__name__)


def step(
    cur,
    args,
    loss_fn,
    model,
    optimizer,
    train_loader,
    val_loader,
    test_loader,
    early_stopping,
):

    for epoch in range(args.max_epochs):
        train_loop(epoch, cur, model, train_loader, optimizer, loss_fn)
        stop = validate(
            cur, epoch, model, val_loader, early_stopping, loss_fn, args.results_dir
        )
        if stop:
            break

    if args.early_stopping:
        model.load_state_dict(
            torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
        )
    else:
        torch.save(
            model.state_dict(),
            os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)),
        )

    val_metrics = summary(model, val_loader, loss_fn)
    log.debug(
        "f{} e{}, {} | loss_D_real: {:.3f}, loss_D_fake: {:.3f}, loss_G_GAN: {:.3f}".format(
            cur,
            epoch,
            "final val",
            val_metrics["loss"],
            val_metrics["acc"],
            val_metrics["kappa"],
            val_metrics["auc"],
        )
    )

    test_metrics = summary(model, test_loader, loss_fn)
    log.debug(
        "f{} e{}, {} | loss_D_real: {:.3f}, loss_D_fake: {:.3f}, loss_G_GAN: {:.3f}".format(
            cur,
            epoch,
            "final test",
            test_metrics["loss"],
            test_metrics["acc"],
            test_metrics["kappa"],
            test_metrics["auc"],
        )
    )

    results = {}

    for key, value in val_metrics.items():
        results[f"val_{key}"] = value
        mlflow.log_metric(key=f"final_val_{key}", value=value, step=cur)
    for key, value in test_metrics.items():
        results[f"val_{key}"] = value
        mlflow.log_metric(key=f"final_test_{key}", value=value, step=cur)

    return results


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


def init_optim(args, model):
    log.debug("Init optimizer ...")
    optimizer = get_optim(model, args)
    return optimizer


def init_model(args):
    log.debug("Init Model...")
    model = SingleTaskAttentionMILClassifier(args)
    print_network(args.results_dir, model)
    return model


def init_loss_function(args):
    log.debug("\nInit loss function...", end=" ")
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn


def train_loop(epoch, cur, model, loader, optimizer, loss_fn):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train().to(device)

    y, y_probs, y_preds = [], [], []

    total_metrics = {
        "loss": 0.0,
        "acc": 0.0,
        "kappa": 0.0,
        "auc": 0.0,
    }
    total_num = 0
    total_correct = 0

    for batch_idx, data in enumerate(loader):
        log.debug(f"f{cur} e{epoch}, batch {batch_idx}")
        patch_embs, label = data  # split data

        # set data on device
        patch_embs = patch_embs.squeeze().to(device)
        label = label.to(device)

        # forward pass
        logits, Y_prob, Y_hat, A_raw, results_dict = model(patch_embs)

        # compute loss
        loss = loss_fn(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_metrics["loss"] += loss.item()

        total_num += 1
        if torch.argmax(Y_prob).item() == label.item():
            # correct if index of y_pred is equal to label
            total_correct += 1

        # labels
        y.append(label.item())
        y_probs.append(torch.max(Y_prob).item())
        y_preds.append(torch.argmax(Y_prob).item())

        if (batch_idx % 20) == 0:
            log.debug(
                "f{} e{}, batch {} | loss: {:.3f}, acc: {:.3f}, kappa: {:.3f}, auc: {:.3f}".format(
                    cur,
                    epoch,
                    batch_idx,
                    total_metrics["loss"],
                    total_metrics["acc"],
                    total_metrics["kappa"],
                    total_metrics["auc"],
                )
            )

    total_metrics["loss"] /= len(loader)
    total_metrics["acc"] = total_correct / total_num
    total_metrics["kappa"] = metrics.cohen_kappa_score(y, y_preds, weights="quadratic")
    total_metrics["auc"] = auc(
        torch.tensor(y).to(device), torch.tensor(y_probs).to(device), reorder=True
    )

    log.debug(
        "f{} e{}, {} | loss: {:.3f}, acc: {:.3f}, kappa: {:.3f}, auc: {:.3f}".format(
            cur,
            epoch,
            "train",
            total_metrics["loss"],
            total_metrics["acc"],
            total_metrics["kappa"],
            total_metrics["auc"],
        )
    )

    mlflow.log_metric(key=f"fold{cur}_train_loss", value=total_metrics, step=epoch)

    return 0.0, total_metrics


def validate(cur, epoch, model, loader, early_stopping, loss_fn=None, results_dir=None):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    y, y_probs, y_preds = [], [], []

    total_metrics = {
        "loss": 0.0,
        "acc": 0.0,
        "kappa": 0.0,
        "auc": 0.0,
    }
    total_num = 0
    total_correct = 0

    with torch.no_grad():
        for patch_embs, label in loader:

            patch_embs = patch_embs.squeeze().to(device)
            label = label.to(device)
            logits, Y_prob, Y_hat, A_raw, results_dict = model(patch_embs)
            loss = loss_fn(logits, label)
            total_metrics["loss"] += loss.item()

            total_num += 1
            if torch.argmax(Y_prob).item() == label.item():
                # correct if index of y_pred is equal to label
                total_correct += 1

            # labels
            y.append(label.item())
            y_probs.append(torch.max(Y_prob).item())
            y_preds.append(torch.argmax(Y_prob).item())

            # print(f"logits: {logits}, label: {label}")

    total_metrics["loss"] /= len(loader)
    total_metrics["acc"] = total_correct / total_num
    total_metrics["kappa"] = metrics.cohen_kappa_score(y, y_preds, weights="quadratic")
    total_metrics["auc"] = auc(
        torch.tensor(y).to(device), torch.tensor(y_probs).to(device), reorder=True
    )

    log.debug(
        "f{} e{}, {} | loss: {:.3f}, acc: {:.3f}, kappa: {:.3f}, auc: {:.3f}".format(
            cur,
            epoch,
            "val",
            total_metrics["loss"],
            total_metrics["acc"],
            total_metrics["kappa"],
            total_metrics["auc"],
        )
    )
    for key, value in total_metrics.items():
        mlflow.log_metric(key=f"fold{cur}_val_{key}", value=value, step=epoch)

    if early_stopping:
        assert results_dir
        early_stopping(
            epoch,
            total_metrics["loss"],
            model,
            ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)),
        )

        if early_stopping.early_stop:
            log.debug("Early stopping")
            return True

    return False


def summary(model, loader, loss_fn):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    y, y_probs, y_probs, y_preds = [], [], [], []

    total_metrics = {
        "loss": 0.0,
        "acc": 0.0,
        "kappa": 0.0,
        "auc": 0.0,
    }
    total_num = 0
    total_correct = 0

    with torch.no_grad():
        for patch_embs, label in loader:
            patch_embs = patch_embs.squeeze().to(device)
            label = label.to(device)
            logits, Y_prob, Y_hat, A_raw, results_dict = model(patch_embs)
            loss = loss_fn(logits, label)
            total_metrics["loss"] += loss.item()

            # labels
            y.append(label.item())
            y_probs.append(torch.max(Y_prob).item())
            y_preds.append(torch.argmax(Y_prob).item())

            total_num += 1
            if torch.argmax(Y_prob).item() == label.item():
                # correct if index of y_pred is equal to label
                total_correct += 1

            # print(f"logits: {logits}, label: {label}")

    total_metrics["loss"] /= len(loader)
    total_metrics["acc"] = total_correct / total_num
    total_metrics["kappa"] = metrics.cohen_kappa_score(y, y_preds, weights="quadratic")
    total_metrics["auc"] = auc(
        torch.tensor(y).to(device), torch.tensor(y_probs).to(device), reorder=True
    )

    return total_metrics


def train_val_test(train_split, val_split, test_split, args, cur):
    """
    Performs train val test for the fold over number of epochs
    """

    # ----> init loss function
    loss_fn = init_loss_function(args)

    # ----> init model
    model = init_model(args)

    # ---> init optimizer
    optimizer = init_optim(args, model)

    # ---> init loaders
    train_loader, val_loader, test_loader = init_loaders(
        args, train_split, val_split, test_split
    )

    # ---> init early stopping
    early_stopping = init_early_stopping(args)

    # ---> do train val test
    results = step(
        cur,
        args,
        loss_fn,
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        early_stopping,
    )

    return results
