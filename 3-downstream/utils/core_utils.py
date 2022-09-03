#----> internal imports
from inspect import trace
# from datasets.dataset import save_splits
from utils.utils import EarlyStopping, get_optim, get_split_loader, print_network

#----> pytorch imports
import torch
import torch.nn as nn 
import torchmetrics

#----> general imports
import numpy as np
import mlflow 
import os
from models.model_attention_mil import SingleTaskAttentionMILClassifier
from sklearn import metrics
from torchmetrics.functional import auc


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

    val_loss, val_acc, val_kappa, val_auc = summary(model, val_loader, loss_fn)
    print('Final Val loss: {:.4f}'.format(val_loss))
    # print('Final Val acc: {:.4f}'.format(val_acc))
    # print('Final Val kappa: {:.4f}'.format(val_kappa))
    # print('Final Val auc: {:.4f}'.format(val_auc))

    test_loss, test_acc, test_kappa, test_auc = summary(model, test_loader, loss_fn)
    print('Final Test loss: {:.4f}'.format(test_loss))
    # print('Final Test acc: {:.4f}'.format(test_acc))
    # print('Final Test kappa: {:.4f}'.format(test_kappa))
    # print('Final Test auc: {:.4f}'.format(test_auc))

    mlflow.log_metric("final_test_loss_fold{}".format(cur), test_loss)
    mlflow.log_metric("final_test_acc_fold{}".format(cur), test_acc)
    mlflow.log_metric("final_test_kappa_fold{}".format(cur), test_kappa)
    mlflow.log_metric("final_test_auc_fold{}".format(cur), test_auc)

    results = {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_kappa": val_kappa,
        "val_auc": val_auc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_kappa": test_kappa,
        "test_auc": test_auc,
    }
    return results

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
    train_loader = get_split_loader(args, train_split, training=True, batch_size = args.batch_size)
    val_loader = get_split_loader(args, val_split,  training=False, batch_size = args.batch_size)
    test_loader = get_split_loader(args, test_split, training = False, batch_size = args.batch_size)
    print('Done!')
    return train_loader,val_loader,test_loader

def init_optim(args, model):
    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    return optimizer

def init_model(args):
    print('\nInit Model...', end=' ')
    model = SingleTaskAttentionMILClassifier(args)
    print_network(args.results_dir, model)
    return model

def init_loss_function(args):
    print('\nInit loss function...', end=' ')
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn

def train_loop(epoch, cur, model, loader, optimizer, loss_fn):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train().to(device)

    total_loss = 0.
    
    for patch_embs, label in loader:

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

        loss_value = loss.item()
        total_loss += loss_value 

    total_loss /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}'.format(epoch, total_loss))

    mlflow.log_metric("train_loss_fold{}".format(cur), total_loss)

    return 0., total_loss


def validate(cur, epoch, model, loader, early_stopping, loss_fn = None, results_dir = None):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    total_loss = 0.
    total_num = 0
    total_correct = 0

    y, y_probs, y_preds = [], [], []
    
    with torch.no_grad():
        for patch_embs, label in loader:

            patch_embs = patch_embs.squeeze().to(device)
            label = label.to(device)
            logits, Y_prob, Y_hat, A_raw, results_dict = model(patch_embs)
            loss = loss_fn(logits, label)
            total_loss += loss.item()

            total_num += 1
            if torch.argmax(Y_prob).item() == label.item():
                # correct if index of y_pred is equal to label
                total_correct += 1

            # labels
            y.append(label.item())
            y_probs.append(torch.max(Y_prob).item())
            y_preds.append(torch.argmax(Y_prob).item())

            # print(f"logits: {logits}, label: {label}")

    total_loss /= len(loader)
    acc = total_correct / total_num
    kappa = metrics.cohen_kappa_score(y, y_preds, weights='quadratic')
    auc_score = auc(torch.tensor(y).to(device), torch.tensor(y_probs).to(device), reorder=True)

    print('Epoch: {}, val_loss: {:.4f}'.format(epoch, total_loss))
    # print('Epoch: {}, val_acc: {:.4f}'.format(epoch, acc))
    # print('Epoch: {}, val_kappa: {:.4f}'.format(epoch, kappa))
    # print('Epoch: {}, val_auc: {:.4f}'.format(epoch, auc_score))

    mlflow.log_metric("val_loss_fold{}".format(cur), total_loss)
    mlflow.log_metric("val_acc_fold{}".format(cur), acc)
    mlflow.log_metric("val_kappa_fold{}".format(cur), kappa)
    mlflow.log_metric("val_auc_fold{}".format(cur), auc_score)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, total_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, loss_fn):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    total_loss = 0.
    total_num = 0
    total_correct = 0

    y, y_probs, y_preds = [], [], []
    
    with torch.no_grad():
        for patch_embs, label in loader:
            patch_embs = patch_embs.squeeze().to(device)
            label = label.to(device)
            logits, Y_prob, Y_hat, A_raw, results_dict = model(patch_embs)
            loss = loss_fn(logits, label)
            total_loss += loss.item()

            # labels
            y.append(label.item())
            y_probs.append(torch.max(Y_prob).item())
            y_preds.append(torch.argmax(Y_prob).item())

            total_num += 1
            if torch.argmax(Y_prob).item() == label.item():
                # correct if index of y_pred is equal to label
                total_correct += 1

            # print(f"logits: {logits}, label: {label}")

    total_loss /= len(loader)
    acc = total_correct / total_num
    kappa = metrics.cohen_kappa_score(y, y_preds, weights='quadratic')
    auc_score = auc(torch.tensor(y).to(device), torch.tensor(y_probs).to(device), reorder=True)

    return total_loss, acc, kappa, auc_score


def train_val_test(train_split, val_split, test_split, args, cur):
    """   
    Performs train val test for the fold over number of epochs
    """

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
    results = step(cur, args, loss_fn, model, optimizer, train_loader, val_loader, test_loader, early_stopping)

    return results