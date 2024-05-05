"""
Utilities for the various architectures.

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
"""

# --- Libraries import
import torch


# --- Functions


def save_model(net: torch.nn.Module, optimizer: torch.optim.Optimizer,
               train_loss: float, val_loss: float,
               batch_size: int, epoch: int,
               path: str):
    path = str(path)
    state = dict(net=net.state_dict(),
                 opt=optimizer.state_dict(),
                 train_loss=train_loss,
                 val_loss=val_loss,
                 batch_size=batch_size,
                 epoch=epoch)
    torch.save(state, path)


def bin_acc(y_pred, y_test) -> float:
    y_pred_sig = torch.sigmoid(y_pred)
    y_pred_tags = y_pred_sig > 0.5
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc *= 100
    return acc


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.nn.functional.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc) * 100
    return acc


def batch_forward(net: torch.nn.Module, device, criterion,
                  data: torch.Tensor, labels: torch.Tensor) -> (torch.Tensor, float):
    if torch.cuda.is_available():
        data = data.to(device)
        labels = labels.to(device)
    out = net(data).squeeze()
    loss = criterion(out, labels)
    with torch.no_grad():
        acc = bin_acc(out, labels)
        #acc = multi_acc(out, labels)
    return loss, acc