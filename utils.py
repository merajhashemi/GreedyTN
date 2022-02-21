import copy
import random

import numpy as np
import torch


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        return time.time() - startTime_for_tictoc


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def accuracy(logits, labels):
    predicted_labels = torch.argmax(logits, dim=1)
    labels = labels.long()
    return torch.mean((predicted_labels == labels).float())


def train_one_epoch(train_loader, model, criterion, optimizer,
                    new_slice_only=False, vertex1=None, vertex2=None):
    losses = AverageMeter()
    accuracies = AverageMeter()

    # switch to train mode
    model.train()

    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        # compute output
        logits = model(images)
        loss = criterion(logits, labels)

        # measure accuracy and record loss
        acc = accuracy(logits, labels)
        accuracies.update(acc.item(), images.shape[0])
        losses.update(loss.item(), images.shape[0])

        # compute gradients
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # set other gradients to None
        if new_slice_only:
            for name, param in model.named_parameters():
                if name == 'tensor_net.tensor_list.' + str(vertex1):
                    param.grad.data.permute([vertex2] + list(range(0, vertex2)) + list(range(vertex2 + 1, model.tensor_net.num_cores)))[:-1] *= 0
                elif name == 'tensor_net.tensor_list.' + str(vertex2):
                    param.grad.data.permute([vertex1] + list(range(0, vertex1)) + list(range(vertex1 + 1, model.tensor_net.num_cores)))[:-1] *= 0
                else:
                    param.grad = None
        # do SGD step
        optimizer.step()

    return losses.avg, accuracies.avg


@torch.no_grad()
def validate(val_loader, model, criterion):
    # switch to eval mode
    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()

    for i, (images, labels) in enumerate(val_loader):
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        # compute output
        logits = model(images)
        loss = criterion(logits, labels)

        # measure accuracy and record loss
        acc = accuracy(logits, labels)
        accuracies.update(acc.item(), images.size(0))
        losses.update(loss.item(), images.size(0))

    return losses.avg, accuracies.avg


def train(train_loader, val_loader, test_loader, model, criterion, optimizer,
          opt, new_slice_only, vertex1=None, vertex2=None):
    best_epoch = 0
    best_val_acc = 0.
    best_model = model

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    test_losses, test_accuracies = [], []

    epoch = 0
    while epoch < opt.epochs:
        train_loss, train_acc = train_one_epoch(train_loader, model,
                                                criterion, optimizer,
                                                new_slice_only=new_slice_only,
                                                vertex1=vertex1,
                                                vertex2=vertex2)

        val_loss, val_acc = validate(val_loader, model, criterion)
        test_loss, test_acc = validate(test_loader, model, criterion)
        print(f'train_loss = {train_loss:.5f} \t train_acc = {train_acc:.5f} \t val_loss = {val_loss:.5f} \t val_acc = {val_acc:.5f} \t test_loss = {test_loss:.5f} \t test_acc = {test_acc:.5f}')
        if val_acc > best_val_acc:
            best_epoch = epoch
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        epoch += 1

    train_metrics = {
        'rank': best_model.tensor_net.adj_matrix,
        'state_dict': best_model.state_dict(),
        'num_params': best_model.num_params,
        'training_loss_best': train_losses[best_epoch],
        'training_acc_best': train_accuracies[best_epoch],
        'val_loss_best': val_losses[best_epoch],
        'val_acc_best': val_accuracies[best_epoch],
        'test_loss_best': test_losses[best_epoch],
        'test_acc_best': test_accuracies[best_epoch],
        'training_loss': train_losses[:best_epoch + 1],
        'training_acc': train_accuracies[:best_epoch + 1],
        'val_loss': val_losses[:best_epoch + 1],
        'val_acc': val_accuracies[:best_epoch + 1],
        'test_loss': test_losses[:best_epoch + 1],
        'test_acc': test_accuracies[:best_epoch + 1],
    }
    return best_model, train_metrics
