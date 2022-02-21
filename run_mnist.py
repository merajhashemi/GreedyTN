import argparse
import os

import torch
import torch.nn as nn
from torch.backends import cudnn

import discrete_optim
from data import MNISTDataModule
from models import TNMnist
from utils import seed_everything


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--seed', type=int, default=0,
                        help='seed')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--steps', type=int, default=40,
                        help='number of discrete optimization steps')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='discrete optimization save frequency')

    # discrete optimization algorithm
    parser.add_argument('--method', type=str, default='greedy',
                        choices=['greedy'],
                        help='discrete optimization method')
    parser.add_argument('--epochs_rank_one', type=int, default=10,
                        help='number of ALS iterations to run on the new slices to find the best edge for the "rank one" heuristic')

    # MPO vs MPS style tensor network
    parser.add_argument('--matrix', action='store_true',
                        help='MPO vs MPS style, default is MPS')

    # tensor network rank increment noise
    parser.add_argument('--pad_noise', type=float, default=1e-6,
                        help='std of gaussian noise for new slices')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0,
                        help='momentum')

    opt = parser.parse_args()

    # set results path and experiment name
    opt.save_dir = os.path.join(os.getcwd(), "results")
    mpo_or_mps = '_mpo' if opt.matrix else '_mps'
    opt.xp_name = f'{opt.method}' + mpo_or_mps + '_noise_{}_epochs_{}_steps_{}_grad2_lr_{}_wd_{}_bsz_{}.ckpt'. \
        format(opt.pad_noise, opt.epochs, opt.steps, opt.learning_rate, opt.weight_decay, opt.batch_size)

    return opt


def main():
    opt = parse_option()
    seed_everything(opt.seed)
    cudnn.benchmark = True

    # Define model
    model = TNMnist(matrix=opt.matrix, rank=1)

    # Define the dataloaders
    mnist_dm = MNISTDataModule(batch_size=opt.batch_size)
    train_loader = mnist_dm.train_dataloader()
    val_loader = mnist_dm.val_dataloader()
    test_loader = mnist_dm.test_dataloader()

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    discrete_func = getattr(discrete_optim, opt.method)
    hist = discrete_func(train_loader, val_loader, test_loader, model, criterion, opt)

    if not os.path.isdir(opt.save_dir):
        os.makedirs(opt.save_dir)
    torch.save(hist, os.path.join(opt.save_dir, opt.xp_name))


if __name__ == '__main__':
    main()
