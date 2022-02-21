import argparse
from argparse import RawTextHelpFormatter

import numpy as np
import scipy.io
import torch

import discrete_optim_tensor_decomposition
from utils import seed_everything


def parse_option():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter, description='Example for greedy ALS:\n \
        "python run_decomposition.py --target_file=data/Einstein.mat --heuristic=rank_one --epochs 50 --method greedy_ALS --steps 20 --iter_rank_one=1 --cvg_threshold=1e-10"')

    parser.add_argument('--seed', type=int, default=0,
                        help='seed')
    parser.add_argument('--steps', type=int, default=200,
                        help='number of discrete optimization steps')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs for SGD / max iterations for ALS')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='discrete optimization save frequency')
    parser.add_argument('--target_file', type=str, default='data/Einstein.mat',
                        help='matlab file with the target tensor and experiment details (indices, shape for tensorization, ...) stored (if the extension of the file is .mat)\n\
                        or pickle file with the target tensor (if extension is .pickle)')
    parser.add_argument('--ratio_observed_entries', type=float, default=0.1,
                        help='ratio of observed entries')

    # discrete optimization algorithm
    parser.add_argument('--method', type=str, default='randomwalk',
                        choices=['randomwalk', 'greedy_SGD', 'greedy_ALS'],
                        help='discrete optimization method')

    parser.add_argument('--heuristic', type=str, default='rank_one',
                        choices=['rank_one', 'full'],
                        help='heuristic to find best edge in greedy:\n "full" for optimizing all parameters\
                        \n"rank_one" for optimizing only the new slices')
    parser.add_argument('--iter_rank_one', type=int, default=10,
                        help='number of ALS iterations to run on the new slices to find the best edge for the "rank one" heuristic')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate for SGD')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay for SGD')
    parser.add_argument('--momentum', type=float, default=0.95,
                        help='momentum for SGD')

    parser.add_argument('--pad_noise', type=float, default=1e-2,
                        help='magnitude of the noise to intialize new slices in greedy')

    parser.add_argument('--cvg_threshold', type=float, default=1e-7,
                        help='convergence cvg_threshold for ALS')
    parser.add_argument('--stopping_threshold', type=float, default=1e-7,
                        help='stopping threshold for greedy')

    # storing results and loading from previous optim
    parser.add_argument('--result_pickle', type=str, default=None,
                        help='pickle file name to store results')
    parser.add_argument('--restart_from_pickle', type=str, default=None,
                        help='previous results pickle from which to restart the optimization')
    parser.add_argument('--restart_from_step', type=int, default=-1,
                        help='step to start from to restart the optimization from the restart-from-pickle result file')

    parser.add_argument('--use_valid_data', type=float, default=0.1,
                        help='proportion of training indicies to use as a validation set to choose best edge in greedy')

    parser.add_argument('--internal_cores', type=bool, default=False,
                        help='whether to allow internal cores in greedy')

    parser.add_argument('--max_arity', type=int, default=4,
                        help='max node arity in TN')

    parser.add_argument('--rank_increment', type=int, default=1,
                        help='rank increment for greedy')

    opt = parser.parse_args()

    return opt


def main():
    opt = parse_option()
    seed_everything(opt.seed)
    order = 'F'

    data = scipy.io.loadmat(opt.target_file)
    image = data['Data']
    target_shape = data['Reshape_Dim'].astype(int)[0]
    image = np.asfortranarray(image).astype(float)
    target = image.reshape(target_shape, order=order)
    weights = data['Omega'].reshape(target.shape, order=order)

    discrete_optim_tensor_decomposition.greedy_decomposition_ALS(torch.from_numpy(target).float(), opt, weights=torch.from_numpy(weights), internal_nodes=opt.internal_cores, max_arity=opt.max_arity)


if __name__ == '__main__':
    main()
