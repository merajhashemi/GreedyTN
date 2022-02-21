import argparse
import pickle
from argparse import RawTextHelpFormatter
from ast import literal_eval

from tqdm.auto import tqdm

import discrete_optim_tensor_decomposition
from random_tensors import *
from tensor_decomposition_models import incremental_tensor_decomposition
from utils import seed_everything
from utils import tic, toc


def parse_option():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument('--nruns', type=int, default=50)

    parser.add_argument('--seed', type=int, default=0,
                        help='seed')
    parser.add_argument('--steps', type=int, default=50,
                        help='number of discrete optimization steps')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs for SGD / max iterations for ALS')

    parser.add_argument('--target_type', type=str, default=None,
                        help='choices = {tucker, tt, tr, triangle}')

    parser.add_argument('--target_rank', type=str, default=None)
    parser.add_argument('--target_dims', type=str, default=None)

    parser.add_argument('--heuristic', type=str, default='rank_one',
                        choices=['rank_one', 'full'],
                        help='heuristic to find best edge in greedy:\n "full" for optimizing all parameters\
                        \n"rank_one" for optimizing only the new slices')
    parser.add_argument('--iter_rank_one', type=int, default=2,
                        help='number of ALS iterations to run on the new slices to find the best edge for the "rank one" heuristic')
    parser.add_argument('--rank_increment', type=int, default=1,
                        help='rank increment for greedy')
    parser.add_argument('--pad_noise', type=float, default=1e-2,
                        help='magnitude of the noise to intialize new slices in greedy')

    parser.add_argument('--cvg_threshold', type=float, default=1e-7,
                        help='convergence threshold for ALS')

    parser.add_argument('--stopping_threshold', type=float, default=1e-6,
                        help='stopping threshold for greedy')

    parser.add_argument('--result_pickle', type=str, default=None,
                        help='pickle file name to store results')

    parser.add_argument('--max_params', type=int, default=3000)

    parser.add_argument('--restart_from_pickle', type=str, default=None)
    parser.add_argument('--use_valid_data', type=float, default=-1)

    opt = parser.parse_args()

    return opt


def main():
    opt = parse_option()
    seed_everything(opt.seed)
    result_pickle = opt.result_pickle
    opt.result_pickle = None

    target_rank = literal_eval(opt.target_rank)
    target_dims = literal_eval(opt.target_dims)
    gen_dic = {'tucker': generate_tucker, 'tt': generate_tensor_train, 'tr': generate_tensor_ring, 'triangle': generate_tensor_tri}
    results = []
    for _ in tqdm(range(opt.nruns)):
        goal_tn = gen_dic[opt.target_type](target_dims, target_rank)

        target_full = cc.get_full_tensor(goal_tn).squeeze()
        target_full = target_full / torch.norm(target_full)
        result = {'target_tn': goal_tn, 'target_full': target_full}

        for decomp in "CP TT Tucker".split():
            print(decomp + "...")
            tic()
            result[decomp] = incremental_tensor_decomposition(target_full, decomp, verbose=False, max_num_params=opt.max_params,
                                                              rank_increment_factor=1.5 if decomp == 'CP' else 1)
            result[decomp + "-time"] = toc()
        print("greedy...")
        tic()
        result["greedy"] = discrete_optim_tensor_decomposition.greedy_decomposition_ALS(target_full, opt, verbose=-1, internal_nodes=False)
        result["greedy-time"] = toc()
        print("greedy w/ internal nodes...")
        tic()
        result["greedyint"] = discrete_optim_tensor_decomposition.greedy_decomposition_ALS(target_full, opt, verbose=-1, internal_nodes=True)
        result["greedyint-time"] = toc()
        tic()
        print("random walk...")
        result["rw"] = discrete_optim_tensor_decomposition.random_walk_decomposition(target_full, opt, verbose=-1, internal_nodes=False)
        result["rw-time"] = toc()

        results.append(result)

        with open(result_pickle, "wb") as f:
            pickle.dump([results, opt], f)


if __name__ == '__main__':
    main()
