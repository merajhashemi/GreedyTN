import numpy as np
import tensorly
import torch

import core_code as cc

np.random.seed(1)
torch.manual_seed(1)


def generate_tucker(input_dims, ranks):
    assert len(input_dims) == len(ranks)
    n = len(ranks)
    rank_mat = np.ones([n + 1, n + 1])
    for i in range(n):
        rank_mat[i, i] = input_dims[i]
        rank_mat[i, n] = ranks[i]
        rank_mat[n, i] = ranks[i]

    tucker_cores = cc.random_tn(rank=rank_mat)

    return tucker_cores


def generate_cp(input_dims, rank):
    factors = []
    for d in input_dims:
        factors.append(np.random.normal(0, 1, [d, rank]))
    T = tensorly.cp_to_tensor((np.ones(rank), factors))
    nparams = np.sum([np.prod(A.shape) for A in factors])

    return T, nparams


def generate_tensor_tri(input_dims, tri_ranks):
    """
    Generate random tensor network with tiangle structure

    Args:
        input_dims: List of input dimensions for each core in the network
        tri_ranks:  List of ranks

    Returns:
        tri_cores:  List of randomly initialized tensor cores for triangle TN
    """
    assert len(input_dims) == len(tri_ranks)
    n_cores = len(input_dims)
    ranks = []
    for i in range(n_cores - 1):
        rank_i = np.ones((n_cores - 1 - i), dtype=np.int32)
        rank_i[0] = tri_ranks[i]
        ranks.append(rank_i.tolist())
    ranks[-1][-1] = 1
    ranks[1][-1] = tri_ranks[-2]
    ranks[2][-1] = tri_ranks[-1]
    tri_cores = cc.random_tn(input_dims=input_dims, rank=ranks)

    return tri_cores


def generate_tensor_train(input_dims, tt_ranks):
    """
    Generate random tensor train

    Args:
        input_dims: List of input dimensions for each core in the network
        tt_ranks:   List of TT ranks

    Returns:
        tt_cores:   List of randomly initialized tensor train cores
    """
    assert len(input_dims) == len(tt_ranks) + 1
    n_cores = len(input_dims)
    ranks = []
    for i in range(n_cores - 1):
        rank_i = np.ones((n_cores - 1 - i), dtype=np.int32)
        rank_i[0] = tt_ranks[i]
        ranks.append(rank_i.tolist())
    tt_cores = cc.random_tn(input_dims=input_dims, rank=ranks)

    return tt_cores


def generate_tensor_ring(input_dims, tr_ranks):
    """
    Generate random tensor ring

    Args:
        input_dims: List of input dimensions for each core in the network
        tr_ranks:   List of TR ranks

    Returns:
        tr_cores:   List of randomly initialized tensor ring cores
    """
    assert len(input_dims) == len(tr_ranks)
    n_cores = len(input_dims)
    ranks = []
    for i in range(n_cores - 1):
        rank_i = np.ones((n_cores - 1 - i), dtype=np.int32)
        rank_i[0] = tr_ranks[i]
        ranks.append(rank_i.tolist())
    ranks[0][-1] = tr_ranks[-1]
    tr_cores = cc.random_tn(input_dims=input_dims, rank=ranks)

    return tr_cores
