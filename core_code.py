import copy
import itertools

import numpy as np
import torch
from torch import nn

from ncon import ncon


# noinspection PyTypeChecker
def random_tn(input_dims=None, rank=1):
    """
    Initialize a tensor network with random (normally distributed) cores

    Args:
        input_dims:  List of input dimensions for each core in the network
        rank:        Scalar or list of rank connecting different cores.
                     For scalar inputs, all ranks will be initialized at
                     the specified number, whereas more fine-grained ranks
                     are specified via a square matrix, or in the
                     following triangular format:
                     [[r_{1,2}, r_{1,3}, ..., r_{1,n}], [r_{2,3}, ..., r_{2,n}],
                      ..., [r_{n-2,n-1}, r_{n-2,n}], [r_{n-1,n}]]

    Returns:
        tensor_list: List of randomly initialized and properly formatted
                     tensors encoding our tensor network
    """
    # Convert rank object to list of shapes
    if hasattr(rank, 'shape'):
        # Matrix format
        shape_mat = rank.shape
        assert len(shape_mat) == 2
        assert shape_mat[0] == shape_mat[1]
        shape_list = [tuple(int(r) for r in row) for row in rank]
    elif hasattr(rank, '__len__'):
        if len(set(len(row) for row in rank)) == 1:
            # Matrix-type format
            shape_list = [tuple(int(r) for r in row) for row in rank]
        else:
            # Upper-triangular format
            shape_list = unpack_ranks(input_dims, rank)
    else:
        # Scalar format
        assert hasattr(input_dims, '__len__')
        r, n_c = rank, len(input_dims)
        shape_list = [(r,) * i + (d,) + (r,) * (n_c - 1 - i)
                      for i, d in enumerate(input_dims)]
    num_cores = len(shape_list)

    # Check that diagonals match input_dims
    if input_dims is not None:
        assert len(input_dims) == num_cores
        assert all(shape_list[i][i] == d for i, d in enumerate(input_dims))

    # Use shapes to instantiate random core tensors
    # The variance of the normal distributions is chosen to make the tensor norm 1 in expectation.
    tensor_list = []
    for i, shape in enumerate(shape_list):
        std = 1 / np.power(torch.prod(torch.tensor(shape)) * shape[i], 0.25).to(dtype=torch.float)
        tensor_list.append(std * torch.randn(shape))

    return tensor_list


def unpack_ranks(in_dims, ranks):
    """Converts triangular `ranks` structure to list of tensor shapes"""
    num_cores = len(in_dims)
    assert [len(rl) for rl in ranks] == list(range(num_cores - 1, 0, -1))

    shape_list = []
    for i in range(num_cores):
        shape = [ranks[j][i - j - 1] for j in range(i)]  # Lower triangular
        shape += [ranks[i][j - i - 1] for j in range(i + 1, num_cores)]  # Upper triangular
        shape.insert(i, in_dims[i])  # Diagonals
        shape_list.append(tuple(shape))

    return shape_list


def increase_rank(slim_list, vertex1, vertex2, rank_inc=1, pad_noise=1e-6):
    """
    Increase the rank of one bond in a tensor network

    Args:
        slim_list: List of tensors encoding a tensor network
        vertex1:   Node number for one end of the edge being increased
        vertex2:   Node number for the other end of the edge being
                   increased, which can't equal vertex1
        rank_inc:  Amount to increase the rank by (default 1)
        pad_noise: Increasing the rank involves embedding the original
                   TN in a larger parameter space, and adding a bit of
                   noise (set by pad_noise) helps later in training

    Returns:
        fat_list:  List of tensors encoding the same network, but with the
                   rank of the edge connecting nodes vertex1 and vertex2
                   increased by rank_inc
    """
    num_tensors = len(slim_list)
    assert 0 <= vertex1 < num_tensors
    assert 0 <= vertex2 < num_tensors
    assert rank_inc >= 0
    assert pad_noise >= 0

    # Function for increasing one index of one tensor
    def pad_tensor(tensor, ind):
        shape = list(tensor.shape)
        shape[ind] = rank_inc
        pad_mat = torch.randn(shape, device=tensor.device) * pad_noise
        padded_tensor = torch.cat([tensor, pad_mat], dim=ind)
        if isinstance(tensor, nn.parameter.Parameter):
            return nn.Parameter(padded_tensor)

        return padded_tensor

    # Pad both of the tensors along the index of the other
    fat_list = slim_list
    fat_list[vertex1] = pad_tensor(fat_list[vertex1], vertex2)
    fat_list[vertex2] = pad_tensor(fat_list[vertex2], vertex1)

    return fat_list


def get_adj_matrix(tensor_list):
    return torch.tensor([t.shape for t in tensor_list])


def get_edge_list(tensor_list):
    num_cores = len(tensor_list)
    num_edges = (num_cores * (num_cores - 1)) // 2
    edge_matrix = torch.zeros(num_cores, num_cores, dtype=torch.long)

    # Set bond dimensions
    edge_matrix[torch.triu_indices(num_cores, num_cores, offset=1).unbind()] = torch.arange(1, num_edges + 1)
    edge_matrix = edge_matrix + edge_matrix.T

    # Set the dangling edges on the diagonal
    torch.diagonal(edge_matrix).copy_(-torch.arange(1, num_cores + 1))

    return edge_matrix.tolist()


def get_contraction_order(tensor_list):
    num_cores = len(tensor_list)
    num_edges = (num_cores * (num_cores - 1)) // 2
    return torch.arange(1, num_edges).tolist()


def efficient_contraction_order(tensor_list, edge_list):
    """
    This function moves the singleton contraction indices to the end of the contraction list
    to avoid unnecessary outer products in the contraction
    """

    order = list(np.unique([x for l in edge_list for x in l if x >= 0]))
    for t, e in zip(tensor_list, edge_list):
        for d, i in zip(t.shape, e):
            if d == 1 and i >= 0:
                order.remove(i)
                order.append(i)

    return order


def get_full_tensor(tensor_list):
    edge_list = get_edge_list(tensor_list)
    return ncon(tensor_list, edge_list, order=efficient_contraction_order(tensor_list, edge_list))


def get_num_params(tensor_list):
    assert isinstance(tensor_list[0], torch.Tensor)
    return np.sum([t.numel() for t in tensor_list])


def unfold(T, modes):
    if not hasattr(modes, '__getitem__'):
        modes = tuple([modes])
    row_dims = [T.shape[i] for i in modes]
    M = torch.movedim(T, modes, tuple(range(len(modes))))
    return M.reshape([np.prod(row_dims), -1])


def fold(M, modes, shape):
    row_dims = [shape[i] for i in modes]
    col_dims = [shape[i] for i in range(len(shape)) if i not in modes]
    T = M.reshape(row_dims + col_dims)
    return torch.movedim(T, tuple(range(len(modes))), modes)


def split_tensor_network(tensor_list, eps, verbose=-1):
    tensor_list = copy.deepcopy(tensor_list)
    T_before = get_full_tensor(tensor_list)
    num_cores = len(tensor_list)

    for core_idx, t in enumerate(tensor_list):
        if t.squeeze().ndim <= 2:
            continue

        for nrows in range(1, int(num_cores / 2 + 1)):
            modes = [i for i in range(num_cores) if t.shape[i] > 1]
            for row_modes in itertools.combinations(modes, nrows):
                col_modes = [i for i in range(num_cores) if i not in row_modes]
                if core_idx in col_modes:
                    col_modes, row_modes = row_modes, col_modes
                M = unfold(t, row_modes)
                U, s, V = torch.linalg.svd(M)
                if torch.any(s < eps):
                    r = np.argmax(s < eps)

                    P_shape = [t.shape[i] if i in row_modes else 1 for i in range(num_cores)] + [r]
                    Q_shape = [t.shape[i] if i in col_modes else 1 for i in range(num_cores)] + [1]
                    Q_shape[core_idx] = r
                    if np.prod(Q_shape) + np.prod(P_shape) >= np.prod(t.shape):
                        continue
                    P = fold(U[:, :r], row_modes, P_shape)
                    Q = fold(torch.diag(s[:r]) @ V[:r, :], [core_idx], Q_shape)

                    M2 = fold(U[:, :r] @ torch.diag(s[:r]) @ V[:r, :], row_modes, t.shape)
                    tensor_list[core_idx] = M2

                    tensor_list = [torch.unsqueeze(T, -1) for T in tensor_list]
                    tensor_list[core_idx] = P
                    tensor_list.append(Q)
                    for i in range(num_cores):
                        if i in col_modes and i != core_idx:
                            tensor_list[i] = torch.transpose(tensor_list[i], core_idx, -1)
                    if verbose > 0:
                        print("approx error:", torch.norm(T_before.squeeze() - get_full_tensor(tensor_list).squeeze()))
                        print()
                    return split_tensor_network(tensor_list, eps)
    return tensor_list
