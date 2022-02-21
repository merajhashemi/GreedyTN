import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import core_code as cc
from ncon import ncon


class TensorNetworkLayer(nn.Module):
    def __init__(self, input_dims, output_dims, matrix, rank):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.matrix = matrix
        if matrix:
            ndim = len(output_dims) + len(input_dims)
            expanded_shape = [None] * ndim
            expanded_shape[::2] = output_dims
            expanded_shape[1::2] = input_dims
            self.expanded_shape = expanded_shape
            self.order = tuple(range(0, ndim, 2)) + tuple(range(1, ndim, 2))
            self.dangling_dims = [i * o for i, o in zip(input_dims, output_dims)]
        else:
            self.dangling_dims = output_dims + input_dims
        self.tensor_list = nn.ParameterList([nn.Parameter(t) for t in cc.random_tn(self.dangling_dims, rank=rank)])
        self.bias = nn.Parameter(torch.zeros(int(np.prod(output_dims))))

    def forward(self, x):
        edge_list = cc.get_edge_list(self.tensor_list)
        weight_tensor = ncon(self.tensor_list, edge_list)
        if self.matrix:
            weight_tensor = weight_tensor.view(self.expanded_shape).permute(*self.order)
        weight_matrix = weight_tensor.contiguous().view(np.prod(self.output_dims), np.prod(self.input_dims))
        return F.linear(x, weight_matrix, self.bias)

    @torch.no_grad()
    def increase_rank(self, vertex1, vertex2, rank_inc=1, pad_noise=1e-6):
        self.tensor_list = cc.increase_rank(self.tensor_list, vertex1, vertex2, rank_inc, pad_noise)

    # @torch.no_grad()
    # def randomize_cores(self):
    #     self.tensor_list = nn.ParameterList(
    #         [nn.Parameter(t) for t in cc.random_tn(self.dangling_dims, rank=self.adj_matrix.tolist())])

    @torch.no_grad()
    def add_internal_node(self):
        new_node_shape = (1,) * (self.num_cores + 1)
        self.tensor_list = nn.ParameterList([nn.Parameter(t.unsqueeze(-1)) for t in self.tensor_list])
        # TODO: think of a proper initializiation for the new internal node
        self.tensor_list.append(nn.Parameter(torch.randn(new_node_shape)))

    @property
    def adj_matrix(self):
        return cc.get_adj_matrix(self.tensor_list)

    @property
    def num_cores(self):
        return len(self.tensor_list)


class TNMnist(nn.Module):
    def __init__(self, matrix, rank=1):
        super().__init__()
        self.tensor_net = TensorNetworkLayer(input_dims=[4, 4, 4, 4, 4],
                                             output_dims=[4, 4, 4, 4, 4],
                                             matrix=matrix,
                                             rank=rank)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(1024, 10)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.tensor_net(x)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

    @property
    def num_params(self):
        return sum(param.numel() for param in self.parameters())
