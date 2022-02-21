import copy
import pickle

import numpy as np
import torch
from tqdm.auto import tqdm

import core_code as cc
from ncon import ncon


def squared_error_loss(A, B, weights=None):
    if weights is not None:
        return torch.norm(torch.mul(A.squeeze() - B.squeeze(), weights.squeeze()))
    else:
        return torch.norm(A.squeeze() - B.squeeze())


def RMSE(target, recov):
    return torch.norm(target.squeeze() - recov.squeeze()) / torch.norm(target) * 100


def weighted_lstsq(A, B, W):
    sol = torch.zeros([B.shape[1], A.shape[1]], dtype=A.dtype, device=A.device)
    for i in range(B.shape[1]):
        sol[i, :] = torch.lstsq(B[W[i, :], i], A[W[i, :]])[0][:A[W[i, :]].shape[1]].T
    return sol


def solve_least_squares(target, tensor_list, vertex, weights=None, target_unfold_list=None):
    num_cores = len(tensor_list)
    edge_list = (cc.get_edge_list(tensor_list))
    it = -1 * num_cores - 1
    new_edge_list = []
    new_tensor_list = []

    for i, L in enumerate(edge_list):
        if i != vertex:
            new_tensor_list.append(tensor_list[i])
            new_edge_list.append(L)
            new_edge_list[-1][vertex] = it
            it -= 1
    G_without_vertex = ncon(new_tensor_list, new_edge_list, order=cc.efficient_contraction_order(new_tensor_list, new_edge_list))

    G_without_vertex = G_without_vertex.reshape([np.prod(G_without_vertex.shape[:num_cores - 1]), np.prod(G_without_vertex.shape[num_cores - 1:])])
    axes = list(range(0, num_cores))
    axes.insert(0, axes.pop(vertex))
    if target_unfold_list is not None:
        target_unfold = target_unfold_list[vertex]
    else:
        target_unfold = cc.unfold(target, vertex)

    if weights is not None:
        weights_unfold = torch.movedim(weights, vertex, 0).reshape(target.shape[vertex], -1)

        sol = weighted_lstsq(G_without_vertex, target_unfold.T, weights_unfold)
    else:
        # sol = torch.lstsq(target_unfold.T, G_without_vertex)[0][:G_without_vertex.shape[1]].T
        sol = torch.linalg.lstsq(G_without_vertex, target_unfold.T, driver='gelsd').solution.T

    G_shape = list(tensor_list[vertex].shape)
    G_shape.insert(0, G_shape.pop(vertex))
    sol = sol.reshape(G_shape)
    sol = torch.movedim(sol, 0, vertex)
    return sol


def ALS(target, tensor_list, vertices, max_iter=500, cvg_threshold=1e-7, verbose=-1, weights=None, target_unfold_list=None):
    loss = squared_error_loss(target, cc.get_full_tensor(tensor_list), weights)
    if verbose > 0:
        print("init:", loss)
    prev_loss = np.inf
    for it in range(max_iter):
        for vertex in vertices:
            tensor_list[vertex] = solve_least_squares(target, tensor_list, vertex, weights, target_unfold_list)
        loss = squared_error_loss(target, cc.get_full_tensor(tensor_list), weights)
        if verbose > 0:
            print(it, ":", loss)
        if torch.abs(prev_loss - loss) < cvg_threshold:
            if verbose > 0:
                print("*** ALS converged at ", it)
            return tensor_list
        prev_loss = loss
    return tensor_list


def split_weights_train_val(weights, validation_ratio):
    weights = copy.deepcopy(weights)
    L = weights.nonzero().T
    N = L.shape[1]
    val_index = N - int(validation_ratio * N)
    L = L[:, torch.randperm(N)]
    idx = tuple(L[:, :val_index])
    val_idx = tuple(L[:, val_index:])
    weights, val_weights = torch.zeros_like(weights, dtype=torch.bool), torch.zeros_like(weights, dtype=torch.bool)
    weights[idx] = True
    val_weights[val_idx] = True
    return weights, val_weights


def increase_rank_and_rank_one_ALS(target, model, i, j, opt, weights=None):
    opt.weights = None
    residual = target - cc.get_full_tensor(model)
    vi_shape = list(model[i].shape)
    vj_shape = list(model[j].shape)
    vi_shape[j] = opt.rank_increment
    vj_shape[i] = opt.rank_increment
    Vi = torch.normal(0, opt.pad_noise, size=vi_shape)
    Vj = torch.normal(0, opt.pad_noise, size=vj_shape)
    current_model = copy.deepcopy(model)
    current_model[i] = Vi
    current_model[j] = Vj
    current_model = ALS(residual, current_model, (i, j), max_iter=opt.iter_rank_one, cvg_threshold=opt.cvg_threshold,
                        verbose=-1, weights=weights)
    current_model[i] = torch.cat((copy.deepcopy(model[i]), current_model[i]), dim=j)
    current_model[j] = torch.cat((copy.deepcopy(model[j]), current_model[j]), dim=i)
    return current_model


def find_best_edge(target, model, allowed_edges, opt, verbose=-1, weights=None, target_unfold_list=None, use_valid_data=-1):
    num_cores = len(target.shape)
    best_loss = np.inf

    if weights is not None and opt.use_valid_data > 0:
        weights, val_weights = split_weights_train_val(weights, opt.use_valid_data)

    for (i, j) in allowed_edges:
        if opt.heuristic == "full":
            current_model = cc.increase_rank(copy.deepcopy(model), i, j, rank_inc=opt.rank_increment, pad_noise=opt.pad_noise)
            current_model = ALS(target, current_model, range(num_cores), cvg_threshold=opt.cvg_threshold, max_iter=opt.epochs,
                                weights=weights, target_unfold_list=target_unfold_list)
            loss = squared_error_loss(target, cc.get_full_tensor(current_model), weights)
        elif opt.heuristic == "rank_one":
            current_model = increase_rank_and_rank_one_ALS(target, model, i, j, opt, weights)
            loss = squared_error_loss(target, cc.get_full_tensor(current_model), weights)

        if opt.use_valid_data > 0:
            train_loss = loss
            loss = squared_error_loss(target, cc.get_full_tensor(current_model), val_weights)
        if verbose > 0:
            print((i, j), ":", loss, " --- ", cc.get_num_params(current_model), "params")
            if use_valid_data > 0:
                print(f"\t train loss: {train_loss}")
        if loss < best_loss:
            best_loss = loss
            best_model = current_model[:]
            best_edge = (i, j)

    return best_model, best_edge, best_loss


def limit_arity(model, allowed_edges, max_arity):
    # removed edges that would make maximum degree above max_arity from allowed_edges list
    L = []
    for edge in allowed_edges:
        i, j = edge
        shape = model[i].shape
        shape = np.array(shape[:j] + shape[j + 1:])
        if (shape > 1).sum() < max_arity:
            L.append(edge)
    return L


def greedy_decomposition_ALS(target, opt, verbose=1, weights=None, max_arity=-1, internal_nodes=False):
    results = []
    num_cores = len(target.shape)

    it = 0
    if opt.restart_from_pickle is not None:
        with open(opt.restart_from_pickle, "rb") as f:
            results, _ = pickle.load(f)
        for r in results:
            print(r['num_params'])
        if opt.restart_from_step > 0:
            results = results[:opt.restart_from_step]
            it = opt.restart_from_step
        model = results[-1]['model']
        for r in results:
            print(r['num_params'])
        print(f"Restarting from previous stored model with {results[-1]['num_params']} params, loss {results[-1]['loss']} and rmse {RMSE(target, cc.get_full_tensor(model))}")

    if weights is not None:
        weights = weights.to(dtype=torch.bool)

    target_unfold_list = None

    for step in tqdm(range(it, it+opt.steps)):
        if not step:
            if opt.restart_from_pickle is None:
                model = cc.random_tn(target.shape, rank=[[512, 512,   1,   1],
                                                            [512, 512,   1,   1],
                                                            [  1,   1,   3,   1],
                                                            [  1,   1,   1,   3]])
            model = ALS(target, model, range(num_cores), max_iter=opt.epochs, cvg_threshold=opt.cvg_threshold, weights=weights, target_unfold_list=target_unfold_list)
            loss = squared_error_loss(target, cc.get_full_tensor(model), weights)
            if verbose > 0:
                print(f"{it}: initial loss (rank 1) {loss}", RMSE(target, cc.get_full_tensor(model)))
            it += 1
            results.append(
                {'iter': step,
                 'model': model,
                 'num_params': cc.get_num_params(model),
                 'loss': loss,
                 'weights': weights,
                 'target': target})
        else:
            allowed_edges = [(i, j) for i in range(num_cores) for j in range(i + 1, num_cores)]
            if max_arity > 0:
                allowed_edges = limit_arity(model, allowed_edges, max_arity)

            print("searching")
            best_model, best_edge, best_loss = find_best_edge(target, model, allowed_edges, opt, verbose=verbose,
                                                              weights=weights, target_unfold_list=target_unfold_list)

            model = best_model

            if opt.heuristic == 'rank_one':
                print("training")
                model = ALS(target, model, range(num_cores), max_iter=opt.epochs,
                            cvg_threshold=opt.cvg_threshold, verbose=-1, weights=weights, target_unfold_list=target_unfold_list)

            loss = squared_error_loss(target, cc.get_full_tensor(model), weights)
            if internal_nodes and step > 1:
                params_before = cc.get_num_params(model)
                print("internal node split search")
                print("loss before:", loss)
                print("params before:", params_before)
                print(cc.get_adj_matrix(model))
                ndim = len(model[0].shape)
                try:
                    new_model = cc.split_tensor_network(model, 1e-5)
                    print("params after:", cc.get_num_params(new_model))
                    print(cc.get_adj_matrix(model))
                    if params_before > cc.get_num_params(new_model):  # len(model[0].shape) != ndim:
                        model = new_model
                        while target.ndim < len(model):
                            target = torch.unsqueeze(target, -1)
                            if weights is not None:
                                weights = torch.unsqueeze(weights, -1)
                            num_cores += 1
                        print("added internal node, retraining")
                        model = ALS(target, model, range(num_cores), max_iter=opt.epochs,
                                    cvg_threshold=opt.cvg_threshold, verbose=-1, weights=weights, target_unfold_list=target_unfold_list)
                        loss = squared_error_loss(target, cc.get_full_tensor(model), weights)
                        print("loss after:", loss)
                except:
                    print("!!! Max number of internal nodes reached...")
                    internal_nodes = False
                    import traceback
                    traceback.print_exc()

            if verbose > 0:
                print(f"{it}: incremented {best_edge} -- {loss}", RMSE(target, cc.get_full_tensor(model)))
                print(cc.get_adj_matrix(model))
                print(cc.get_num_params(model))
            it += 1

            results.append(
                {'iter': step,
                 'model': model,
                 'num_params': cc.get_num_params(model),
                 'loss': loss,
                 'weights': weights})

            if opt.result_pickle:
                with open(opt.result_pickle, "wb") as f:
                    pickle.dump([results, opt], f)

        if loss <= opt.stopping_threshold:
            return results

    return results


def random_walk_decomposition(target, opt, verbose=1, weights=None, max_arity=-1, internal_nodes=True):
    results = []
    num_cores = len(target.shape)

    it = 0
    if opt.restart_from_pickle is not None:
        with open(opt.restart_from_pickle, "rb") as f:
            results, _ = pickle.load(f)
        for r in results:
            print(r['num_params'])
        if opt.restart_from_step > 0:
            results = results[:opt.restart_from_step]
            it = opt.restart_from_step
        model = results[-1]['model']
        for r in results:
            print(r['num_params'])
        print(f"Restarting from previous stored model with {results[-1]['num_params']} params, "
              f"loss {results[-1]['loss']} and rmse {RMSE(target, cc.get_full_tensor(model))}")

    if weights is not None:
        weights = weights.to(dtype=torch.bool)

    target_unfold_list = None

    for step in tqdm(range(it, it+opt.steps)):
        if not step:
            if opt.restart_from_pickle is None:
                model = cc.random_tn(target.shape, rank=1)

            model = ALS(target, model, range(num_cores), max_iter=opt.epochs, cvg_threshold=opt.cvg_threshold, weights=weights,
                        target_unfold_list=target_unfold_list)
            loss = squared_error_loss(target, cc.get_full_tensor(model), weights)
            if verbose > 0:
                print(f"{it}: initial loss (rank 1) {loss}", RMSE(target, cc.get_full_tensor(model)))
            it += 1
            results.append(
                {'iter': step,
                 'model': model,
                 'num_params': cc.get_num_params(model),
                 'loss': loss,
                 'weights': weights,
                 'target': target})
        else:
            allowed_edges = [(i, j) for i in range(num_cores) for j in range(i + 1, num_cores)]
            if max_arity > 0:
                allowed_edges = limit_arity(model, allowed_edges, max_arity)

            i, j = allowed_edges[torch.randperm(len(allowed_edges))[0]]
            model = cc.increase_rank(model, i, j, rank_inc=1, pad_noise=opt.pad_noise)
            model = ALS(target, model, range(num_cores), cvg_threshold=opt.cvg_threshold, max_iter=opt.epochs,
                        weights=weights, target_unfold_list=target_unfold_list)
            loss = squared_error_loss(target, cc.get_full_tensor(model), weights)
            if internal_nodes and step > 1:
                params_before = cc.get_num_params(model)
                print("internal node split search")
                print("loss before:", loss)
                print("params before:", params_before)
                print(cc.get_adj_matrix(model))
                try:
                    new_model = cc.split_tensor_network(model, 1e-5)
                    print("params after:", cc.get_num_params(new_model))
                    print(cc.get_adj_matrix(model))
                    if params_before > cc.get_num_params(new_model):  # len(model[0].shape) != ndim:
                        model = new_model
                        while target.ndim < len(model):
                            target = torch.unsqueeze(target, -1)
                            num_cores += 1
                        print("added internal node, retraining")
                        model = ALS(target, model, range(num_cores), max_iter=opt.epochs,
                                    cvg_threshold=opt.cvg_threshold, verbose=-1, weights=weights, target_unfold_list=target_unfold_list)
                        loss = squared_error_loss(target, cc.get_full_tensor(model), weights)
                        print("loss after:", loss)
                except:
                    print("!!! Max number of internal nodes reached...")
                    internal_nodes = False

            if verbose > 0:
                print(f"{it}: incremented {(i, j)} -- {loss}", RMSE(target, cc.get_full_tensor(model)))
                print(cc.get_adj_matrix(model))
                print(cc.get_num_params(model))
            it += 1
        num_params = cc.get_num_params(model)
        print(num_params)
        results.append(
            {'iter': step,
             'model': model,
             'num_params': num_params,
             'loss': loss,
             'weights': weights})

        if opt.result_pickle:
            with open(opt.result_pickle, "wb") as f:
                pickle.dump([results, opt], f)

        if loss <= opt.stopping_threshold or num_params > opt.max_params:
            return results

    return results
