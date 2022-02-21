import numpy as np
import tensorly as tl
import tensorly.decomposition
tl.set_backend('pytorch')


def l2_distance(T1, T2):
    return tl.norm(T1 - T2)


def CP(target, rank):
    lambd, factors = tl.decomposition.parafac(target, rank, init='svd')
    num_params = np.sum([A.numel() for A in factors])
    return tl.kruskal_to_tensor((lambd, factors)), num_params


def TT(target, rank):
    factors = tl.decomposition.tensor_train(target, rank)
    num_params = np.sum([A.numel() for A in factors])
    return tl.tt_to_tensor(factors), num_params


def Tucker(target, rank):
    ranks = [min(rank, d) for d in target.shape]
    (G, factors) = tl.decomposition.tucker(target, ranks)
    num_params = np.sum([G.numel()] + [A.numel() for A in factors])
    return tl.tucker_to_tensor((G, factors)), num_params


def incremental_tensor_decomposition(target, decomposition, loss_threshold=1e-4, max_num_params=1500, verbose=False, rank_increment_factor=1):
    if decomposition not in "CP TT Tucker".split():
        raise (NotImplementedError())

    results = []

    rank = 1
    loss, num_params = np.infty, 0
    decomposition_algo = {"TT": TT, "Tucker": Tucker, "CP": CP}
    it = 0
    rank_increment = 1
    while (loss > loss_threshold) and (num_params < max_num_params):
        it += 1
        tensor, num_params = decomposition_algo[decomposition](target, int(rank))
        loss = l2_distance(target, tensor)
        results.append({"iter": it, "num_params": num_params, "loss": loss, "rank": int(rank)})
        print(rank, loss, num_params)
        rank += rank_increment

        rank_increment = rank_increment * rank_increment_factor
        if verbose:
            print(results[-1])

    return results
