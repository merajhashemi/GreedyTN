"""
A module for the ncon function to contract multiple tensors.
References:
    - https://github.com/mhauru/ncon/blob/master/src/ncon/ncon.py
"""

from collections.abc import Iterable

import torch


def ncon(tensor_list, edge_list, order=None, forder=None, check_indices=True):
    """
    Args:
        tensor_list:    list of tensors to be contracted
        edge_list:      list of lists of indices e.g. edge_list[0] = [3, 4, -1]
                        labels the three indices of tensor tensor_list[0],
                        with -1 indicating an uncontracted index (open leg)
                        and 3 and 4 being the contracted indices.
        order:          list of contraction order (default [1, 2, 3, 4, ...])
        forder:         list of final ordering (default [-1, -2, ...])
        check_indices:  sanity  check the inputs (default True)

    Returns:
        A: contracted tensor
    """

    # We want to handle the tensors as a list, regardless of what kind
    # of iterable we are given. Inputs are assumed to be non-empty.
    tensor_list = list(tensor_list)
    edge_list = list(edge_list)
    if not isinstance(edge_list[0], Iterable):
        # edge_list is not a list of lists, so make it such.
        edge_list = [edge_list]
    else:
        edge_list = list(map(list, edge_list))

    if order is None:
        order = create_order(edge_list)
    if forder is None:
        forder = create_forder(edge_list)

    if check_indices:
        # Raise a RuntimeError if the indices are wrong.
        do_check_indices(tensor_list, edge_list, order, forder)

    # If the graph is disconnected, connect it with trivial indices that
    # will be contracted at the very end.
    connect_graph(tensor_list, edge_list, order)

    while len(order) > 0:
        tcon = get_tcon(edge_list, order[0])  # tcon = tensors to be contracted
        # Find the indices icon that are to be contracted.
        if len(tcon) == 1:
            tracing = True
            icon = [order[0]]
        else:
            tracing = False
            icon = get_icon(edge_list, tcon)
        # Position in tcon[0] and tcon[1] of indices to be contracted.
        # In the case of trace, pos2 = []
        pos1, pos2 = get_pos(edge_list, tcon, icon)
        if tracing:
            # Trace on a tensor
            new_A = trace(tensor_list[tcon[0]], axis1=pos1[0], axis2=pos1[1])
        else:
            # Contraction of 2 tensors
            new_A = con(tensor_list[tcon[0]], tensor_list[tcon[1]], (pos1, pos2))
        tensor_list.append(new_A)
        edge_list.append(find_newv(edge_list, tcon, icon))  # Add the edge_list for the new tensor
        for i in sorted(tcon, reverse=True):
            # Delete the contracted tensors and indices from the lists.
            # tcon is reverse sorted so that tensors are removed starting from
            # the end of tensor_list, otherwise the order would get messed.
            del tensor_list[i]
            del edge_list[i]
        order = renew_order(order, icon)  # Update order

    vlast = edge_list[0]
    A = tensor_list[0]
    A = permute_final(A, vlast, forder)
    return A


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def create_order(edge_list):
    """Identify all unique, positive indices and return them sorted."""
    flat_v = sum(edge_list, [])
    x = [i for i in flat_v if i > 0]
    # Converting to a set and back removes duplicates
    x = list(set(x))
    return sorted(x)


def create_forder(edge_list):
    """Identify all unique, negative indices and return them reverse sorted
    (-1 first).
    """
    flat_v = sum(edge_list, [])
    x = [i for i in flat_v if i < 0]
    # Converting to a set and back removes duplicates
    x = list(set(x))
    return sorted(x, reverse=True)


def connect_graph(tensor_list, edge_list, order):
    """Connect the graph of tensors to be contracted by trivial
    indices, if necessary. Add these trivial indices to the end of the
    contraction order.

    tensor_list, edge_list and order are modified in place.
    """
    # Build ccomponents, a list of the connected components of the graph,
    # where each component is represented by a a set of indices.
    unvisited = set(range(len(tensor_list)))
    visited = set()
    ccomponents = []
    while unvisited:
        component = set()
        next_visit = unvisited.pop()
        to_visit = {next_visit}
        while to_visit:
            i = to_visit.pop()
            unvisited.discard(i)
            component.add(i)
            visited.add(i)
            # Get the indices of tensors neighbouring tensor_list[i].
            i_inds = set(edge_list[i])
            neighs = (
                j for j, j_inds in enumerate(edge_list) if i_inds.intersection(j_inds)
            )
            for neigh in neighs:
                if neigh not in visited:
                    to_visit.add(neigh)
        ccomponents.append(component)
    # If there is more than one connected component, take one of them, a
    # take an arbitrary tensor (called c) out of it, and connect that
    # tensor with an arbitrary tensor (called d) from all the other
    # components using a trivial index.
    c = ccomponents.pop().pop()
    while ccomponents:
        d = ccomponents.pop().pop()
        A_c = tensor_list[c]
        A_d = tensor_list[d]
        c_axis = len(edge_list[c])
        d_axis = len(edge_list[d])
        tensor_list[c] = torch.unsqueeze(A_c, c_axis)
        tensor_list[d] = torch.unsqueeze(A_d, d_axis)
        try:
            dim_num = max(order) + 1
        except ValueError:
            dim_num = 1
        edge_list[c].append(dim_num)
        edge_list[d].append(dim_num)
        order.append(dim_num)
    return None


def get_tcon(edge_list, index):
    """Gets the list indices in tensor_list of the tensors that have index as their
    leg.
    """
    tcon = []
    for i, inds in enumerate(edge_list):
        if index in inds:
            tcon.append(i)
    l = len(tcon)
    # If check_indices is called and it does its work properly then these
    # checks should in fact be unnecessary.
    if l > 2:
        raise ValueError(
            "In ncon.get_tcon, more than two tensors share a contraction "
            "index."
        )
    elif l < 1:
        raise ValueError(
            "In ncon.get_tcon, less than one tensor share a contraction index."
        )
    elif l == 1:
        # The contraction is a trace.
        how_many = edge_list[tcon[0]].count(index)
        if how_many != 2:
            # Only one tensor has this index but it is not a trace because it
            # does not occur twice for that tensor.
            raise ValueError(
                "In ncon.get_tcon, a trace index is listed != 2 times for the "
                "same tensor."
            )
    return tcon


def get_icon(edge_list, tcon):
    """Returns a list of indices that are to be contracted when contractions
    between the two tensors numbered in tcon are contracted.
    """
    inds1 = edge_list[tcon[0]]
    inds2 = edge_list[tcon[1]]
    icon = set(inds1).intersection(inds2)
    icon = list(icon)
    return icon


def get_pos(edge_list, tcon, icon):
    """Get the positions of the indices icon in the list of legs the tensors
    tcon to be contracted.
    """
    pos1 = [[i for i, x in enumerate(edge_list[tcon[0]]) if x == e] for e in icon]
    pos1 = sum(pos1, [])
    if len(tcon) < 2:
        pos2 = []
    else:
        pos2 = [[i for i, x in enumerate(edge_list[tcon[1]]) if x == e] for e in icon]
        pos2 = sum(pos2, [])
    return pos1, pos2


def find_newv(edge_list, tcon, icon):
    """Find the list of indices for the new tensor after contraction of
    indices icon of the tensors tcon.
    """
    if len(tcon) == 2:
        newv = edge_list[tcon[0]] + edge_list[tcon[1]]
    else:
        newv = edge_list[tcon[0]]
    newv = [i for i in newv if i not in icon]
    return newv


def renew_order(order, icon):
    """Returns the new order with the contracted indices removed from it."""
    return [i for i in order if i not in icon]


def permute_final(A, edge_list, forder):
    """Returns the final tensor A with its legs permuted to the order given
    in forder.
    """
    perm = [edge_list.index(i) for i in forder]
    permuted = A.permute(tuple(perm))

    return permuted


def do_check_indices(tensor_list, edge_list, order, forder):
    """Check that
    1) the number of tensors in tensor_list matches the number of index lists in edge_list.
    2) every tensor is given the right number of indices.
    3) every contracted index is featured exactly twice and every free index
       exactly once.
    4) the dimensions of the two ends of each contracted index match.
    """

    # 1)
    if len(tensor_list) != len(edge_list):
        raise ValueError(
            (
                "In ncon.do_check_indices, the number of tensors %i"
                " does not match the number of index lists %i"
            )
            % (len(tensor_list), len(edge_list))
        )

    # 2)
    # Create a list of lists with the shapes of each A in tensor_list.
    shapes = list(map(lambda A: list(A.shape), tensor_list))
    for i, inds in enumerate(edge_list):
        if len(inds) != len(shapes[i]):
            raise ValueError(
                (
                    "In ncon.do_check_indices, len(edge_list[%i])=%i does not match "
                    "the numbers of indices of tensor_list[%i] = %i"
                )
                % (i, len(inds), i, len(shapes[i]))
            )

    # 3) and 4)
    # v_pairs = [[(0,0), (0,1), (0,2), ...], [(1,0), (1,1), (1,2), ...], ...]
    v_pairs = [[(i, j) for j in range(len(s))] for i, s in enumerate(edge_list)]
    v_pairs = sum(v_pairs, [])
    v_sum = sum(edge_list, [])
    # For t, o in zip(v_pairs, v_sum) t is the tuple of the number of
    # the tensor and the index and o is the contraction order of that
    # index. We group these tuples by the contraction order.
    order_groups = [
        [t for t, o in zip(v_pairs, v_sum) if o == e] for e in order
    ]
    forder_groups = [[1 for fo in v_sum if fo == e] for e in forder]
    for i, o in enumerate(order_groups):
        if len(o) != 2:
            raise ValueError(
                (
                    "In ncon.do_check_indices, the contracted index %i is not "
                    "featured exactly twice in edge_list."
                )
                % order[i]
            )
        else:
            A0, ind0 = o[0]
            A1, ind1 = o[1]
            try:
                compatible = tensor_list[A0].compatible_indices(tensor_list[A1], ind0, ind1)
            except AttributeError:
                compatible = tensor_list[A0].shape[ind0] == tensor_list[A1].shape[ind1]
            if not compatible:
                raise ValueError(
                    "In ncon.do_check_indices, for the contraction index %i, "
                    "the leg %i of tensor number %i and the leg %i of tensor "
                    "number %i are not compatible."
                    % (order[i], ind0, A0, ind1, A1)
                )
    for i, fo in enumerate(forder_groups):
        if len(fo) != 1:
            raise ValueError(
                (
                    "In ncon.do_check_indices, the free index %i is not "
                    "featured exactly once in edge_list."
                )
                % forder[i]
            )

    # All is well if we made it here.
    return True


######################################################################
# The following are simple wrappers around pytorch Tensor functions, #
# but may be replaced with fancier stuff later.                      #
######################################################################


def con(A, B, inds):
    return torch.tensordot(A, B, inds)


def trace(tensor, axis1=0, axis2=1):
    """Return summed entries along diagonals.
    If tensor is 2-D, the sum is over the
    diagonal of tensor with the given offset,
    i.e., the collection of elements of the form a[i, i+offset].
    If a has more than two dimensions, then the axes specified by
    axis1 and axis2 are used to determine the 2-D sub-array whose diagonal is
    summed.
    Args:
      tensor: A tensor.
      axis1, axis2: Axis to be used as the first/second axis of the 2D
                    sub-arrays from which the diagonals should be taken.
                    Defaults to second-last/last axis.
    Returns:
      array_of_diagonals: The batched summed diagonals.
    """
    return torch.sum(torch.diagonal(tensor, dim1=axis1, dim2=axis2), dim=-1)
