import copy
import os

import torch
from tqdm.auto import tqdm

from utils import train


def find_best_edge(train_loader, val_loader, test_loader, model, criterion, opt):
    num_cores = model.tensor_net.num_cores
    allowed_edges = [(i, j) for i in range(num_cores) for j in range(i + 1, num_cores)]
    best_acc = 0.0

    for (i, j) in allowed_edges:
        current_model = copy.deepcopy(model)
        current_model.tensor_net.increase_rank(vertex1=i, vertex2=j, pad_noise=opt.pad_noise)
        optimizer = torch.optim.Adam(current_model.parameters(),
                                     lr=opt.learning_rate,
                                     weight_decay=opt.weight_decay,
                                     amsgrad=True)

        current_model, current_step_hist = train(train_loader, val_loader, test_loader,
                                                 current_model, criterion, optimizer, opt,
                                                 new_slice_only=True,
                                                 vertex1=i,
                                                 vertex2=j)

        if current_step_hist['val_acc_best'] > best_acc:
            best_acc = current_step_hist['val_acc_best']
            best_model = current_model
            best_edge = (i, j)
            find_edge_hist = current_step_hist

    return best_model, find_edge_hist, best_edge


def greedy(train_loader, val_loader, test_loader, model, criterion, opt):
    hist = []

    for step in tqdm(range(opt.steps)):
        if not step:
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=opt.learning_rate,
                                         weight_decay=opt.weight_decay,
                                         amsgrad=True)

            model, hist_0 = train(train_loader, val_loader, test_loader,
                                  model, criterion, optimizer, opt, new_slice_only=False,
                                  vertex1=None, vertex2=None)
            hist.append(hist_0)

        else:
            model, find_edge_hist, best_edge = find_best_edge(train_loader, val_loader, test_loader, model, criterion, opt)
            hist.append(find_edge_hist)

            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=opt.learning_rate,
                                         weight_decay=opt.weight_decay,
                                         amsgrad=True)

            model, current_step_hist = train(train_loader, val_loader, test_loader,
                                             model, criterion, optimizer, opt,
                                             new_slice_only=False,
                                             vertex1=best_edge[0],
                                             vertex2=best_edge[1])
            hist.append(current_step_hist)

        # Save
        if not step and not os.path.isdir(opt.save_dir):
            os.makedirs(opt.save_dir)
        if step % opt.save_freq == 0 or step == opt.steps - 1:
            torch.save(hist, os.path.join(opt.save_dir, opt.xp_name))

    return hist
