#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy, sys
import time
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import random
from pathlib import Path
import logging

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from options import args_parser
from update import LocalUpdate, test_inference_new_het_lt
from models import ResNet, ConvNext, RegNet, VGG, DenseNet
from utils import (
    get_dataset, 
    exp_details, 
    param_aggregation, 
    agg_func, 
    weighted_param_aggregation,
    param_aggregation_fednova,
    weight_fednova_param_aggregation
)


def FedProtoAdap_heter(
        args, 
        logger,
        train_dataset, 
        test_dataset, 
        user_groups, 
        user_groups_lt, 
        local_model_list
    ):
    summary_writer = SummaryWriter(
        '../tensorboard/' + args.dataset + args.log_name.split('.')[0] 
        + str(args.num_users) + 'u_' + str(args.rounds) + 'r'
    )

    idxs_users = user_groups.keys()
    global_protos = []  # Initialize empty global prototypes
    global_adapters = []  # Initialize empty global adapters

    # train_loss = []
    form='{desc:<5.5}{percentage:3.0f}%|{bar:20}{r_bar}'
    for round in tqdm(range(args.rounds), ascii=True, bar_format=form):
        local_weights, local_losses = [], []
        local_protos = {}
        local_adapters = {}
        client_data_sizes = {}
        for i, idx in enumerate(idxs_users):
            client_data_sizes[i] = len(user_groups[idx])

        print(f'\n | Global Training Round : {round + 1} |\n')

        local_steps = {}
        for idx, idx_u in enumerate(idxs_users):
            local_update = LocalUpdate(
                args=args, 
                logger=logger,
                dataset=train_dataset, 
                idxs=user_groups[idx_u]
            )

            w, loss, acc, val_loss, val_acc, adapters, protos, l_steps = local_update.train_local(
                idx=idx,
                global_protos=global_protos,
                global_adapter=global_adapters,
                model=copy.deepcopy(local_model_list[idx]), 
                global_round=round
            )

            local_steps[idx] = l_steps
            agg_user_protos = agg_func(protos)
            agg_user_adapters = agg_func(adapters)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos[idx] = agg_user_protos
            local_adapters[idx] = agg_user_adapters

            summary_writer.add_scalar(
                'Train/Loss/user' + str(idx + 1), loss['total'], 
                round
            )
            summary_writer.add_scalar(
                'Train/CE_Loss/user' + str(idx + 1), loss['class_loss'], 
                round
            )
            summary_writer.add_scalar(
                'Train/Adapter_Loss/user' + str(idx + 1), loss['adapter_reg'], 
                round
            )
            summary_writer.add_scalar(
                'Train/Proto_Loss/user' + str(idx + 1), loss['proto_reg'], 
                round
            )
            summary_writer.add_scalar(
                'Train/Acc/user' + str(idx + 1), acc, 
                round
            )
            summary_writer.add_scalar(
                'Valid/Loss/user' + str(idx + 1), val_loss, 
                round
            )
            summary_writer.add_scalar(
                'Valid/Acc/user' + str(idx + 1), val_acc, 
                round
            )

        # Aggregate global prototypes and adapters
        global_protos = param_aggregation(local_protos)
        if args.adapt_agg == 'simple_agg':
            global_adapters = param_aggregation(local_adapters)
        elif args.adapt_agg == 'weighted_agg':
            global_adapters = weighted_param_aggregation(
                local_adapters,
                client_data_sizes
            )
        elif args.adapt_agg == 'fednova_agg':
            global_adapters = param_aggregation_fednova(
                local_adapters, 
                local_steps
            )
        elif args.adapt_agg == 'fednova_weighted_agg':
            global_adapters = weight_fednova_param_aggregation(
                local_adapters,
                local_steps,
                client_data_sizes
            )

        # update global weights
        local_weights_list = local_weights
        for idx, _ in enumerate(idxs_users):
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model
        
    # saving the global model
    model_name = args.log_name.split('.')[0]
    directory = '../save_model/{}'.format(model_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for idx in range(args.num_users):
        torch.save(
            local_model_list[idx].state_dict(), 
            '../save_model/{}/{}_fedprotoadap_{}.pt'.format(
                model_name, args.dataset, idx
            )
        )

        torch.save(
            local_model_list[idx].prototype_layer.state_dict(),
            '../save_model/{}/{}_protos_{}.pt'.format(
                model_name, args.dataset, idx
            )
        )

        torch.save(
            local_model_list[idx].adapters.state_dict(),
            '../save_model/{}/{}_adapters_{}.pt'.format(
                model_name, args.dataset, idx
            )
        )

    acc_list, loss_list = test_inference_new_het_lt(
        args, 
        logger,
        local_model_list, 
        test_dataset, 
        user_groups_lt, 
    )

    logger.info('For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list), np.std(acc_list)))
    logger.info('For all users, mean of test loss is {:.5f}, std of test loss is {:.5f}'.format(np.mean(loss_list), np.std(loss_list)))

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    
    log_file = '../logs/' + args.log_name
    log_fh = logging.FileHandler(log_file)
    log_sh = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=logging.INFO, 
        format='%(message)s', 
        handlers=[log_fh, log_sh]
    )
    logger = logging.getLogger(__name__)

    exp_details(args, logger)

    # set random seeds
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        set_random_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    train_dataset, test_dataset, user_groups, user_groups_lt = get_dataset(args)

    # Build models
    local_model_list = []
    if args.mode == 'task_heter':
        for i in range(args.num_users):
            # if args.dataset in ['fundus', 'oct']:               
            local_model = ResNet(args)

            local_model.to(args.device)
            local_model.train()
            local_model_list.append(local_model)
    elif args.mode == 'model_heter':
        models = [ConvNext, RegNet, VGG, DenseNet]
        for i in range(args.num_users):
            model = models[i]
            local_model = model(args)
            local_model.to(args.device)
            local_model.train()
            local_model_list.append(local_model)
    else:
        exit('Error: unrecognized mode')

    FedProtoAdap_heter(
        args, 
        logger,
        train_dataset, 
        test_dataset, 
        user_groups, 
        user_groups_lt, 
        local_model_list
    )