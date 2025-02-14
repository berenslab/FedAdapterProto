#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import pandas as pd
import torch.nn.functional as F
from torchvision import transforms
from sampling import fundus_iid, fundus_noniid, OCT_iid, create_noniid_split
import eyepacs
import ERM
import kermany


def get_dataset(args):
    """ 
    Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    data_dir = args.data_dir + args.dataset

    if args.dataset == 'fundus':
        data = pd.read_csv(
            "/gpfs01/berens/data/data/eyepacs/data_processed/metadata/metadata_image_circular_crop.csv",
            low_memory=False
        )
        data = data.loc[:, ~data.columns.str.match('Unnamed')]
        data = data[data['image_path'].str.contains('images-set-1')]
        data = data[(data['image_field'] == 'field 1')]
        quality = ['Adequate', 'Good', 'Excellent']
        data = data[data['session_image_quality'].isin(quality)]
        data['image_path'] = data['image_path'].str.replace(
            r'\.\w+$', 
            '.png', 
            regex=True
        )
        if args.target == 'diabetes':
            data = data.dropna(subset=['diagnosis_image_dr_level']).reset_index(drop=True)
        elif args.target == 'gender':
            data = data[data['patient_gender'].isin(['Female', 'Male'])]
            data = data.dropna(subset=['patient_gender']).reset_index(drop=True)
        
        train_data = data.copy()  # Create a separate copy for training
        test_data = data.copy()

        train_transform = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(
                brightness=0.05,
                hue=0.05,
                saturation=[0.8, 1.2],
                contrast=0.9
            ),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
            ]
        )

        test_transform = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
            ]
        )

        train_dataset = eyepacs.FundusDataset(
            data=train_data,
            tgt=args.target,
            root_dir='/gpfs01/berens/data/data/eyepacs/data_processed/images', 
            transformations=train_transform
        )
        test_dataset = eyepacs.FundusDataset(
            data=test_data,
            tgt=args.target,
            root_dir='/gpfs01/berens/data/data/eyepacs/data_processed/images', 
            transformations=test_transform
        )

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from EyePACS
            user_groups = fundus_iid(
                train_dataset, 
                args.num_users, 
                phase='train'
            )

            user_groups_lt = fundus_iid(
                test_dataset, 
                args.num_users, 
                phase='test'
            )
        else:
            # Sample Non-IID user data from EyePACS
            user_groups, _ = fundus_noniid(
                train_dataset,
                args.num_users,
                phase='train'
            )
            user_groups_lt, _ = fundus_noniid(
                test_dataset, 
                args.num_users, 
                phase='test'
            )
    elif args.dataset == 'ERM':
        train_df = pd.read_csv(
            '/gpfs01/berens/user/smensah/Postdoc/ERM/split_data/train.csv'
        )
        valid_df = pd.read_csv(
            '/gpfs01/berens/user/smensah/Postdoc/ERM/split_data/valid.csv'
        )
        test_df = pd.read_csv(
            '/gpfs01/berens/user/smensah/Postdoc/ERM/split_data/test.csv'
        )

        train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
        valid_df = valid_df.loc[:, ~valid_df.columns.str.contains('^Unnamed')]
        test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

        train_dataset = pd.concat([train_df, valid_df], ignore_index=True)
        train_dataset = train_dataset.reset_index(drop=True)
        test_dataset = test_df

        train_transform = transforms.Compose(
            [transforms.CenterCrop(args.img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation((-45, 45)),
            transforms.ColorJitter(
                brightness=(0.9, 1.1)
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.3, 0.3)
            ),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
            ]
        )

        test_transform = transforms.Compose(
            [transforms.CenterCrop((384, 384)),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )]
        )

        train_dataset = ERM.ERMDataset(
            train_dataset,
            '/gpfs01/berens/data/data/UKT/GeliskenJonas/',
            'label',
            binary=True,
            transformations=train_transform
        )

        test_dataset = ERM.ERMDataset(
            test_dataset,
            '/gpfs01/berens/data/data/UKT/GeliskenJonas/',
            'label',
            binary=True,
            transformations=test_transform
        )

        if args.iid:
            # Sample IID user data from EyePACS
            user_groups = OCT_iid(
                train_dataset, 
                args.num_users,
                phase='train'
            )

            user_groups_lt = OCT_iid(
                test_dataset, 
                args.num_users,
                phase='test'
            )

    elif args.dataset == 'Kermany':
        train_df = pd.read_csv(
            '/gpfs01/berens/data/data/kermani_oct/CellData/csv/train.csv'
        )
        valid_df = pd.read_csv(
            '/gpfs01/berens/data/data/kermani_oct/CellData/csv/val.csv'
        )
        test_df = pd.read_csv(
            '/gpfs01/berens/data/data/kermani_oct/CellData/csv/test.csv'
        )

        train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
        valid_df = valid_df.loc[:, ~valid_df.columns.str.contains('^Unnamed')]
        test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

        train_dataset = pd.concat([train_df, valid_df], ignore_index=True)
        train_dataset = train_dataset.reset_index(drop=True)
        train_dataset['p_fix'] = train_dataset['filename'].str.split('-').str[0] 
        test_dataset = train_dataset.copy()
        # test_dataset['p_fix'] = test_dataset['filename'].str.split('-').str[0] 

        train_transform = transforms.Compose(
            [
                transforms.CenterCrop(args.img_size),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation((-45, 45)),
                transforms.ColorJitter(
                    brightness=(0.9, 1.1)
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.3, 0.3)
                ),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                )
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.CenterCrop(args.img_size),
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                )
            ]
        )

        train_dataset = kermany.Kermany(
            train_dataset,
            '/gpfs01/berens/data/data/kermani_oct/CellData/OCT/train/',
            'label',
            transformations=train_transform
        )

        test_dataset = kermany.Kermany(
            test_dataset,
            '/gpfs01/berens/data/data/kermani_oct/CellData/OCT/train/',
            'label',
            transformations=test_transform
        )

        if args.iid:
            print('Implement IID sampling for Kermany dataset')
        else:
            user_groups = create_noniid_split(
                train_dataset, 
            )

            user_groups_lt = create_noniid_split(
                test_dataset, 
                phase='test'
            )

    else:
        raise NotImplementedError()

    return train_dataset, test_dataset, user_groups, user_groups_lt

def agg_func(params):
    """
    Returns the average of the weights.
    """

    for keys, values in params.items():
        if len(values) > 1:
            value = 0 * values[0].data
            for i in values:
                value += i.data
            params[keys] = value / len(values)
        else:
            params[keys] = values[0]

    return params

def param_aggregation(local_param_list):
    agg_params_label = dict()
    for idx in local_param_list:
        local_params = local_param_list[idx]
        for k in local_params.keys():
            if k in agg_params_label:
                agg_params_label[k].append(local_params[k])
            else:
                agg_params_label[k] = [local_params[k]]

    for [k, values] in agg_params_label.items():
        if len(values) > 1:
            value = 0 * values[0].data
            for i in values:
                value += i.data
            agg_params_label[k] = value / len(values)
        else:
            agg_params_label[k] = values[0].data

    return agg_params_label

def global_centroid_update(global_prototypes, client_prototypes_list):
    """
    global_prototypes: dict{k -> tensor (embedding_dim,)}
    client_prototypes_list: list of (client_protos, client_class_counts)
        where client_protos is dict{k -> prototype vector}
        and client_class_counts is dict{k -> int}
    Returns updated global_prototypes
    """
    from collections import defaultdict
    numerator = defaultdict(lambda: None)
    denominator = defaultdict(int)

    # 1. Accumulate
    for (client_protos, client_counts) in client_prototypes_list:
        for k, proto_vec in client_protos.items():
            count_k = client_counts.get(k, 0)
            if numerator[k] is None:
                numerator[k] = proto_vec * count_k
            else:
                numerator[k] += proto_vec * count_k
            denominator[k] += count_k
    
    # 2. Compute new global
    for k in denominator.keys():
        if denominator[k] > 0:
            global_prototypes[k] = numerator[k] / denominator[k]
        else:
            # If no samples, maybe keep old or set to zero
            pass
    
    return global_prototypes

def weighted_param_aggregation(local_param_list, client_data_sizes):
    """
    Perform weighted averaging of parameters.
    Args:
        local_param_list: dict of client_id -> dict(param_name -> tensor)
            E.g. {
            0: {'layer.weight': tensor, 'layer.bias': tensor, ...},
            1: {'layer.weight': tensor, 'layer.bias': tensor, ...},
            ...
            }
        client_data_sizes: dict of client_id -> number_of_samples
    Returns:
        agg_params_label: dict of param_name -> aggregated_tensor
    """
    # 1. Summation dictionary: for each param_name, 
    # we'll store (sum_of_weighted_params, sum_of_weights)
    sum_params = {}
    sum_weights = {}

    # 2. Aggregate across clients (total number of samples across all clients)
    total_samples = sum(client_data_sizes.values())  

    for client_id, local_params in local_param_list.items():
        # This client's weight
        client_weight = client_data_sizes[client_id]

        for param_name, param_tensor in local_params.items():
            if param_name not in sum_params:
                # Initialize accumulators
                sum_params[param_name] = torch.zeros_like(param_tensor.data)
                sum_weights[param_name] = 0.0

            # Accumulate weighted sums
            sum_params[param_name] += param_tensor.data * client_weight
            sum_weights[param_name] += client_weight

    # 3. Compute final averaged parameters
    agg_params_label = {}
    for param_name in sum_params.keys():
        # Weighted average
        agg_params_label[param_name] = sum_params[param_name] / sum_weights[param_name]

    return agg_params_label

def param_aggregation_fednova(local_param_list, local_steps):
    """
    Adapts your param_aggregation to do FedNova-style aggregation.
    
    Args:
        local_param_list: dict of {client_id: param_dict}, 
            where param_dict is {param_name: torch.Tensor}.
        local_steps: dict of {client_id: float}, 
            where each value is the number of local steps for that client.

    Returns:
        A dict of {param_name: torch.Tensor} for the new global parameters.
    """
    # 1) Pick a base set of parameters (e.g., first client in dictionary).
    base_idx = list(local_param_list.keys())[0]
    base_params = local_param_list[base_idx]

    # 2) Sum of all local steps
    total_steps = sum(local_steps.values())

    # 3) Create a result dict
    fednova_params = {}

    # 4) For each parameter key, compute the FedNova update
    for k in base_params.keys():
        # Get the base value
        base_val = base_params[k]
        # Initialize an accumulator for the weighted differences
        param_sum = torch.zeros_like(base_val)

        for idx, params in local_param_list.items():
            diff = params[k] - base_val
            param_sum += diff * local_steps[idx]

        fednova_params[k] = base_val + (param_sum / total_steps)

    return fednova_params

def weight_fednova_param_aggregation(
    local_param_list, 
    local_steps, 
    client_data_sizes
):
    """
    Weighted FedNova parameter aggregation.

    Args:
        local_param_list (dict): 
            {client_id: {param_name: tensor}}, the local parameters from each client.
        local_steps (dict): 
            {client_id: float}, number of local training steps for each client.
        client_data_sizes (dict): 
            {client_id: int}, number of samples (or any weighting factor) for each client.

    Returns:
        agg_params_label (dict): 
            {param_name: aggregated_tensor}, the newly aggregated global parameters.
    """
    # 1. Select a base model's parameters (e.g., the first client in the dictionary)
    #    In some implementations, you might want the previous global model instead.
    base_id = next(iter(local_param_list))
    base_params = local_param_list[base_id]

    # 2. Compute total weight = sum of (local_steps[i] * client_data_sizes[i])
    total_weight = 0.0
    for cid in local_param_list:
        total_weight += local_steps[cid] * client_data_sizes[cid]

    # 3. For each parameter key, accumulate weighted differences from the base
    agg_params_label = {}
    for param_name in base_params.keys():
        base_val = base_params[param_name]
        # Accumulate weighted differences
        diff_sum = torch.zeros_like(base_val)

        for cid, local_params in local_param_list.items():
            diff = local_params[param_name] - base_val
            diff_sum += diff * (local_steps[cid] * client_data_sizes[cid])

        # Add the normalized sum of differences back to the base parameter
        agg_params_label[param_name] = base_val + diff_sum / total_weight

    return agg_params_label


def exp_details(args, logger):
    logger.info(f'\nExperimental details:')
    logger.info(f'    Model     : {args.model}')
    logger.info(f'    Prototype_shape : {args.proto_shape}')
    logger.info(f'    Prototype activation : {args.proto_activation}') 
    logger.info(f'    Reduction factor : {args.red_factor}')
    logger.info(f'    Optimizer : {args.optimizer}')
    logger.info(f'    Learning  : {args.lr}')
    logger.info(f'    Global Rounds   : {args.rounds}\n')
    

    logger.info('    Federated parameters:')
    if args.iid:
        logger.info('    IID')
    else:
        logger.info('    Non-IID')
    logger.info(f'    Local Batch size   : {args.local_bs}')
    logger.info(f'    Local Epochs       : {args.train_ep}')
    if args.use_prox:
        logger.info(f'    Proximal term      : {args.prox}')
    logger.info(f'    Adapter Aggregation : {args.adapt_agg}')
    logger.info(f'    Number of users    : {args.num_users}\n')

    logger.info('    Dataset parameters:')
    logger.info(f'    Target             : {args.target}')
    logger.info(f'    Dataset            : {args.dataset}')
    logger.info(f'    Image Size         : {args.img_size}')
    if args.use_sampler:
        logger.info('    User Sampler       : True')
    else:
        logger.info('    User Sampler       : False')

    # more to come
    return