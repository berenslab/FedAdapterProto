#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import eyepacs
import random

import numpy as np
import pandas as pd

def OCT_iid(dataset, num_users, train_ratio=0.8, phase='train'):
    """
    Sample I.I.D. client data from Fundus dataset and split into train and test sets
    :param dataset: The dataset object
    :param num_users: Number of users (clients)
    :param train_ratio: Proportion of data to use for training (default 80%)
    :param phase: 'train' or 'test' phase
    :return: Dictionary mapping user IDs to image indices
    """
    df = dataset.dataframe
    total_items = len(df)
    
    num_items = int(total_items / num_users)
    dict_users, all_idxs = {}, np.arange(total_items)

    for i in range(num_users):
        dict_users[i] = set(
            np.random.choice(
                all_idxs, 
                num_items,
                replace=False
            )
        )
        all_idxs = np.setdiff1d(all_idxs, list(dict_users[i]))

    # Step 2: Split train/test based on train_ratio
    dict_train, dict_test = {}, {}

    for user, indices in dict_users.items():
        indices = list(indices)
        np.random.shuffle(indices)
        split_idx = int(len(indices) * train_ratio)
        
        dict_train[user] = set(indices[:split_idx])
        dict_test[user] = set(indices[split_idx:])

    return dict_train if phase == 'train' else dict_test

def create_noniid_split(
        dataset, 
        num_clients=5, 
        split_type='random', 
        alpha=0.5, 
        phase='train'
    ):
    """
    Create non-IID split of data for federated learning
    
    Args:
        dataset: pandas object
        num_clients: number of clients (default=4)
        split_type: type of splitting strategy
            'even' - equal samples per client
            'random' - random uneven split
            'dirichlet' - Dirichlet distribution based split
        alpha: concentration parameter for Dirichlet distribution
            (lower alpha = more uneven split)
    
    Returns:
        dict_users: dictionary mapping client_id to data indices
        client_stats: dictionary with statistics about the split
    """

    labels = np.array(dataset.dataframe.label)
    num_samples = len(labels)
    indices = np.arange(num_samples)
    
    # Separate indices by class
    class_0_indices = indices[labels == 0]
    class_1_indices = indices[labels == 1]
    
    # Create balanced test set (last client)
    test_size = min(len(class_0_indices), len(class_1_indices)) // 5  # Ensure balanced classes
    test_class0 = np.random.choice(class_0_indices, test_size, replace=False)
    test_class1 = np.random.choice(class_1_indices, test_size, replace=False)
    
    # Remove test indices from available pools
    class_0_indices = np.setdiff1d(class_0_indices, test_class0)
    class_1_indices = np.setdiff1d(class_1_indices, test_class1)
    
    # Calculate minimum samples per training client (20% of remaining samples)
    remaining_samples = len(class_0_indices) + len(class_1_indices)
    min_samples = int(0.2 * remaining_samples)
    
    # Distribute remaining samples among training clients
    if split_type == 'even':
        samples_per_client = remaining_samples // (num_clients - 1)  # -1 for test client
        client_samples = [samples_per_client] * (num_clients - 1)
    elif split_type == 'dirichlet':
        proportions = np.random.dirichlet([alpha] * (num_clients - 1))
        
        # Adjust proportions to ensure minimum 20% constraint
        min_proportion = 0.2
        while np.any(proportions < min_proportion):
            excess_mask = proportions > min_proportion
            excess = proportions[excess_mask] - min_proportion
            proportions[excess_mask] = min_proportion
            
            remaining_mask = proportions == min_proportion
            if np.sum(~remaining_mask) > 0:
                proportions[~remaining_mask] += excess.sum() / np.sum(~remaining_mask)
        
        client_samples = [int(p * remaining_samples) for p in proportions]
        client_samples[-1] = remaining_samples - sum(client_samples[:-1])
    else:  # random split
        remaining = remaining_samples - (min_samples * (num_clients - 1))
        extra_samples = np.random.multinomial(remaining, [1/(num_clients-1)] * (num_clients-1))
        client_samples = [min_samples + extra for extra in extra_samples]
    
    # Initialize results
    dict_users = {}
    
    # Create non-IID splits for training clients
    for i in range(num_clients - 1):
        minority_fraction = random.uniform(0.25, 0.45)
        n_minority = int(client_samples[i] * minority_fraction)
        n_majority = client_samples[i] - n_minority
        
        if random.random() < 0.5:
            n_class0, n_class1 = n_minority, n_majority
        else:
            n_class0, n_class1 = n_majority, n_minority
            
        client_class0 = np.random.choice(class_0_indices, n_class0)
        client_class1 = np.random.choice(class_1_indices, n_class1)
        
        class_0_indices = np.setdiff1d(class_0_indices, client_class0)
        class_1_indices = np.setdiff1d(class_1_indices, client_class1)
        
        dict_users[i] = np.concatenate([client_class0, client_class1])
    
    dict_test = {'0': np.concatenate([test_class0, test_class1])}

    return dict_users if phase == 'train' else dict_test

# def fundus_iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from EyePACs dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(
#             np.random.choice(
#                 all_idxs, 
#                 num_items,
#                 replace=False
#             )
#         )
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users

def fundus_iid_old(dataset, num_users, train_ratio=0.8, phase='train'):
    """
    Sample I.I.D. client data from Fundus dataset and split into train and test sets
    :param dataset: The dataset object
    :param num_users: Number of users (clients)
    :param train_ratio: Proportion of data to use for training (default 80%)
    :param phase: 'train' or 'test' phase
    :return: Dictionary mapping user IDs to image indices
    """
    df = dataset.data_frame
    total_items = len(df)
    
    num_items = int(total_items / num_users)
    dict_users, all_idxs = {}, np.arange(total_items)

    for i in range(num_users):
        dict_users[i] = set(
            np.random.choice(
                all_idxs, 
                num_items,
                replace=False
            )
        )
        all_idxs = np.setdiff1d(all_idxs, list(dict_users[i]))

    # Step 2: Split train/test based on train_ratio
    dict_train, dict_test = {}, {}

    for user, indices in dict_users.items():
        indices = list(indices)
        np.random.shuffle(indices)
        split_idx = int(len(indices) * train_ratio)
        
        dict_train[user] = set(indices[:split_idx])
        dict_test[user] = set(indices[split_idx:])

    return dict_train if phase == 'train' else dict_test

def fundus_iid(dataset, num_users, phase='train'):
    """
    Sample I.I.D. client data from Fundus dataset
    :param dataset: The dataset object
    :param num_users: Number of users (clients)
    :param phase: 'train' or 'test' phase
    :return: Dictionary mapping user IDs to image indices
    """
    df = dataset.data_frame
    col = 'clinical_siteIdentifier'

    if phase == 'train':
        sites = df[col].value_counts().head(n=num_users).index

        df = df[df[col].isin(sites)].reset_index(drop=True)

        num_items = int(len(df) / num_users)
        dict_users, all_idxs = {}, np.arange(len(df))

        for i in range(num_users):
            dict_users[i] = set(
                np.random.choice(
                    all_idxs, 
                    num_items,
                    replace=False
                )
            )
            all_idxs = np.setdiff1d(all_idxs, list(dict_users[i]))
    else:
        sites = df[col].value_counts().iloc[num_users:num_users+1].index
        num_users = 2  # Keeping it consistent with non-IID function

        df = df[df[col].isin(sites)].reset_index(drop=True)
        dict_sites = df.groupby(col).apply(lambda x: x.index.tolist()).to_dict()
        dict_users = {item: np.array(dict_sites.get(item, [])) for item in sites}

    return dict_users


def fundus_noniid(dataset, num_users, phase='train'):
    """
    Sample non-I.I.D client data from Fundus dataset
    :param dataset:
    :param num_users:
    :return:
    """

    col = 'clinical_siteIdentifier'
    df = dataset.data_frame
    if phase == 'train':
        sites = df[col].value_counts().head(n=num_users).index
    else:
        # sites = df[col].value_counts().iloc[num_users:num_users+2].index
        sites = df[col].value_counts().iloc[num_users:num_users+1].index
        # sites = df[col].value_counts().head(n=2).index
        num_users = 2
    df = df[df[col].isin(sites)].reset_index(drop=True)
    dict_sites = df.groupby(col).apply(lambda x: x.index.tolist()).to_dict()
    dict_users = {item: np.array(dict_sites.get(item, [])) for item in sites}
    
    classes_list =  np.tile(range(0, 2), (num_users, 1)).tolist()
    return dict_users, classes_list


if __name__ == '__main__':
    path = "..."
    data = pd.read_csv(
        os.path.join(
            path, 
            "metadata_image_circular_crop.csv"
        ),
        low_memory=False
    )
    dataset = eyepacs.FundusDataset(
        data=data,
        root_dir="...", 
        transformations=None
    )
    num = 100
    d = fundus_iid(dataset, num)
    print(d)