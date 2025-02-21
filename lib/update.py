#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torchvision import transforms
from torchsampler import ImbalancedDatasetSampler


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        
        return image, label

    def get_labels(self):
        parent_dataset_labels = self.dataset.get_labels()
        return parent_dataset_labels[self.idxs]


class LocalUpdate(object):
    def __init__(self, args, logger, dataset, idxs):
        self.args = args
        self.logger = logger
        self.random_state = args.seed
        self.trainloader, self.validloader, weights = self.train_val(
            dataset, 
            list(idxs)
        )
        self.device = args.device
        # self.tr_criterion = nn.CrossEntropyLoss(weight=weights).to(self.device)
        self.tr_criterion = nn.CrossEntropyLoss().to(self.device)
        self.ev_criterion = nn.CrossEntropyLoss().to(self.device)
        self.mse_criterion = nn.MSELoss().to(self.device)

    def train_val(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """

        idxs_train, idxs_valid = train_test_split(
            idxs,
            test_size=0.2,
            random_state=self.random_state,
            shuffle=True
        )

        tr_dataset = copy.deepcopy(dataset)
        val_dataset = copy.deepcopy(dataset)

        train_transform = transforms.Compose([
            transforms.Resize(self.args.img_size),
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

        val_transform = transforms.Compose([
            transforms.Resize(self.args.img_size),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
            ]
        )

        tr_dataset.transformations = train_transform
        val_dataset.transformations = val_transform

        if not self.args.use_sampler:
            trainloader = DataLoader(
                DatasetSplit(tr_dataset, idxs_train),
                batch_size=self.args.local_bs, 
                shuffle=True, 
                num_workers=64,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2
            )

            validloader = DataLoader(
                DatasetSplit(val_dataset, idxs_valid),
                batch_size=self.args.local_bs, 
                shuffle=False, 
                num_workers=64,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2
            )

        elif self.args.use_sampler:

            trainloader = DataLoader(
                DatasetSplit(tr_dataset, idxs_train),
                batch_size=self.args.local_bs, 
                sampler=ImbalancedDatasetSampler(
                    DatasetSplit(tr_dataset, idxs_train)
                ),
                num_workers=64,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2
            )

            validloader = DataLoader(
                DatasetSplit(val_dataset, idxs_valid),
                batch_size=self.args.local_bs, 
                shuffle=False, 
                num_workers=64,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2
            )

        if self.args.dataset == 'fundus':
            if self.args.target == 'diabetes':
                weights = torch.tensor(
                    class_weight.compute_class_weight(
                        class_weight='balanced',
                        classes=np.unique(
                            trainloader.dataset.dataset.data_frame['diagnosis_image_dr_level'].iloc[idxs_train]
                        ),
                        y=trainloader.dataset.dataset.data_frame['diagnosis_image_dr_level'].iloc[idxs_train]
                    ),
                    dtype=torch.float
                )
            elif self.args.target == 'gender':
                weights = torch.tensor(
                    class_weight.compute_class_weight(
                        class_weight='balanced',
                        classes=np.unique(
                            trainloader.dataset.dataset.data_frame['patient_gender'].iloc[idxs_train]
                        ),
                        y=trainloader.dataset.dataset.data_frame['patient_gender'].iloc[idxs_train]
                    ),
                    dtype=torch.float
                )
        elif self.args.dataset == 'ERM':
            weights = torch.tensor(
                class_weight.compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(
                        trainloader.dataset.dataset.dataframe['label'].iloc[idxs_train]
                    ),
                    y=trainloader.dataset.dataset.dataframe['label'].iloc[idxs_train]
                ),
                dtype=torch.float
            )
        
        elif self.args.dataset == 'Kermany':
            weights = torch.tensor(
                class_weight.compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(
                        trainloader.dataset.dataset.dataframe['label'].iloc[idxs_train]
                    ),
                    y=trainloader.dataset.dataset.dataframe['label'].iloc[idxs_train]
                ),
                dtype=torch.float
            )

        return trainloader, validloader, weights
    
    def get_adapter_params(self):
        """If you only want to optimize adapter parameters."""
        return [p for n, p in self.model.named_parameters() 
                if 'adapters' in n and p.requires_grad]
    
    def get_prototype_params(self):
        """If you only want to optimize prototype parameters."""
        return [p for n, p in self.model.named_parameters() 
                if 'prototype_layer' in n and p.requires_grad]
    
    def compute_prototype_loss(
            self, 
            model,
            lbl,
            min_distances,
            use_l1_mask=False
        ):

        max_dist = (model.prototype_layer.prototype_shape[1]
                    * model.prototype_layer.prototype_shape[2]
                    * model.prototype_layer.prototype_shape[3])

        # prototypes_of_correct_class is a tensor of shape 
        # batch_size * num_prototypes
        # calculate cluster cost

        prototypes_of_correct_class = torch.t(
            model.prototype_layer.prototype_class_identity[:,lbl]
        ).to(self.device)
        inverted_distances, _ = torch.max(
            (max_dist - min_distances) * prototypes_of_correct_class, 
            dim=1
        )
        cluster_cost = torch.mean(max_dist - inverted_distances)

        # calculate separation cost
        prototypes_of_wrong_class = 1 - prototypes_of_correct_class
        inverted_distances_to_nontarget_prototypes, _ = \
            torch.max(
                (max_dist - min_distances) * prototypes_of_wrong_class, 
                dim=1
            )
        separation_cost = torch.mean(
            max_dist - inverted_distances_to_nontarget_prototypes
        )

        # calculate avg cluster cost
        avg_separation_cost = \
            torch.sum(
                min_distances * prototypes_of_wrong_class, 
                dim=1
            ) / torch.sum(prototypes_of_wrong_class, dim=1)
        avg_separation_cost = torch.mean(avg_separation_cost)
        
        if use_l1_mask:
            l1_mask = 1 - torch.t(
                model.prototype_layer.prototype_class_identity
            ).to(self.device)
            l1 = (
                model.prototype_layer.last_layer.weight * l1_mask
            ).norm(p=1)
        else:
            l1 = model.prototype_layer.last_layer.weight.norm(p=1) 
        
        total_loss = 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
        
        return total_loss
    
    def compute_adapter_loss(self, model):
        """
        Compute adapter loss
        """

        # Adapter regularization (L2 + orthogonality)
        adapter_reg = 0.0
        ortho_reg = 0.0
        for a in model.adapters:
            # L2 regularization
            adapter_reg += sum(
                torch.norm(p, p=2) for p in model.adapters[a].parameters()
            )
        
        # total loss
        total_loss = 1e-5 * adapter_reg #+ 1e-5 * ortho_reg

        return total_loss

    def train_local(
        self,
        idx,
        global_protos,
        global_adapter,
        model, 
        global_round,
    ):
        model.train()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                trainable_params, 
                lr=self.args.lr,
                momentum=0.5
            )
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.args.lr,
                weight_decay=5e-4
            )

        epoch_loss = {
            'total_loss': [], 
            'class_loss': [], 
            'adapter_reg': [],
            'proto_reg': [],
            'total': []
        }
        
        best_acc = 0.0
        total_steps = 0.0
        for iter in range(self.args.train_ep):
            batch_loss = {
                'total_loss': [], 
                'class_loss': [], 
                'adapter_reg': [],
                'proto_reg': [],
                'total': []
            }
            agg_adapters = {}
            agg_prototypes = {}

            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs, distances = model(images)
                    loss = self.tr_criterion(outputs, labels)

                    adapter_reg = self.compute_adapter_loss(model)
                    proto_reg = self.compute_prototype_loss(
                        model,
                        labels, 
                        distances, 
                        loss,
                    )

                    loss_mse_p = nn.MSELoss()
                    loss_proto_init = 0.0
                    if len(global_protos) > 0:
                        for n, i in model.prototype_layer.named_parameters():
                            if n != 'prototype_layer.ones':
                                if self.args.use_prox:
                                    loss_proto = (
                                        i - global_protos[
                                            'prototype_layer.' + n
                                        ].to(self.device)).norm(p=2)**2
                                else:
                                    loss_proto = loss_mse_p(
                                        i,
                                        global_protos[
                                            'prototype_layer.' + n
                                        ].to(self.device)
                                    )
                                loss_proto_init += (
                                    self.args.prox / 2.0
                                ) * loss_proto
                    else:
                        loss_proto_init = 0.0

                    loss_mse_a = nn.MSELoss()
                    loss_adapter_init = 0.0
                    if len(global_adapter) > 0:
                        for n, p in model.adapters.named_parameters():
                            if self.args.use_prox:
                                loss_adapter = (
                                    p - global_adapter[n].to(self.device)
                                ).norm(p=2)**2
                            else:
                                loss_adapter = loss_mse_a(
                                    p,
                                    global_adapter[n].to(self.device)
                                )
                            loss_adapter_init += (
                                self.args.prox / 2.0
                            ) * loss_adapter
                    else:
                        loss_adapter_init = 0.0

                    total_loss = (
                        loss +
                        adapter_reg +
                        proto_reg +
                        loss_proto_init +
                        loss_adapter_init
                    )

                    total_loss.backward()
                    optimizer.step()
                    total_steps += 1

                # get keys of adapter and prototype parameters
                adapter_keys = [".".join(k.split('.')[1:]) for k in model.state_dict().keys() if 'adapter' in k]
                proto_keys = [k for k in model.state_dict().keys() if 'prototype' in k]
                
                # aggregate adapter and prototype parameters
                parent = 'adapters.'
                for i in adapter_keys:
                    if adapter_keys in list(agg_adapters.keys()):
                        agg_adapters[i].append(
                            model.state_dict()[parent + i]
                        )
                    else:
                        agg_adapters[i] = [model.state_dict()[parent + i]]

                for i in proto_keys:
                    if i in list(agg_prototypes.keys()):
                        agg_prototypes[i].append(
                            model.state_dict()[i]
                        )
                    else:
                        agg_prototypes[i] = [model.state_dict()[i]]                

                _, y_hat = outputs.max(1)
                acc = torch.eq(y_hat, labels.squeeze()).float().mean().item()

                if self.args.verbose and (batch_idx % 10 == 0):
                    self.logger.info(
                        '| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                            global_round, idx, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100. * batch_idx / len(self.trainloader),
                            total_loss.item(),
                            acc)
                    )
                
                batch_loss['total'].append(total_loss.item())
                batch_loss['class_loss'].append(loss.item())
                batch_loss['adapter_reg'].append(adapter_reg.item())
                batch_loss['proto_reg'].append(proto_reg.item())

            epoch_loss['total'].append(sum(batch_loss['total'])/len(batch_loss['total']))
            epoch_loss['class_loss'].append(sum(batch_loss['class_loss'])/len(batch_loss['class_loss']))
            epoch_loss['adapter_reg'].append(sum(batch_loss['adapter_reg'])/len(batch_loss['adapter_reg']))
            epoch_loss['proto_reg'].append(sum(batch_loss['proto_reg'])/len(batch_loss['proto_reg']))
            
            val_loss, val_acc = self.validate_local(
                idx, 
                model
            )

            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # Final epoch averages
        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['class_loss'] = sum(epoch_loss['class_loss']) / len(epoch_loss['class_loss'])
        epoch_loss['adapter_reg'] = sum(epoch_loss['adapter_reg']) / len(epoch_loss['adapter_reg'])
        epoch_loss['proto_reg'] = sum(epoch_loss['proto_reg']) / len(epoch_loss['proto_reg'])

        model.load_state_dict(best_model_wts)
        
        return model.state_dict(), epoch_loss, acc, val_loss, val_acc, agg_adapters, agg_prototypes, total_steps

    def validate_local(
        self,
        idx,
        model,
    ):
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in self.validloader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            with torch.no_grad():    
                outputs, _ = model(images)

                loss = self.ev_criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)

                # Prediction and accuracy
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                total_correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total_samples += len(labels)

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        self.logger.info(f'| User: {idx} | Validation Loss: {avg_loss:.4f} | Validation Acc: {avg_accuracy:.4f}')

        model.train()
        return avg_loss, avg_accuracy

def test_inference_new_het_lt(
    args, 
    logger,
    local_model_list, 
    test_dataset, 
    user_groups_gt
):
    idx_users = user_groups_gt.keys()
    device = args.device
    criterion = nn.CrossEntropyLoss().to(device)
    total_correct = 0.0
    total_samples = 0.0
    total_loss = 0.0

    acc_list = []
    loss_list = []

    for idx in range(args.num_users):
        model = local_model_list[idx]
        model.to(device)

        testloader = DataLoader(
            DatasetSplit(test_dataset, list(user_groups_gt.values())[0]), 
            batch_size=args.local_bs,
            shuffle=False,
            num_workers=64,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

        model.eval()
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():    
                outputs, _ = model(images)

                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)

                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                total_correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total_samples += len(labels)

        avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        logger.info(f'| User: {idx} | Test Loss: {avg_loss:.4f} | Test Acc: {avg_acc:.4f}')

        acc_list.append(avg_acc)
        loss_list.append(avg_loss)

    return acc_list, loss_list