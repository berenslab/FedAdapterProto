#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument('--log_name', type=str, default='fedAdapterProto.log', 
                        required=True, help='log file name')

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--rounds', type=int, default=100,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=20,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.04,
                        help='the fraction of clients: C')
    parser.add_argument('--train_ep', type=int, default=1,
                        help="the number of local episodes: E")
    parser.add_argument('--local_bs', type=int, default=4,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--mode', type=str, default='task_heter', help="mode")
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--use_prox', type=int, default='1', 
                        help="whether to use proximal term")
    parser.add_argument('--prox', type=float, default=0.01,
                        help='proximal term')
    parser.add_argument('--proto_shape', type=tuple, default=(10, 512, 1, 1), 
                        help="shape of proto")
    parser.add_argument('--red_factor', type=int, default=16,
                        help="reduction factor of adapter")
    parser.add_argument('--init_weight_proto', type=int, default='0', 
                        help="whether to initialize proto weights with \
                            Kaiming Normal")
    parser.add_argument('--init_weight_adapter', type=int, default='0',
                        help="whether to initialize adapter weights with \
                            Kaiming Normal")
    parser.add_argument('--proto_activation', type=str, default='linear', 
                        help="prototype activation function")
    parser.add_argument('--adapt_agg', type=str, default='simple_agg', 
                        help="adapter aggregation", required=True)
    parser.add_argument('--fine_tune', type=int, default=1, help="fine tune \
                        the pretrained model")

    # other arguments
    parser.add_argument('--data_dir', type=str, default='../data/', 
                        help="directory of dataset")
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--target', type=str, default='diabetes', help="target \
                        for classification")
    parser.add_argument('--img_size', type=int, default=512, help="img size")
    parser.add_argument('--use_sampler', type=int, default=0, 
                        help="use sampler")
    parser.add_argument('--gpu', default=0, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1000, help='random seed')

    args = parser.parse_args()
    return args
