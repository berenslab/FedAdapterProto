# Prototype-Guided Lightweight Adapters for Communication-Efficient and Generalisable Federated Learning

Implementation of the paper to be submitted to MICCAI 2025.

## Requirments
This code requires the following:

## Data Preparation
This is how to prepare the data for the experiments.

## Running the experiments

* To train the proposed algorithm
```
python federated_main.py --mode task_heter --dataset fundus --num_classes 2 --num_users 4 --rounds 50 --local_bs 16 --num_channels 3 --lr 0.0001 --optimizer adam --train_ep 1 --use_sampler 1 --model FedAdapterPrototype
```

You can change the default values of other parameters to simulate different conditions. Refer to the options section.

## Options
The default values for various paramters parsed to the experiment are given in ```options.py```.

## Citation
If you find this project helpful, please consider to cite the following paper:
