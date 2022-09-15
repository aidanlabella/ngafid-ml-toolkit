#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader, TensorDataset
from resnet import ResNet
import torch
import numpy as np
import os

training_data_labels = {}
testing_data_labels = {}

training_data = np.array
testing_data = np.array

def parse_csvs(root_dir):
    for file in os.listdir(root_dir):
        if 'csv' in file:
            if 'test' in file:
                parse_file(root_dir, file, testing_data_labels, testing_data)
            elif 'train' in file:
                parse_file(root_dir, file, training_data_labels, training_data)

def parse_file(root_dir, file, labels, data):
    kvp = parse_file_label(file)
    labels[kvp[0]] = kvp[1]
    data = np.append(data, parse_file_data(root_dir, file))
    print(data)

def parse_file_label(file):
    fns = file.split('.')
    return (fns[0], fns[1])

def parse_file_data(root_dir, file):
    csv_buffer = open(root_dir + '/' + file)
    return np.array(np.loadtxt(csv_buffer, delimiter=","))

def load_data(path):
    parse_csvs(path);
    train_inputs = []
    test_inputs = []

    input_channels = 8
    mid_channels = 4

    n_classes = 3;

    train_input_labels = []
    print('shape')
    print(training_data)

    # training_data_arr = list(training_data.values())
    # for d2arr in training_data_arr:
        # train_inputs.append(torch.Tensor(d2arr))

    # testing_data_arr = list(testing_data.values())
    # for d2arr in testing_data_arr:
        # test_inputs.append(torch.Tensor(d2arr))
        # demo_tensor = torch.Tensor(d2arr)
    
    resnet = ResNet(in_channels=input_channels, mid_channels=mid_channels, n_classes=n_classes)
    print(train_inputs)
    print(test_inputs)
    print("done!")

    print(len(training_data[0][0]))
    # x_in = torch.cat(training_data, dim=1)

    x_in = torch.Tensor(training_data)
    print(x_in)
    x = resnet.forward(train_inputs)
    print(x)

    return resnet

def main():
    classifier = load_data('/mnt/ngafid/extracted_loci_events/sept_2022')

if __name__ == "__main__":
    main()
