import numpy as np
import os

training_data_labels = []
testing_data_labels = []

training_data = []
testing_data = []

def parse_csvs(root_dir):
    for file in os.listdir(root_dir):
        if 'csv' in file:
            if 'test' in file:
                parse_file(root_dir, file, testing_data_labels, testing_data)
            elif 'train' in file:
                parse_file(abs_path, training_data_labels, training_data)

def parse_file(root_dir, file, labels, data):
    kvp = parse_file_label(file)
    labels.append(kvp)
    print(kvp.items())
    [(k, v)] = kvp.items()
    print(k)
    data.append({k : parse_file_data(root_dir, file)})

def parse_file_label(file):
    fns = file.split('.')
    return {fns[0] : fns[1]}

def parse_file_data(root_dir, file):
    csv_buffer = open(root_dir + '/' + file)
    return  np.loadtxt(csv_buffer, delimiter=",")

