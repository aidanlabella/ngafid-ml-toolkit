import numpy as np
import os

training_data_labels = {}
testing_data_labels = {}

training_data = {}
testing_data = {}

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
    data[kvp[0]] = parse_file_data(root_dir, file)

def parse_file_label(file):
    fns = file.split('.')
    return (fns[0], fns[1])

def parse_file_data(root_dir, file):
    csv_buffer = open(root_dir + '/' + file)
    return  np.loadtxt(csv_buffer, delimiter=",")

def get_arrays_3d(data):
    arr = []
    for value in data.items():
        print(value)
        arr.append(value)

    return arr

def load_data(path):
    parse_csvs(path);
    train_inputs = []
    test_inputs = []

    arr = get_arrays_3d(training_data)

def main():
    print("Main fxn invoked")
    load_data('/mnt/ngafid/extracted_loci_events/sept_2022')

if __name__ == "__main__":
    main()
