#!/usr/bin/env python3
import torch
import sys, os
import numpy as np
import pandas as pd

training_data = []
test_data = []

def parse_args():
    dir_root = sys.argv[1];
    print("Retrieving CSV files from " + dir_root)
    return os.listdir(dir_root)

def get_files(root_dir):
    for file in root_dir:
        if ("test" in file):
            test_data.append(file)
            fname = str(file)
            print(fname.split('.'))
        elif ("train" in file):
            training_data.append(file)

def main():
    get_files(parse_args())

    print(str(training_data))
    print(str(test_data))

if __name__ == '__main__':
    main()
