#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from numpy.linalg.linalg import double
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from resnet import ResNet
import torch
import numpy as np
import os

training_data_labels = {}
testing_data_labels = {}

def get_files(root_dir):
    files = []
    for file in os.listdir(root_dir):
        if 'csv' in file:
            files.append(parse_file_data(root_dir, file))

    return files

def get_labels():
    return []

def parse_file(root_dir, file, labels):
    kvp = parse_file_label(file)
    labels[kvp[0]] = kvp[1]
    ndata = parse_file_data(root_dir, file)

    np.append(data, ndata)

def parse_file_label(file):
    fns = file.split('.')
    return (fns[0], fns[1])

def parse_file_data(root_dir, file):
    csv_buffer = open(root_dir + '/' + file)
    return np.array(np.loadtxt(csv_buffer, delimiter=",", dtype="double"))

def get_default_device():
    #Pick GPU if available, else CPU
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def create_classifier():
    input_channels = 8
    mid_channels = 4
    n_classes = 3;

    resnet = ResNet(in_channels=input_channels, mid_channels=mid_channels, n_classes=n_classes)

    return resnet

def train():
    class TSClassificationBase(nn.Module):
        def training_step(self, batch):
            images, labels = batch
            out = self(images)                  # Generate predictions
            loss = F.cross_entropy(out, labels) # Calculate loss
            return loss

        def validation_step(self, batch):
            images, labels = batch
            out = self(images)                    # Generate predictions
            loss = F.cross_entropy(out, labels)   # Calculate loss
            acc = accuracy(out, labels)           # Calculate accuracy
            return {'val_loss': loss.detach(), 'val_acc': acc}

        def validation_epoch_end(self, outputs):
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

        def epoch_end(self, epoch, result):
            print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

        def evaluate(model, val_loader):
            outputs = [model.validation_step(batch) for batch in val_loader]
            return model.validation_epoch_end(outputs)

        def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
            history = []
            optimizer = opt_func(model.parameters(), lr)
            for epoch in range(epochs):
                # Training Phase
                for batch in train_loader:
                    loss = model.training_step(batch)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                # Validation phase
                result = evaluate(model, val_loader)
                model.epoch_end(epoch, result)
                history.append(result)
            return history

    input_size = 3*32*32
    output_size = 10

def main():
    path = '/mnt/ngafid/extracted_loci_events/sept_2022'
    classifier = create_classifier()

    csv_files = get_files(path)

    for data in csv_files:
        data = np.reshape(data, (1, 8, 64))
        print(data.shape)

        x_in = torch.from_numpy(data).float()
        
        # TODO: need to create TorchDataset with the labels
        print(classifier.forward(x_in))

        classifier.forward(x_in)


if __name__ == "__main__":
    main()
