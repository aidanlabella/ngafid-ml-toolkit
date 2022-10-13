import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class MCDCNN():
    def __init__(self, classes):
        super().__init__()
        self.classes = classes

    def create(self, n_cols, n_timesteps):
        self.conv1_layers = []

        for col in range(n_cols):



