#!/usr/bin/env python3
import torch
from torch import nn
import torch.nn.functional as F

from ngafid_data import parse_csvs, training_data, training_data_labels, testing_data, testing_data_labels

class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        return self.layers(x)

class ResNet(nn.Module):
    def __init__(self, in_channels, mid_channels: int = 64, n_classes):
        super().__init__()
        self.input_args = {
           'in_channels': in_channels,
           'num_pred_classes': n_classes
        }

        self.layers = nn.Sequential(*[
            ResNetBlock(in_channels=in_channels, out_channels=mid_channels),
            ResNetBlock(in_channels=mid_channels, out_channels=mid_channels * 2),
            ResNetBlock(in_channels=mid_channels * 2, out_channels=mid_channels * 2),

        ])

        self.final = nn.Linear(mid_channels * 2, num_pred_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.layers(x)
        return self.final(x.mean(dim=-1))

class ResNetBlock(nn.Module):

def __init__(self, in_channels: int, out_channels: int) -> None:
    super().__init__()

    channels = [in_channels, out_channels, out_channels, out_channels]
    kernel_sizes = [8, 5, 3]

    self.layers = nn.Sequential(*[
        ConvBlock(in_channels=channels[i], out_channels=channels[i + 1],
                  kernel_size=kernel_sizes[i], stride=1) for i in range(len(kernel_sizes))
    ])

    self.match_channels = False
    if in_channels != out_channels:
        self.match_channels = True
        self.residual = nn.Sequential(*[
            Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=1, stride=1),
            nn.BatchNorm1d(num_features=out_channels)
        ])

def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

    if self.match_channels:
        return self.layers(x) + self.residual(x)
    return self.layers(x)

parse_csvs('/mnt/ngafid/extracted_loci_events/sept_2022')
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)

print(training_data)


