# @Author: aaronmishkin
# @Email:  amishkin@cs.ubc.ca

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module


#########################################################
### MultiLayer Perceptron with Multiple Output Layers ###
#########################################################

class MultiHeadMLP(nn.Module):
    """ A multi-layer perceptron (MLP) with multiple output layers.
        Each output layer can be used to parameterize the layer of a
        normalizing flow. """
    def __init__(self, input_size, hidden_sizes, output_sizes, act_func):
        super(type(self), self).__init__()
        self.input_size = input_size

        self.act = F.tanh if act_func == "tanh" else F.relu

        self.output_dictionary = {}
        self.output_layers = []

        if len(hidden_sizes) == 0:
            self.hidden_layers = []
            for key in output_sizes:
                outputs = nn.Linear(self.input_size, output_sizes[key])
                self.output_dictionary[key] = outputs
                self.output_layers = self.output_layers + [outputs]
        else:
            self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size) for in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
            for key in output_sizes:
                outputs = nn.Linear(hidden_sizes[-1], output_sizes[key])
                self.output_dictionary[key] = outputs
                self.output_layers = self.output_layers + [outputs]

        # register outputs layers:
        self.output_layers = nn.ModuleList(self.output_layers)

    def forward(self, x):

        x = x.view(-1, self.input_size)
        out = x

        for layer in self.hidden_layers:
            Z = layer(out)
            out = self.act(Z)

        network_outputs = {}
        for key in self.output_dictionary:
            Z = self.output_dictionary[key](out)
            network_outputs[key] = Z

        return network_outputs
