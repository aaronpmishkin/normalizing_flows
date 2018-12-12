# @Author: aaronmishkin
# @Email:  amishkin@cs.ubc.ca

import math
import torch
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn as nn

import  bounded_flows.transformations.support_transformations                   as      ST
import  bounded_flows.transformations.flow_transformations                      as      FT
from    bounded_flows.models.normalizing_flow                                   import  NormalizingFlow, CouplingTransformLayer
import  bounded_flows.transformations.conditional_flow_transformations          as      FT
from    bounded_flows.models.conditional_normalizing_flow                       import  ConditionalNormalizingFlow, ConditionalCouplingTransformLayer
from    bounded_flows.models.multi_head_mlp                                     import  MultiHeadMLP
from    lib.models.mlp                                                          import  MLP
from    torch.nn.utils                                                          import  parameters_to_vector, vector_to_parameters


# Helper function for instatiating normalizing flows with real NVP transformation layers.
def real_nvp_flow(D, flow_layers, nn_layers, nn_act_func, support_transform="r", support_bounds=None):
    """ Instantiate a real NVP flow with scale and shift functions given by neural networks. """
    initial_dist = torch.distributions.MultivariateNormal(torch.zeros(D), torch.eye(D))

    params = []

    scaling = 1

    for layer in range(flow_layers):
        mask = random_mask(D)
        # scale and shift functions for the real NVP flow:
        s_fn = MLP(input_size=D, hidden_sizes=nn_layers[layer], output_size=D, act_func=nn_act_func)

        # for param in s_fn.parameters():
        #     param.data.uniform_(0, -10)

        t_fn = MLP(input_size=D, hidden_sizes=nn_layers[layer], output_size=D, act_func=nn_act_func)

        # for param in t_fn.parameters():
        #     param.data.uniform_(0, -10)


        params.append((mask, s_fn, t_fn))

    # Initialize the normalizing flow:
    model = NormalizingFlow(initial_dist,
                            dimension=D,
                            num_layers=flow_layers,
                            TransformLayer=CouplingTransformLayer,
                            transform_params=params,
                            support_transform=support_transform,
                            support_bounds=support_bounds)

    return model

# Helper function for instatiating normalizing flows with real NVP transformation layers.
# This can be improved by adding support for an embedding network for y.
def conditional_real_nvp_flow(D_y, D_x, flow_layers, nn_layers, nn_act_func, support_transform="r", support_bounds=None):
    """ Instantiate a real NVP flow with scale and shift functions given by neural networks. """
    initial_dist = torch.distributions.MultivariateNormal(torch.zeros(D_x), torch.eye(D_x))

    params = []

    for layer in range(flow_layers):
        mask = random_mask(D_x)
        # scale and shift functions for the real NVP flow:
        s_fn = MLP(input_size=D_x + D_y, hidden_sizes=nn_layers[layer], output_size=D_x, act_func=nn_act_func)
        t_fn = MLP(input_size=D_x + D_y, hidden_sizes=nn_layers[layer], output_size=D_x, act_func=nn_act_func)

        params.append((mask, s_fn, t_fn))

    # Initialize the normalizing flow:
    model = ConditionalNormalizingFlow(initial_dist,
                                       dimension=D_x + D_y,
                                       num_layers=flow_layers,
                                       TransformLayer=ConditionalCouplingTransformLayer,
                                       transform_params=params,
                                       support_transform=support_transform,
                                       support_bounds=support_bounds)

    return model


def random_mask(D):
    """ Generate a random mask """
    mask = torch.ones(D)
    index = torch.randperm(D-1)[0] + 1
    direction = torch.distributions.Bernoulli(torch.tensor([0.5])).sample()

    if direction == 1:
        mask[index:] = 0
    else:
        mask[0:index] = 0

    return mask























# Extremely Slow... trying a different approach...

class ConditionalNVPFlow(nn.Module):
    def __init__(self, D_y, D_x, conditional_nn_layers, flow_layers, flow_nn_layers, nn_act_func, support_transform="r", support_bounds=None):
        """ Instantiate a real NVP flow with scale and shift functions given by a parameterizing neural network. """
        super(type(self), self).__init__()
        params = []
        output_sizes = {}

        for layer in range(flow_layers):


            mask = random_mask(D_x)
            # scale and shift functions for the real NVP flow:
            s_fn = MLP(input_size=D_x, hidden_sizes=flow_nn_layers[layer], output_size=D_x, act_func=nn_act_func)
            t_fn = MLP(input_size=D_x, hidden_sizes=flow_nn_layers[layer], output_size=D_x, act_func=nn_act_func)

            # set the output sizes for the parameterizing neural network:
            n = parameters_to_vector(s_fn.parameters()).size()[0]
            output_sizes["layer_" + str(layer) + "_s_fn"] = n
            n = parameters_to_vector(s_fn.parameters()).size()[0]
            output_sizes["layer_" + str(layer) + "_t_fn"] = n

            params.append((mask, s_fn, t_fn))

        # define the parameterizing neural network:
        self.parameterizing_network = MultiHeadMLP(D_y, conditional_nn_layers, output_sizes, "relu")

        # register the parameters of the parameterizing network
        for i, param in enumerate(self.parameterizing_network.parameters()):
            self.register_parameter("network_param_" + str(i), param)


        initial_dist = torch.distributions.MultivariateNormal(torch.zeros(D_x), torch.eye(D_x))

        # Initialize the normalizing flow:
        self.flow = NormalizingFlow(initial_dist,
                                    dimension=D_x,
                                    num_layers=flow_layers,
                                    TransformLayer=CouplingTransformLayer,
                                    transform_params=params,
                                    support_transform=support_transform,
                                    support_bounds=support_bounds)


    def _assign_params_to_flow(self, index, flow_params_dict):
        """ Compute and assign the parameters of the normalzing flow using the parameterizing network. """
        # compute the parameters with a forward pass through the parameterizing neural network:
        # instatiate the normalizing flow:
        for i, transform_layer in enumerate(self.flow.transform_layers):
            # assign the outputs to the normalizing flow layers. "vector_to_parameters" performs the assignment.
            vector_to_parameters(flow_params_dict["layer_" + str(i) + "_s_fn"][index], transform_layer.s_fn.parameters())
            vector_to_parameters(flow_params_dict["layer_" + str(i) + "_t_fn"][index], transform_layer.t_fn.parameters())

    # Forward pass implements sampling given a set of observations y
    def forward(self, y, z=None, num_samples=1):
        # This is going to be very slow...
        flow_params_dict = self.parameterizing_network(y)
        outs = []
        log_probs = []
        for i, example in enumerate(y):
            self._assign_params_to_flow(i, flow_params_dict)
            # sample from the normalzing flow:
            out, log_prob = self.flow.forward(z=z[i], num_samples=num_samples)
            outs.append(out)
            log_probs.append(log_prob)

        outs = torch.stack(outs, dim=0)
        log_probs = torch.stack(log_probs, dim=0)

        return outs, log_probs

    # Inverse pass computes the log_prob of an example "out" given observations y
    def inverse(self, y, out):
        # This is going to be very slow...
        flow_params_dict = self.parameterizing_network(y)
        Zs = []
        log_probs = []
        for i, example in enumerate(y):
            self._assign_params_to_flow(i, flow_params_dict)
            # sample from the normalzing flow:
            z, log_prob = self.flow.inverse(out)
            Zs.append(z)
            log_probs.append(log_prob)

        Zs = torch.stack(Zs, dim=0)
        log_probs = torch.stack(log_probs, dim=0)

        return Zs, log_probs
