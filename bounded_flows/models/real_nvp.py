# @Author: aaronmishkin
# @Email:  amishkin@cs.ubc.ca

import math
import torch
import numpy as np


import bounded_flows.transformations.support_transformations    as      ST
import bounded_flows.transformations.flow_transformations       as      FT
from bounded_flows.models.normalizing_flow                      import NormalizingFlow, CouplingTransformLayer
from lib.models.mlp                                             import  MLP

def real_nvp_flow(D, flow_layers, nn_layers, nn_act_func, support_transform="r", support_bounds=None):
    """ Instantiate a real NVP flow with scale and shift functions given by neural networks. """
    initial_dist = torch.distributions.MultivariateNormal(torch.zeros(D), torch.eye(D))

    params = []

    for layer in range(flow_layers):
        mask = random_mask(D)
        # linear models
        s_fn = MLP(input_size=D, hidden_sizes=nn_layers[layer], output_size=D, act_func=nn_act_func)
        t_fn = MLP(input_size=D, hidden_sizes=nn_layers[layer], output_size=D, act_func=nn_act_func)

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
