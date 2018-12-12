# @Author: aaronmishkin
# @Email:  amishkin@cs.ubc.ca

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def coupling_transform(O, Z, mask, s_fn, t_fn):
    Z_masked = Z.mul(mask)
    # Add the observations (or observation embeddings) to the network inputs:
    network_inputs =  torch.cat([Z_masked, O], dim=1)

    s = F.softplus(s_fn(network_inputs))

    t = t_fn(network_inputs)
    Y = (Z.mul(s) + t).mul(1 - mask) + Z_masked

    return Y

def coupling_inverse_transform(O, Y, mask, s_fn, t_fn):
    Y_masked = Y.mul(mask)
    # Add the observations (or observation embeddings) to the network inputs:
    network_inputs =  torch.cat([Y_masked, O], dim=1)

    s = F.softplus(s_fn(network_inputs))

    t = t_fn(network_inputs)

    Z = (Y - t).div(s).mul(1 - mask) + Y_masked

    return Z

def coupling_log_det_jac(O, Z, mask, s_fn, t_fn):
    Z_masked = Z.mul(mask)
    # Add the observations (or observation embeddings) to the network inputs:
    network_inputs =  torch.cat([Z_masked, O], dim=1)
    s = F.softplus(s_fn(network_inputs))
    log_s = torch.log(s).mul(1 - mask)
    log_det = torch.sum(log_s, dim=1)

    return log_det
