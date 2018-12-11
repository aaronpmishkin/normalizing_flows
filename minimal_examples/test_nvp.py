# @Author: aaronmishkin
# @Email:  amishkin@cs.ubc.ca


import math
import torch
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt

# library imports
from bounded_flows.models.real_nvp          import real_nvp_flow
from bounded_flows.metrics.inf_compilation  import forward_kl, density_estimation_closure_factory
from bounded_flows.plotting.densities       import plot_simplex_density

from lib.optimizers.closure_factories       import  vi_closure_factory
import lib.metrics.metric_factory           as      metric_factory
from lib.experiments.printers               import  print_objective
from lib.experiments.evaluate_model         import  evaluate_model


D = 2
flow_layers = 1
nn_layers = [[] for i in range(flow_layers)]
print(nn_layers)
support_transform = "simplex"
nn_act_func = "relu"

model = real_nvp_flow(D, flow_layers, nn_layers, nn_act_func)

print(list(model.parameters()))

x, log_prob = model.forward(z=torch.tensor([1, 2.]))
z, log_prob_inverse = model.inverse(x)


print("output", x, log_prob, z, log_prob_inverse)
