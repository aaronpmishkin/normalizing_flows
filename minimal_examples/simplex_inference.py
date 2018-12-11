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

############################################
### Inference on the Probability Simplex ###
############################################

# symmetric, unimodal dirichlet distribution
mixture_components = torch.distributions.Categorical(torch.tensor([ 0.34, 0.33, 0.33]))
prior = torch.distributions.Bernoulli(torch.tensor([mixture]))
dist_1 = torch.distributions.Dirichlet(torch.tensor([15., 4., 15.]))
dist_2 = torch.distributions.Dirichlet(torch.tensor([4., 15., 15.]))
dist_3 = torch.distributions.Dirichlet(torch.tensor([15., 15., 4.]))

def draw_samples(n_samples=1):
    indicators = mixture_components.sample_n(n_samples)
    dist_1_samples = dist_1.sample_n(n_samples)
    dist_2_samples = dist_2.sample_n(n_samples)
    dist_3_samples = dist_3.sample_n(n_samples)
    samples = dist_1_samples
    samples[indicators == 1, :] = dist_2_samples[indicators == 1, :]
    samples[indicators == 2, :] = dist_3_samples[indicators == 2, :]

    return samples

# Model:
D = 3
support_transform = "simplex"
nn_act_func = "relu"
flow_layers = 2
nn_layers = [[30, 30, 30] for i in range(flow_layers)]
model_2 = real_nvp_flow(D-1, flow_layers, nn_layers, nn_act_func, support_transform)
flow_layers = 4
nn_layers = [[30, 30, 30] for i in range(flow_layers)]
model_4 = real_nvp_flow(D-1, flow_layers, nn_layers, nn_act_func, support_transform)
flow_layers = 8
nn_layers = [[30, 30, 30] for i in range(flow_layers)]
model_8 = real_nvp_flow(D-1, flow_layers, nn_layers, nn_act_func, support_transform)

# model = real_nvp_flow(D, flow_layers, nn_layers, nn_act_func)
# Training Parameters:
num_mc_samples = 500
num_epochs = 1000
learning_rate = 0.0005

# Initialize the optimizer:
optimizer_2 = Adam(model_2.parameters(), lr=learning_rate)
optimizer_4 = Adam(model_4.parameters(), lr=learning_rate)
optimizer_8 = Adam(model_8.parameters(), lr=learning_rate)

objective = forward_kl

#####################################
########### Training Loop ###########
#####################################

for epoch in range(num_epochs):
    # Set model in training mode
    model.train(True)

    # generate data from the model:
    X = draw_samples(num_mc_samples)

    closure_2 = density_estimation_closure_factory(X, None, objective, model_2, None, optimizer_2, num_mc_samples)
    closure_4 = density_estimation_closure_factory(X, None, objective, model_4, None, optimizer_4, num_mc_samples)
    closure_8 = density_estimation_closure_factory(X, None, objective, model_8, None, optimizer_8, num_mc_samples)
    loss_2 = optimizer_2.step(closure_2)
    loss_4 = optimizer_4.step(closure_4)
    loss_8 = optimizer_8.step(closure_8)

    model.train(False)
    # Evaluate model

    # Print progress
    print("2-Layers: ")
    print_objective(epoch, num_epochs, loss_2)
    print("4-Layers: ")
    print_objective(epoch, num_epochs, loss_4)
    print("8-Layers: ")
    print_objective(epoch, num_epochs, loss_8)

# Set model in test mode
model.train(False)

samples, log_probs = model.forward(num_samples=100)

z, log_probs_inverse = model.inverse(samples)

samples_forward, log_probs = model.forward(z)

f = plt.figure(figsize=(24, 6))

plt.subplot(1, 4, 1)

density_fn = lambda y:  (0.34 * torch.exp(dist_1.log_prob(y)) + 0.33 * torch.exp(dist_2.log_prob(y)) + 0.33 * torch.exp(dist_3.log_prob(y)))

v = plot_simplex_density(density_fn)
plt.title("True Distribution")


plt.subplot(1, 4, 2)

density_fn = lambda y: torch.exp(model_2.inverse(y)[1])
plot_simplex_density(density_fn, levels=v.levels)

plt.title("2 Layers")

plt.subplot(1, 4, 3)

density_fn = lambda y: torch.exp(model_4.inverse(y)[1])
plot_simplex_density(density_fn, levels=v.levels)

plt.title("4 Layers")


plt.subplot(1, 4, 4)

density_fn = lambda y: torch.exp(model_8.inverse(y)[1])
plot_simplex_density(density_fn, levels=v.levels)

plt.title("8 Layers")

plt.show()
