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
from bounded_flows.plotting.densities       import plot_density

from lib.optimizers.closure_factories       import  vi_closure_factory
import lib.metrics.metric_factory           as      metric_factory
from lib.experiments.printers               import  print_objective
from lib.experiments.evaluate_model         import  evaluate_model

############################################
### Inference on the Probability Simplex ###
############################################

# symmetric, unimodal dirichlet distribution
mixture = 0.5
prior = torch.distributions.Bernoulli(torch.tensor([mixture]))
dist_1 = torch.distributions.MultivariateNormal(torch.tensor([2,2.]), torch.eye(2)*0.5)
dist_2 = torch.distributions.MultivariateNormal(torch.tensor([-2,-2.]), torch.eye(2)*0.5)

def draw_samples(n_samples=1):
    indicators = prior.sample_n(n_samples)
    dist_1_samples = dist_1.sample_n(n_samples)
    dist_2_samples = dist_2.sample_n(n_samples)
    samples = dist_1_samples.mul(indicators) + dist_2_samples.mul(1-indicators)

    return samples

# def sample_n(n_samples=1):
#     indicators = prior.sample_n(n_samples)
#     print(indicators)
#     dist_1_samples = dist_1.sample_n(n_samples)
#     dist_2_samples = dist_2.sample_n(n_samples)
#     samples = dist_1_samples.mul(indicators) - dist_2_samples.mul(1-indicators)
#     return samples

# Model:
D = 2
flow_layers = 10
nn_layers = [[50, 50] for i in range(flow_layers)]
nn_act_func = "relu"

model = real_nvp_flow(D, flow_layers, nn_layers, nn_act_func)
# Training Parameters:
num_mc_samples = 500
num_epochs = 1000
learning_rate = 0.001

# Initialize the optimizer:
optimizer = Adam(model.parameters(), lr=learning_rate)

objective = forward_kl

#####################################
########### Training Loop ###########
#####################################

for epoch in range(num_epochs):
    # Set model in training mode
    model.train(True)

    # generate data from the model:
    X = draw_samples(num_mc_samples)

    closure = density_estimation_closure_factory(X, None, objective, model, None, optimizer, num_mc_samples)
    loss = optimizer.step(closure)

    model.train(False)
    # Evaluate model

    # Print progress
    print_objective(epoch, num_epochs, loss)

# Set model in test mode
model.train(False)

grid_spacing = 0.1
w1, w2 = np.mgrid[-10:10:grid_spacing, -10:10:grid_spacing]
W = np.squeeze(np.dstack((w1.reshape(w1.size), w2.reshape(w2.size))))
W_tensor = torch.tensor(W).to(torch.float)

density_fn = lambda y:  (mixture * torch.exp(dist_1.log_prob(y)) + (1-mixture) * torch.exp(dist_2.log_prob(y)))

density = density_fn(W_tensor)

density = density.reshape((w1.shape[0], w2.shape[0]))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

c1 = plot_density(ax1, w1, w2, density, title="Posterior", xlim=[-10,10], ylim=[-10,10])

z, log_probs = model.inverse(W_tensor)

nf_density = torch.exp(log_probs.reshape(w1.shape[0], w2.shape[0]))

contour = ax2.contourf(w1, w2, nf_density.detach().numpy(), levels=c1.levels)

plt.show()
