# @Author: aaronmishkin
# @Date:   18-11-25
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-11-26

import math
import torch
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


import bounded_flows.data.logreg_synthetic as data
from bounded_flows.models.normalizing_flow import NormalizingFlow, CouplingTransformLayer
import bounded_flows.plotting.log_reg_example as vis

# VI_LIB
from lib.models.mlp import MLP
from lib.optimizers.closure_factories import vi_closure_factory
import lib.metrics.metric_factory as metric_factory
from lib.experiments.printers import print_progress, print_objective
from lib.experiments.evaluate_model import evaluate_model

# Prior Distribution:
prior_prec = 0.01
prior_dist = MultivariateNormal(loc=torch.zeros(2), covariance_matrix=torch.eye(2).div(prior_prec))
# Generate data:
X, y = data.logreg_synthetic_data()
y = y.to(torch.long)

N,D = X.size()

# Initialize the base distribution be standard normal:
initial_dist = MultivariateNormal(torch.zeros(2), torch.eye(2))

# Setup the neural networks parameterizing the scale-shift transformations

def generate_mask(D):
    """ Generate a random mask """
    mask = torch.ones(D)
    mask[math.floor(D / 2):] = 0
    mask = mask[torch.randperm(D)]
    return mask

num_layers = 10

params = []
for layer in range(num_layers):
    mask = generate_mask(D)
    # linear models
    s_fn = MLP(input_size=D, hidden_sizes=[5], output_size=D, act_func="relu")
    t_fn = MLP(input_size=D, hidden_sizes=[5], output_size=D, act_func="relu")

    params.append((mask, s_fn, t_fn))


# Initialize the normalizing flow:
model = NormalizingFlow(initial_dist, dimension=2, num_layers=num_layers, TransformLayer=CouplingTransformLayer, transform_params=params)

# Training Parameters:
num_mc_samples = 20
num_epochs = 1000
learning_rate = 0.01

# Initialize the optimizer:
optimizer = Adam(model.parameters(), lr=learning_rate)


# Set up prediction and loss functions:
def predict_fn(x, mc_samples):
    # Sample parameters from the flow:
    thetas = []
    theta, log_prob = model.forward(num_samples=mc_samples)
    logits = torch.matmul(theta, x.t())
    return logits

# Compute the KL divergence by sampling from q and p:
def kl_fn():
    theta, q_log_prob = model.forward(num_samples=num_mc_samples)
    p_log_prob = prior_dist.log_prob(theta)
    kl_divergence = torch.sum(q_log_prob - p_log_prob) / num_mc_samples

    return kl_divergence


objective = metric_factory.make_objective_closure("avneg_elbo_bernoulli", kl_fn, N)
metrics = metric_factory.BAYES_BINCLASS
train_metrics, test_metrics = metric_factory.make_metric_closures(metrics, kl_fn, N)

#####################################
########### Training Loop ###########
#####################################

metric_history = {}
for name in train_metrics.keys():
    metric_history[name] = []
for name in test_metrics.keys():
    metric_history[name] = []


for epoch in range(num_epochs):
    # Set model in training mode
    model.train(True)

    closure = vi_closure_factory(X, y, objective, model, predict_fn, optimizer, num_mc_samples)
    loss = optimizer.step(closure)

    model.train(False)
    # Evaluate model
    with torch.no_grad():
        metric_history = evaluate_model(predict_fn, train_metrics, test_metrics, metric_history, X, y, X, y, num_mc_samples, {'x':False, 'y':False})
    # Print progress
    print_progress(epoch, num_epochs, metric_history)

# Set model in test mode
model.train(False)


############################
###### PLOT POSTERIOR ######
############################

grid_spacing = 0.5
w1, w2 = np.mgrid[-30:30:grid_spacing, -30:30:grid_spacing]
# Compute all combinations of w1 and w2 values from the grid.
W = np.squeeze(np.dstack((w1.reshape(w1.size), w2.reshape(w2.size))))

X_np = X.numpy()
y_np = y.numpy()

# Plot "True" Posterior (computed numerically)

f = W @ X_np.T

log_prior = np.log(stats.multivariate_normal(mean=torch.zeros(2).numpy(), cov=torch.eye(2).div(prior_prec).numpy()).pdf(W))
prior = np.exp(log_prior)

log_like = np.sum(f * np.squeeze(y_np) - np.log(1+np.exp(f)), 1)
log_joint = log_like + log_prior
joint = np.exp(log_joint)

# Simple approximation of the Trapezoidal Rule.
log_marginal = np.log((grid_spacing**2) * np.sum(np.exp(log_joint)))

log_post = log_joint - log_marginal
post = np.exp(log_post)

# Reshape the densities back into a grid.
prior_density = prior.reshape((w1.shape[0], w2.shape[0]))
joint_density = joint.reshape((w1.shape[0], w2.shape[0]))
posterior_density = post.reshape((w1.shape[0], w2.shape[0]))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
c1 = vis.plot_density(ax3, w1, w2, posterior_density, title="Posterior", xlim=[-2,10], ylim=[-2,10])

# Plot normalizing flow posterior:

grid_spacing = 0.1
w1, w2 = np.mgrid[-2:10:grid_spacing, -2:10:grid_spacing]
# Compute all combinations of w1 and w2 values from the grid.
W = np.squeeze(np.dstack((w1.reshape(w1.size), w2.reshape(w2.size))))

W_tensor = torch.tensor(W).to(torch.float)

log_probs = []

z, log_probs = model.inverse(W_tensor)

nf_density = torch.exp(log_probs.reshape(w1.shape[0], w2.shape[0]))
#
# ax.set_title("Normalizing Flow")
contour = ax2.contourf(w1, w2, nf_density.detach().numpy())
# ax.set_xlabel("w_1"); ax2.set_ylabel("w_2")
# xlim=[-2,10]
# ylim=[-2,10]
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)

# Try Kernel density estimation:

# take a larger number of samples from the normalizing flow

samples, log_probs = model.forward(num_samples=10000)
samples = samples.detach().numpy().T

kernel = stats.gaussian_kde(samples)

Z = kernel(W.T)
print(Z.shape)
Z = Z.reshape((w1.shape[0], w2.shape[0]))
contour = ax1.contourf(w1, w2, Z)

plt.show()
