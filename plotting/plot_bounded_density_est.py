# @Author: aaronmishkin
# @Email:  amishkin@cs.ubc.ca


import math
import torch
from torch.optim import Adam
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle

# library imports
from bounded_flows.models.real_nvp          import real_nvp_flow
from bounded_flows.metrics.inf_compilation  import forward_kl, density_estimation_closure_factory
from bounded_flows.plotting.densities       import plot_simplex_density
import bounded_flows.transformations.support_transformations as ST

from lib.optimizers.closure_factories       import  vi_closure_factory
import lib.metrics.metric_factory           as      metric_factory
from lib.experiments.printers               import  print_objective
from lib.experiments.evaluate_model         import  evaluate_model


matplotlib.rcParams.update({'font.size': 14})
f = plt.figure(figsize=(12, 9))

#############################
### Plot R Plus Inference ###
#############################

pickle_in = open("saved_models/r_plus_inference.pkl","rb")
mixture, model_2, model_4, model_8 = pickle.load(pickle_in)

grid_spacing = 0.1
w1, w2 = np.mgrid[-2.25:25:grid_spacing, -2.25:25:grid_spacing]
W = np.squeeze(np.dstack((w1.reshape(w1.size), w2.reshape(w2.size))))
W_tensor = torch.tensor(W).to(torch.float)

ax = plt.subplot(3, 4, 2)
ax.set_facecolor('#E9BCAE')

Z, log_probs = model_2.inverse(W_tensor)
nf_density = torch.exp(log_probs.reshape(w1.shape[0], w2.shape[0]))
v = contour = plt.contourf(w1, w2, nf_density.detach().numpy(), 200)
plt.title("K = 2")

cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])
cur_axes.spines["top"].set_visible(False)
cur_axes.spines["right"].set_visible(False)
cur_axes.spines["bottom"].set_visible(False)
cur_axes.spines["left"].set_visible(False)

ax = plt.subplot(3, 4, 1)
ax.set_facecolor('#E9BCAE')

log_probs = mixture.log_prob(W_tensor)
nf_density = torch.exp(log_probs.reshape(w1.shape[0], w2.shape[0]))
v = contour = plt.contourf(w1, w2, nf_density.detach().numpy(), 200)
plt.title("True Density")
ax.set_ylabel("Non-Negative", size='large')

cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])
cur_axes.spines["top"].set_visible(False)
cur_axes.spines["right"].set_visible(False)
cur_axes.spines["bottom"].set_visible(False)
cur_axes.spines["left"].set_visible(False)

ax = plt.subplot(3, 4, 3)
ax.set_facecolor('#E9BCAE')

Z, log_probs = model_4.inverse(W_tensor)
nf_density = torch.exp(log_probs.reshape(w1.shape[0], w2.shape[0]))
contour = plt.contourf(w1, w2, nf_density.detach().numpy(), levels=v.levels)
plt.title("K = 4")

cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])
cur_axes.spines["top"].set_visible(False)
cur_axes.spines["right"].set_visible(False)
cur_axes.spines["bottom"].set_visible(False)
cur_axes.spines["left"].set_visible(False)

ax = plt.subplot(3, 4, 4)
ax.set_facecolor('#E9BCAE')

Z, log_probs = model_8.inverse(W_tensor)
nf_density = torch.exp(log_probs.reshape(w1.shape[0], w2.shape[0]))
contour = plt.contourf(w1, w2, nf_density.detach().numpy(), levels=v.levels)
plt.title("K = 8")

cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])
cur_axes.spines["top"].set_visible(False)
cur_axes.spines["right"].set_visible(False)
cur_axes.spines["bottom"].set_visible(False)
cur_axes.spines["left"].set_visible(False)

################################
### Plot R Bounded Inference ###
################################

pickle_in = open("saved_models/r_bounded_inference.pkl","rb")
mixture, model_2, model_4, model_8 = pickle.load(pickle_in)

grid_spacing = 0.1
w1, w2 = np.mgrid[1:13:grid_spacing, 1:13:grid_spacing]
W = np.squeeze(np.dstack((w1.reshape(w1.size), w2.reshape(w2.size))))
W_tensor = torch.tensor(W).to(torch.float)

ax = plt.subplot(3, 4, 5)
ax.set_facecolor('#E9BCAE')
log_probs = mixture.log_prob(W_tensor)
nf_density = torch.exp(log_probs.reshape(w1.shape[0], w2.shape[0]))
v = contour = plt.contourf(w1, w2, nf_density.detach().numpy(), 200)
ax.set_ylabel("Bounded", size='large')
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])
cur_axes.spines["top"].set_visible(False)
cur_axes.spines["right"].set_visible(False)
cur_axes.spines["bottom"].set_visible(False)
cur_axes.spines["left"].set_visible(False)

ax = plt.subplot(3, 4, 6)
ax.set_facecolor('#E9BCAE')
Z, log_probs = model_2.inverse(W_tensor)
nf_density = torch.exp(log_probs.reshape(w1.shape[0], w2.shape[0]))
contour = plt.contourf(w1, w2, nf_density.detach().numpy(), levels=v.levels)
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])
cur_axes.spines["top"].set_visible(False)
cur_axes.spines["right"].set_visible(False)
cur_axes.spines["bottom"].set_visible(False)
cur_axes.spines["left"].set_visible(False)

ax = plt.subplot(3, 4, 7)
ax.set_facecolor('#E9BCAE')
Z, log_probs = model_4.inverse(W_tensor)
nf_density = torch.exp(log_probs.reshape(w1.shape[0], w2.shape[0]))
contour = plt.contourf(w1, w2, nf_density.detach().numpy(), levels=v.levels)

cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])
cur_axes.spines["top"].set_visible(False)
cur_axes.spines["right"].set_visible(False)
cur_axes.spines["bottom"].set_visible(False)
cur_axes.spines["left"].set_visible(False)

ax = plt.subplot(3, 4, 8)
ax.set_facecolor('#E9BCAE')

Z, log_probs = model_8.inverse(W_tensor)
nf_density = torch.exp(log_probs.reshape(w1.shape[0], w2.shape[0]))
contour = plt.contourf(w1, w2, nf_density.detach().numpy(), levels=v.levels)

cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])
cur_axes.spines["top"].set_visible(False)
cur_axes.spines["right"].set_visible(False)
cur_axes.spines["bottom"].set_visible(False)
cur_axes.spines["left"].set_visible(False)

##############################
### Plot Simplex Inference ###
##############################

# symmetric, unimodal dirichlet distribution
pickle_in = open("saved_models/simplex_inference.pkl","rb")
mixture, model_2, model_4, model_8 = pickle.load(pickle_in)
density_fn = lambda y: torch.exp(mixture.log_prob(y))

ax = plt.subplot(3, 4, 9)
ax.set_facecolor('#E9BCAE')

v = plot_simplex_density(density_fn)
ax.set_ylabel("Probability Simplex", size='large')
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])
cur_axes.spines["top"].set_visible(False)
cur_axes.spines["right"].set_visible(False)
cur_axes.spines["bottom"].set_visible(False)
cur_axes.spines["left"].set_visible(False)

ax = plt.subplot(3, 4, 10)

density_fn = lambda y: torch.exp(model_2.inverse(y)[1])
plot_simplex_density(density_fn, levels=v.levels)
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])
cur_axes.spines["top"].set_visible(False)
cur_axes.spines["right"].set_visible(False)
cur_axes.spines["bottom"].set_visible(False)
cur_axes.spines["left"].set_visible(False)

ax = plt.subplot(3, 4, 11)
ax.set_facecolor('#E9BCAE')
density_fn = lambda y: torch.exp(model_4.inverse(y)[1])
plot_simplex_density(density_fn, levels=v.levels)
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])
cur_axes.spines["top"].set_visible(False)
cur_axes.spines["right"].set_visible(False)
cur_axes.spines["bottom"].set_visible(False)
cur_axes.spines["left"].set_visible(False)

ax = plt.subplot(3, 4, 12)
ax.set_facecolor('#E9BCAE')

density_fn = lambda y: torch.exp(model_8.inverse(y)[1])
plot_simplex_density(density_fn, levels=v.levels)
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])
cur_axes.spines["top"].set_visible(False)
cur_axes.spines["right"].set_visible(False)
cur_axes.spines["bottom"].set_visible(False)
cur_axes.spines["left"].set_visible(False)

f.tight_layout()

# Save figure with several different qualities:
f.savefig("bounded_density_est.png", bbox_inches='tight', dpi=250)
