# @Author: aaronmishkin
# @Email:  amishkin@cs.ubc.ca

import math
import torch
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import pickle

# library imports
from bounded_flows.models.real_nvp          import real_nvp_flow
from bounded_flows.metrics.inf_compilation  import forward_kl, density_estimation_closure_factory
from bounded_flows.plotting.densities       import plot_simplex_density
from bounded_flows.models.distributions     import BoundedNormal, Mixture

from lib.optimizers.closure_factories       import  vi_closure_factory
import lib.metrics.metric_factory           as      metric_factory
from lib.experiments.printers               import  print_objective
from lib.experiments.evaluate_model         import  evaluate_model

############################################
### Inference on the Probability Simplex ###
############################################

def run_simplex_experiment(seed=111):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # symmetric, unimodal dirichlet distribution
    mixture_components = torch.distributions.Categorical(torch.tensor([ 0.34, 0.33, 0.33]))

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
    nn_layers = [[50, 50, 50] for i in range(flow_layers)]
    model_2 = real_nvp_flow(D-1, flow_layers, nn_layers, nn_act_func, support_transform)
    flow_layers = 4
    nn_layers = [[50, 50, 50] for i in range(flow_layers)]
    model_4 = real_nvp_flow(D-1, flow_layers, nn_layers, nn_act_func, support_transform)
    flow_layers = 8
    nn_layers = [[50, 50, 50] for i in range(flow_layers)]
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
        model_2.train(True)
        model_4.train(True)
        model_8.train(True)

        # generate data from the model:
        X = draw_samples(num_mc_samples)

        closure_2 = density_estimation_closure_factory(X, None, objective, model_2, None, optimizer_2, num_mc_samples)
        closure_4 = density_estimation_closure_factory(X, None, objective, model_4, None, optimizer_4, num_mc_samples)
        closure_8 = density_estimation_closure_factory(X, None, objective, model_8, None, optimizer_8, num_mc_samples)
        loss_2 = optimizer_2.step(closure_2)
        loss_4 = optimizer_4.step(closure_4)
        loss_8 = optimizer_8.step(closure_8)

        model_2.train(False)
        model_4.train(False)
        model_8.train(False)
        # Evaluate model

        # Print progress
        print("2-Layers: ")
        print_objective(epoch, num_epochs, loss_2)
        print("4-Layers: ")
        print_objective(epoch, num_epochs, loss_4)
        print("8-Layers: ")
        print_objective(epoch, num_epochs, loss_8)

    # Set model in test mode
    model_2.train(False)
    model_4.train(False)
    model_8.train(False)


    mixture = Mixture(torch.tensor([0.34, 0.33, 0.33]), [dist_1, dist_2, dist_3])

    return mixture, model_2, model_4, model_8


result = run_simplex_experiment()

pickle_out = open("simplex_inference.pkl","wb")
pickle.dump(result, pickle_out)
pickle_out.close()
