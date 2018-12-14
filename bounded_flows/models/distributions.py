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
import bounded_flows.transformations.support_transformations as ST

from lib.optimizers.closure_factories       import  vi_closure_factory
import lib.metrics.metric_factory           as      metric_factory
from lib.experiments.printers               import  print_objective
from lib.experiments.evaluate_model         import  evaluate_model


class MultivariateLogNormal():
    def __init__(self, loc, scale):
        self.normal_dist = torch.distributions.MultivariateNormal(loc, scale)

    def log_prob(self, W):
        log_W = torch.log(W)
        log_probs = self.normal_dist.log_prob(log_W)

        log_probs = log_probs - torch.sum(log_W, dim=1)

        return log_probs

    def sample_n(self, n):
        log_samples = self.normal_dist.sample_n(n)

        return torch.exp(log_samples)

class BinaryMixture():
    def __init__(self, p, dist_1, dist_2):
        self.p = p
        self.mixture_dist = torch.distributions.Bernoulli(p)
        self.dist_1 = dist_1
        self.dist_2 = dist_2

    def log_prob(self, W):
        probs = self.p * torch.exp(self.dist_1.log_prob(W)) + (1 - self.p) * torch.exp(self.dist_2.log_prob(W))

        return torch.log(probs)

    def sample_n(self, n):
        indicators = self.mixture_dist.sample_n(n).unsqueeze(dim=1)

        samples = self.dist_1.sample_n(n).mul(indicators) + self.dist_2.sample_n(n).mul(1-indicators)

        return samples

class BoundedNormal():
    """Defines a probability distribution by applying a scaled + shifted sigmoid to a normal distirbution. """
    def __init__(self, loc, scale, c, b):
        self.c = c
        self.b = b
        self.base_dist = torch.distributions.MultivariateNormal(loc, scale)

    def log_prob(self, W):
        inverse_W = ST.r_bounded_inverse_transform(W, self.c, self.b)
        log_probs = self.base_dist.log_prob(inverse_W)
        log_probs = log_probs - ST.r_bounded_log_det_jac(inverse_W, self.c, self.b)

        return log_probs

    def sample_n(self, n):
        samples = self.base_dist.sample_n(n)
        samples = ST.r_bounded_transform(samples, self.c, self.b)

        return samples

class Mixture():
    def __init__(self, proportions, dists):
        self.proportions = proportions
        self.dists = dists

        self.mixture_components = torch.distributions.Categorical(proportions)


    def log_prob(self, W):
        probs = 0
        for i, dist in enumerate(self.dists):
            probs = probs + self.proportions[i] * torch.exp(dist.log_prob(W))

        return torch.log(probs)

    def sample_n(self, n):
        indicators = self.mixture_components.sample_n(n)
        samples = None
        for i, dist in enumerate(self.dists):
            dist_samples = dist.sample_n(n)

            if samples is None:
                samples = torch.zeros_like(dist_samples)

            samples[indicators == i, :] = dist_samples[indicators == i, :]

        return samples
