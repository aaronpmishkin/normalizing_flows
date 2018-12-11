# @Author: aaronmishkin
# @Email:  amishkin@cs.ubc.ca

import math
import torch
import numpy as np


def forward_kl(log_probs):

    neg_expected_log_prob = -1 * torch.mean(log_probs)

    return neg_expected_log_prob


def density_estimation_closure_factory(x, y, objective, model, predict_fn, optimizer, num_samples):

    def closure():
        optimizer.zero_grad()
        z, log_probs = model.inverse(x)
        loss = objective(log_probs)
        loss.backward()
        return loss

    return closure
