# @Author: aaronmishkin
# @Email:  amishkin@cs.ubc.ca

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import bounded_flows.root_finding.newton as newton

# Planar Flow Transformation

def planar_transform(Z, u, w, b):
    """ A planar flow as described in 'Variational Inference with Normalizing FLows, Rezende and Mohamed, ICML, 2015' """
    linear_combs = torch.mv(Z, w) + b

    return Z + u.mul(F.tanh(linear_combs).unsqueeze(1))


def planar_inverse_transform(Y, u, w, b):
    """ Inverse of a planar flow. The parameters of the planar flow must satisfy
            torch.dot(w, u) >= -1
        to be invertible. This function assumes that condition holds.
        Evaluating the inverse requires solving a non-linear root-finding problem.
    """
    Z = []
    wu = torch.dot(w, u)

    for y in Y:
        wy = torch.dot(w, y)
        fn, grad_fn = planar_root_fns(wu, wy, b)
        alpha = newton.newtons_method(fn, grad_fn)

        z_parallel = w.mul(alpha).div(torch.dot(w,w))
        z_orthogonal = y - z_parallel - u.mul(F.tanh(alpha + b))
        z = z_parallel + z_orthogonal

        Z.append(z)

    Z = torch.stack(Z)

    return Z

def planar_log_det_jac(Z, u, w, b):
    linear_combs = torch.mv(Z, w) + b

    linear_combs_exp = torch.exp(2 * linear_combs)
    h_prime = 4 * linear_combs_exp.div((linear_combs_exp + 1).pow(2))

    psi = torch.mul(w, h_prime.unsqueeze(1))
    det = 1 + torch.mv(psi, u)

    return torch.log(torch.abs(det))


def coupling_transform(Z, mask, s_fn, t_fn):
    Z_masked = Z.mul(mask)
    s = torch.exp(s_fn(Z_masked))

    t = t_fn(Z_masked)
    Y = (Z.mul(s) + t).mul(1 - mask) + Z_masked

    return Y

def coupling_inverse_transform(Y, mask, s_fn, t_fn):
    Y_masked = Y.mul(mask)

    s = torch.exp(-1 * s_fn(Y_masked))
    t = t_fn(Y_masked)

    Z = (Y - t).mul(s).mul(1 - mask) + Y_masked

    return Z

def coupling_log_det_jac(Z, mask, s_fn, t_fn):
    Z_masked = Z.mul(mask)
    log_s = s_fn(Z_masked).mul(1 - mask)
    log_det = torch.sum(log_s, dim=1)

    return log_det


# Helper Functions:
def planar_root_fns(wu, wy, b):

    fn = lambda alpha: torch.exp(2 * (alpha + b))*(alpha + wu - wy) - (wu + wy - alpha)
    grad_fn = lambda alpha: 1 + torch.exp(2 * (alpha + b)) * (1 + 2 * (alpha + wu - wy))

    return fn, grad_fn
