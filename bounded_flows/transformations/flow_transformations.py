# @Author: aaronmishkin
# @Date:   18-11-25
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-11-25

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import bounded_flows.root_finding.newton as newton

# Planar Flow Transformation

def planar_transform(z, u, w, b):
    """ A planar flow as described in 'Variational Inference with Normalizing FLows, Rezende and Mohamed, ICML, 2015' """
    return z + u.mul(F.tanh(torch.dot(w,z) + b))

def planar_inverse_transform(y, u, w, b):
    """ Inverse of a planar flow. The parameters of the planar flow must satisfy
            torch.dot(w, u) >= -1
        to be invertible. This function assumes that condition holds.
        Evaluating the inverse requires solving a non-linear root-finding problem.
    """
    wu = torch.dot(w, u)
    wy = torch.dot(w, y)

    fn, grad_fn = planar_root_fns(wu, wy, b)
    alpha = newton.newtons_method(fn, grad_fn)

    z_parallel = w.mul(alpha).div(torch.dot(w,w))
    z_orthogonal = y - z_parallel - u.mul(F.tanh(alpha + b))

    return z_parallel + z_orthogonal


def planar_log_det_jac(z, u, w, b):
    uz_b = torch.dot(u,z) + b
    exp_uz_b = torch.exp(uz_b)
    h_prime = 4 * exp_uz_b / ((exp_uz_b) + 1)**2

    psi = torch.mul(w, h_prime)
    det = 1 + torch.dot(u, psi)

    return torch.log(torch.abs(det))



# Radial Flow Transformation

def radial_transform(z):
    pass

def radial_inverse_transform(z):
    pass

def radial_log_det_jac(z):
    pass




# Helper Functions:

def planar_root_fns(wu, wy, b):

    fn = lambda alpha: torch.exp(2 * (alpha + b))*(alpha + wu - wy) - (wu + wy - alpha)
    grad_fn = lambda alpha: 1 + torch.exp(2 * (alpha + b)) * (1 + 2 * (alpha + wu - wy))

    return fn, grad_fn
