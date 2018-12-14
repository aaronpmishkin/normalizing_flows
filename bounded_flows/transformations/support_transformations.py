# @Author: aaronmishkin
# @Email:  amishkin@cs.ubc.ca

import math
import torch
import torch.nn.functional as F

"""
Implements parameter-free, invertible, linear time transformations from R^N to A, where A can be:
    1) the set of positive real vectors R^N_+;
    2) a rectangular subset of R^N defined by { y | c <= y <= b };
    3) the probability simplex ∆^{N+1}.
"""

# IDENTITY

def identity_transform(z):
    """
    """

    return z


def identity_log_det_jac(z):
    """
    """
    return 0

# R PLUS

def r_plus_transform(z):
    return torch.exp(z)

def r_plus_inverse_transform(y):
    z = torch.log(y)
    return z

def r_plus_log_det_jac(z):
    log_det = torch.sum(z , dim=1)
    return log_det

# R BOUNDED

def r_bounded_transform(z, c, b):
    """An invertible transformation from R^N -> A, where A is a rectangular region of R^N given by
            A := { y | c <= y <= b }
       This is achieved using the sigmoid function as
            y = b * sigmoid(z) + c
   """

    return F.sigmoid(z).mul(b) + c

def r_bounded_inverse_transform(y, c, b):
    """ Inverse of r_bounded_transform. This is the inverse sigmoid with scale and location parameters."""

    return torch.log(y - c) - torch.log(c + b - y)

def r_bounded_log_det_jac(z, c, b):
    """Log-determinant of the Jacobian of the r_bounded_transform. This is the log-determinant of the softplus function with a scaling factor.
    """

    log_det = torch.sum(torch.log(torch.abs(b)) - z - 2 * F.softplus(-z, beta=1, threshold=math.inf), dim=1)
    return log_det

# PROBABILITY SIMPLEX

def simplex_transform(z):
    """ An invertible transformation from R^N -> ∆^{N+1} (the probability simplex) given by the inverse additive log-ratio (ALR) transform.
            y = exp(z) / (1 + sum(exp(z)))
    """
    exp_z = torch.exp(z)
    y_final = 1 / (1 + torch.sum(exp_z, dim=1))
    y_final = y_final.unsqueeze(dim=1)
    y = exp_z.mul(y_final)

    y = torch.cat([y, y_final], dim=1)

    return y

def simplex_inverse_transform(y):
    """ Inverse of simplex_transform. This is simply the additive log-ratio (ALR) transform.
    """
    y_final = y[:, -1].unsqueeze(dim=1)
    z = torch.log(y)[:, 0:-1] - torch.log(y_final)

    return z


def simplex_log_det_jac(z):
    """ Log-determinant of the Jacobian of simplex_inverse_transform. This is computed efficiently using the matrix determinant lemma.
    """
    D = z.size()[1]
    exp_z = torch.exp(z)
    alpha = 1 + torch.sum(exp_z, dim=1)
    uDv = torch.sum(exp_z, dim=1).div(alpha)
    log_det = torch.sum(z, dim=1) + torch.log(1 - uDv) - D * torch.log(alpha)
    return log_det




#
