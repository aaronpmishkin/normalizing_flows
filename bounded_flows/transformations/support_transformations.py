# @Author: aaronmishkin
# @Date:   18-11-25
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-11-25

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
    """An invertible transformation from R^N -> R^N_+ defined using the softplus function
        y = log(1 + exp(z))
    """
    # TODO: Investigate whether or not the function is properly invertible if it uses a linear approximation above 'threshold'.
    return F.softplus(z, beta=1, threshold=math.inf)

def r_plus_inverse_transform(y):
    """Inverse of r_plus_transform. This is only defined for y >= 0.
        z = log(exp(y) - 1)
    """
    return torch.log(torch.exp(y) - 1)

def r_plus_log_det_jac(z):
    """Log-determinant of the Jacobian of the r_plus_transform. This is identical to the log-determinant of the softplus function.
        log(det J(z)) = sum(log(sigma(z)))
    """
    # determinant is always positive in this case since sigma(z) \in (0,1)
    log_det = torch.sum(F.logsigmoid(z))
    return log_det

# R BOUNDED

def r_bounded_transform(z, c, b):
    """An invertible transformation from R^N -> A, where A is a rectangular region of R^N given by
            A := { y | c <= y <= b }
       This is achieved using the sigmoid function as
            y = b * sigma(z) + c
   """

    return F.sigmoid(z).mul(b) + c

def r_bounded_inverse_transform(y, c, b):
    """ Inverse of r_bounded_transform. This is the inverse sigmoid with scale and location parameters."""

    return torch.log(y - c) - torch.log(c + b - y)

def r_bounded_log_det_jac(z, c, b):
    """Log-determinant of the Jacobian of the r_bounded_transform. This is the log-determinant of the softplus function with a scaling factor.
    """

    # May not be able to simply take the log...
    log_det = torch.sum(torch.log(torch.abs(b)) - z - 2 * torch.softplus(-z, beta=1, threshold=math.inf))
    return log_det

# PROBABILITY SIMPLEX

def simplex_transform(z):
    """ An invertible transformation from R^N -> ∆^{N+1} (the probability simplex) given by the inverse additive log-ratio (ALR) transform.
            y = exp(z) / (1 + sum(exp(z)))
    """
    # TODO: decide whether or not to return the N+1 elements of the probability vector or just N elements (the last is implicitly defined).
    exp_z = torch.exp(z)
    return exp_z.div(1 + torch.sum(exp_z))

def simplex_inverse_transform(y):
    """ Inverse of simplex_transform. This is simply the additive log-ratio (ALR) transform.
    """
    # TODO: same as above.
    y_final = 1 - torch.sum(y)

    return torch.log(y) - torch.log(y_final)


def simplex_log_det_jac(z):
    """ Log-determinant of the Jacobian of simplex_inverse_transform. This is computed efficiently using the matrix determinant lemma.
    """
    exp_z = torch.exp(z)
    alpha = 1 + torch.sum(exp_z)
    uDv = torch.sum(exp_z) / alpha
    # The determinant is postive because all factors in the determinant's product are postive.
    log_det = torch.sum(z) + torch.log(1 + uDv) - torch.log(alpha)
    return log_det




#